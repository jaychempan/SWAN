import os,random,copy
import shutil
import torch
import argparse
import tensorboard_logger as tb_logger
import logging
import torch.distributed as dist
import utils
import data
import engine
from vocab import deserialize_vocab

# Hyper Parameters setting
def parser_options():
    parser = argparse.ArgumentParser()
    # training path setting
    parser.add_argument('-e', '--experiment_name', default='test', type=str, help="the file name of ckpt save")
    parser.add_argument('-m', '--model_name', default='SWAN', type=str, help="Model Name")
    parser.add_argument('--data_name', default='rsitmd', type=str, help="Dataset Name.(eg.rsitmd or rsicd)")
    parser.add_argument('--data_path', default='./data/', type=str, help=" Preprocessed data file path")
    parser.add_argument('--image_path', default='../rs_data/', type=str, help="remote images data path")
    parser.add_argument('--vocab_path', default='./vocab/', type=str, help="vocab data path")
    parser.add_argument('--resnet_ckpt', default='./layers/aid_28-rsp-resnet-50-ckpt.pth', type=str,
                        help="restnet pre model path.eg.(aid_28-rsp-resnet-50-ckpt.pth / resnet50-19c8e357.pth)")
    parser.add_argument('--resume', default=False, type=str,help="the pre-trained model path")
    parser.add_argument('--fix_data', default=False, action='store_true', help='Whether stratified sampling is used')
    parser.add_argument('--step_sample', default=False, action='store_true', help='Whether stratified sampling is used')
    parser.add_argument('--epochs', default=100, type=int, help="the epochs of train")
    parser.add_argument('--eval_step', default=1, type=int, help="the epochs of eval")
    parser.add_argument('--test_step', default=0, type=int, help="the epochs of test")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch train size")
    parser.add_argument('--batch_size_val', default=100, type=int, help="Batch val size")
    parser.add_argument('--shard_size', default=256, type=int, help="Batch shard size")
    parser.add_argument('--workers', default=3, type=int, help="the worker num of dataloader")
    parser.add_argument('-kf', '--k_fold_nums', default=1, type=int, help="the total num of k_flod")
    parser.add_argument('--k_fold_current_num', default=0, type=int, help="current num of k_fold")
    # Model parameter setting
    parser.add_argument('--embed_dim', default=512, type=int, help="the embedding's dim")
    parser.add_argument('--margin', default=0.2, type=float)
    parser.add_argument('--max_violation', default=False, action='store_true')
    parser.add_argument('--grad_clip', default=0.0, type=float)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--il_measure', default=False, help='Similarity measure used (cosine|l1|l2|msd)')
    # RNN/GRU model parameter
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.(e.g. 300, 512)')
    parser.add_argument('--use_bidirectional_rnn', default=True, type=str)
    parser.add_argument('--is_finetune', default=False, type=str,  help='Finetune resnet or not')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of GRU layers.')
    # GPU setting
    parser.add_argument('-g', '--gpuid', default=2, type=int, help="which gpu to use")
    parser.add_argument('--distributed', default=False, action='store_true', help='Whether to use parallel computing')
    parser.add_argument('--init_method', default='tcp://localhost:18888', help="init-method")
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')
    parser.add_argument('--world_size', default=2, type=int, help="world size")
    parser.add_argument('--use_mix_precision', default=False, action='store_true',
                        help="whether to use mix precision")
    # no set setting
    parser.add_argument('--logger_name', default='logs/', type=str, help="the path of logs")
    parser.add_argument('-p', '--ckpt_save_path', default='checkpoint_fix_data/', type=str,
                        help="the path of checkpoint save")
    parser.add_argument('--print_freq', default=10, type=int,  help="Print result frequency")
    parser.add_argument('--lr', default=0.0002, type=float, help="learning rate")
    parser.add_argument('--lr_update_epoch', default=20, type=int, help="the update epoch of learning rate")
    parser.add_argument('--lr_decay_param', default=0.7, type=float, help="the decay_param of learning rate")
    # SWAN 对比实验调参变量
    parser.add_argument('--sk_1', default=2, type=int)
    parser.add_argument('--sk_2', default=3, type=int)
    # SCAN 超参
    parser.add_argument('--cross_attn', default="t2i", help='t2i|i2t')
    parser.add_argument('--agg_func', default="LogSumExp", help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--lambda_lse', default=6., type=float, help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,  help='Attention softmax temperature.')
    parser.add_argument('--raw_feature_norm', default="softmax",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    args = parser.parse_args()
    # generate dataset path
    args.data_path = args.data_path + args.data_name + '_precomp/'
    args.image_path = args.image_path + args.data_name + '/images/'
    args.vocab_path = args.vocab_path + args.data_name + '_splits_vocab.json'
    # print hyperparameters
    print('-------------------------')
    print('# Hyper Parameters setting')
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('-------------------------')
    print('')
    return args


def main(args):
    # create random seed
    utils.setup_seed(args.seed)

    # init_process_group
    if args.distributed:
        # tcp model
        dist.init_process_group(backend='nccl', init_method=args.init_method,
                                rank=args.rank, world_size=args.world_size)

    # choose model
    if args.model_name == "SWAN":
        from layers import SWAN as models
    else:
        raise NotImplementedError

    # remove last train_info txt
    path_train_info = args.ckpt_save_path + args.model_name + "_" + args.data_name + ".txt"
    if os.path.exists(path_train_info):
        os.remove(path_train_info)
    # make ckpt save dir
    if not os.path.exists(args.ckpt_save_path) and args.rank == 0:
        os.makedirs(args.ckpt_save_path)

    # print & save args

    utils.log_to_txt(contexts='# Hyper Parameters setting', filename=path_train_info)
    utils.log_to_txt(contexts=args.__dict__, filename=path_train_info)
    utils.log_to_txt(contexts='-------------------------', filename=path_train_info)
    utils.log_to_txt(contexts='', filename=path_train_info)

    # make vocab
    vocab = deserialize_vocab(args.vocab_path)

    # Create dataset, model, criterion and optimizer
    train_loader, val_loader = data.get_loaders(args, vocab)
    if args.test_step:
        test_loader = data.get_test_loader(args, vocab)
    print("len of train_loader is {}, len of val_loader is {}".format(len(train_loader), len(val_loader)))

    model = models.factory(args,
                           vocab.word2idx,
                           cuda=True, 
                           data_parallel=args.distributed)

    # print & save model info
    if args.rank == 0:
        path_model_info = args.ckpt_save_path + args.model_name + "_info.txt"
        if os.path.exists(path_model_info):
            os.remove(path_model_info)
        log = open(path_model_info,mode="a",encoding="utf-8")
        print("Total Params: ", sum(p.numel() for p in model.parameters()))
        print("Total Requires_grad Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Total Params: ", sum(p.numel() for p in model.parameters()), file=log)
        print("Total Requires_grad Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), file=log)
        print("========================================================", file=log)
        print(model, file=log)
        print("========================================================", file=log)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # optionally resume from a checkpoint
    start_epoch = 0
    best_rsum = 0
    best_rsum_ = 0
    best_score = ""
    best_score_ = ""
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpuid))
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'], strict =False)
         
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
   
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(args.resume, start_epoch, best_rsum))
            rsum, all_scores = engine.validate(args, val_loader, model)
            print(all_scores)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Train the Model
    for epoch in range(start_epoch, args.epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        utils.adjust_learning_rate(args, optimizer, epoch)

        # # test validate
        # engine.validate(args, val_loader, model)

        # train for one epoch
        engine.train(args, train_loader, model, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_step == 0:
            rsum, all_scores = engine.validate(args, val_loader, model)

            is_best = rsum > best_rsum
            if is_best:
                best_score = all_scores
            best_rsum = max(rsum, best_rsum)

            if args.rank == 0:
                # save ckpt
                utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'best_rsum': best_rsum,
                        'args': args,
                        'Eiters': model.Eiters,
                    },
                    is_best,
                    filename='ckpt_{}_{}_{:.2f}.pth.tar'.format(args.model_name ,epoch, best_rsum),
                    prefix=args.ckpt_save_path,
                    model_name=args.model_name)
                print('')
                print("================ evaluate result on val set =====================")
                print("Current => [{}/{}] fold & [{}/{}] epochs"
                      .format(args.k_fold_current_num + 1, args.k_fold_nums, epoch + 1, args.epochs))
                print("Now val score:")
                print(all_scores)
                print("Best val score:")
                print(best_score)
                print("=================================================================")
                print('')

                utils.log_to_txt(contexts="",
                                 filename=path_train_info)
                utils.log_to_txt(contexts="================ evaluate on val set ============================",
                                 filename=path_train_info)
                utils.log_to_txt(contexts="Current => [{}/{}] fold & [{}/{}] epochs"
                                 .format(args.k_fold_current_num + 1, args.k_fold_nums, epoch + 1, args.epochs),
                                 filename=path_train_info)
                utils.log_to_txt("Now val score:",
                                 filename=path_train_info)
                utils.log_to_txt(contexts=all_scores, filename=path_train_info)
                utils.log_to_txt("Best val score:",
                                 filename=path_train_info)
                utils.log_to_txt(best_score, filename=path_train_info)
                utils.log_to_txt(contexts="=================================================================",
                                 filename=path_train_info)
                utils.log_to_txt(contexts="",
                                 filename=path_train_info)
        # evaluate on test set
        if args.test_step and (epoch + 1) % args.test_step == 0:
            rsum_, all_scores_ = engine.validate_test(args, test_loader, model)

            is_best_ = rsum_ > best_rsum_
            if is_best_:
                best_score_ = all_scores_
            best_rsum_ = max(rsum_, best_rsum_)

            if args.rank == 0:
                print('')
                print("================ evaluate result on test set =====================")
                print("Current => [{}/{}] fold & [{}/{}] epochs"
                      .format(args.k_fold_current_num + 1, args.k_fold_nums, epoch + 1, args.epochs))
                print("Now test score:")
                print(all_scores_)
                print("Best test score:")
                print(best_score_)
                print("=================================================================")
                print('')

                utils.log_to_txt(contexts="",
                                 filename=path_train_info)
                utils.log_to_txt(contexts="================ evaluate on test set ============================",
                                 filename=path_train_info)
                utils.log_to_txt(contexts="Current => [{}/{}] fold & [{}/{}] epochs"
                                 .format(args.k_fold_current_num + 1, args.k_fold_nums, epoch + 1, args.epochs),
                                 filename=path_train_info)
                utils.log_to_txt("Now test score:",
                                 filename=path_train_info)
                utils.log_to_txt(contexts=all_scores_,filename=path_train_info)
                utils.log_to_txt("Best test score:",
                                 filename=path_train_info)
                utils.log_to_txt(best_score_,filename=path_train_info)
                utils.log_to_txt(contexts="=================================================================",
                                 filename=path_train_info)
                utils.log_to_txt(contexts="",
                                 filename=path_train_info)

    if args.distributed:
        # destroy process
        dist.destroy_process_group()


def generate_random_samples(args):
    # load all anns
    caps = utils.load_from_txt(args.data_path+'train_caps.txt')
    fnames = utils.load_from_txt(args.data_path+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, args.data_path+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, args.data_path+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
            val_fnames.append(item[1])
    utils.log_to_txt(val_caps, args.data_path+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, args.data_path+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(args.data_path))

    ######################################################################################
    data_info_path = args.ckpt_save_path + 'data/'
    if os.path.exists(data_info_path):
        shutil.rmtree(data_info_path)
    if not os.path.exists(data_info_path) and args.rank == 0:
        os.makedirs(data_info_path)

    # cpoy tran & val set
    utils.log_to_txt(train_caps, data_info_path + 'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, data_info_path + 'train_filename_verify.txt',mode='w')
    utils.log_to_txt(val_caps, data_info_path + 'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, data_info_path + 'val_filename_verify.txt',mode='w')

    # vis & cal data set split
    utils.vis_cal_data_info(args, data_info_path, train_fnames, val_fnames)

    print("Copy random samples and Cal data info to {} complete.".format(args.ckpt_save_path))
    ######################################################################################

# stratified_random_samples

def generate_stratified_random_samples(args):
    # load all ans
    caps = utils.load_from_txt(args.data_path+'train_caps.txt')
    fnames = utils.load_from_txt(args.data_path+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)
    ff = [a[1] for a in all_infos]
    class_ = utils.gen_class_from_list(ff)
    cnt_cl = utils.cnt_class(class_)
    p = 0.8
    cnt_p = {}

    for i in cnt_cl.keys():
        cnt_p[i] = int(round(cnt_cl[i] * p))

    train_infos = []
    val_infos = []
    for i in range(len(all_infos)):
        if cnt_p[class_[i]] > 0:
            train_infos.append(all_infos[i])
            cnt_p[class_[i]] -= 1
        else:
            val_infos.append(all_infos[i])

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, args.data_path+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, args.data_path+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
            val_fnames.append(item[1])
    utils.log_to_txt(val_caps, args.data_path+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, args.data_path+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(args.data_path))
    ######################################################################################
    data_info_path = args.ckpt_save_path + 'data/'
    if os.path.exists(data_info_path):
        shutil.rmtree(data_info_path)
    if not os.path.exists(data_info_path) and args.rank == 0:
        os.makedirs(data_info_path)

    # cpoy tran & val set
    utils.log_to_txt(train_caps, data_info_path + 'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, data_info_path + 'train_filename_verify.txt',mode='w')
    utils.log_to_txt(val_caps, data_info_path + 'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, data_info_path + 'val_filename_verify.txt',mode='w')

    # vis & cal data set split
    utils.vis_cal_data_info(args, data_info_path, train_fnames, val_fnames)

    print("Copy random samples and Cal data info to {} complete.".format(args.ckpt_save_path))
    ######################################################################################


def update_options_savepath(args, k):
    args_new = copy.deepcopy(args)

    args_new.k_fold_current_num= k
    if args.k_fold_nums > 1:
        args_new.ckpt_save_path = args.ckpt_save_path + args.data_name + '/' + args.experiment_name + "/" + str(k) + "/"
    else:
        args_new.ckpt_save_path = args.ckpt_save_path + args.data_name + '/' + args.experiment_name + "/"
    return args_new


if __name__ == '__main__':

    args = parser_options()

    # make logger
    logger_path = args.ckpt_save_path + args.data_name + '/' + args.experiment_name + "/"
    tb_logger.configure(logger_path, flush_secs=5)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # k_fold verify
    for k in range(args.k_fold_nums):

        print("Start {}th fold, total {} flod".format(k + 1, args.k_fold_nums))

        # update save path

        args_new = update_options_savepath(args, k)

        # generate random train and val samples
        if not args.fix_data:
            if args_new.step_sample:
                generate_stratified_random_samples(args_new)
            else:
                generate_random_samples(args_new)
        else:
            print('==> This experiment uses fixed data partition <==')
            args_new.data_path = './fix_data/'+ args_new.data_name + '_precomp/'

        # run experiment
        main(args_new)
