import numpy as np
import os, copy
import torch
import argparse
import logging
import utils
import data
import engine
from vocab import deserialize_vocab

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    # training path setting
    parser.add_argument('-g', '--gpuid', default=1, type=int, help="which gpu to use")
    parser.add_argument('-e', '--experiment_name', default='SWAN', type=str, help="Model Name")
    parser.add_argument('-m', '--model_name', default='SWAN', type=str, help="Model Name")
    parser.add_argument('--ckpt_save_path', default='./checkpoint/', type=str,
                        help="the path of ckpt save.eg.(./checkpoint/ or ./checkpoint_fix_data/)")
    parser.add_argument('--data_name', default='rsicd', type=str, help="Dataset Name.(eg.rsitmd or rsicd)")
    parser.add_argument('--k_fold_nums', default=3, type=int, help="the total num of k_flod")
    parser.add_argument('--resnet_ckpt', default='./layers/aid_28-rsp-resnet-50-ckpt.pth', type=str,
                        help="restnet pre model path.eg.(aid_28-rsp-resnet-50-ckpt.pth / resnet50-19c8e357.pth)")
    # --------------------------------------------------------------------------------------------------------
    parser.add_argument('--resume', default=False, type=str, help="the pre-trained model path")
    parser.add_argument('--data_path', default='./data/', type=str, help=" Preprocessed data")
    parser.add_argument('--image_path', default='../rs_data/', type=str, help="images data path")
    parser.add_argument('--vocab_path', default='./vocab/', type=str, help="vocab data path")
    parser.add_argument('--epochs', default=100, type=int, help="the epochs of train")
    parser.add_argument('--eval_step', default=1, type=int, help="the epochs of eval")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch train size")
    parser.add_argument('--batch_size_val', default=100, type=int, help="Batch val size")
    parser.add_argument('--shard_size', default=128, type=int, help="Batch shard size")
    parser.add_argument('--workers', default=3, type=int, help="the worker num of dataloader")

    parser.add_argument('--k_fold_current_num', default=0, type=int, help="current num of k_fold")
    # GPU setting
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--init_method', default='tcp://localhost:18888', help="init-method")
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')
    parser.add_argument('--world_size', default=2, type=int, help="world size")
    parser.add_argument('--use_mix_precision', default=False, action='store_true', help="whether to use mix precision")

    # Model parameter setting
    parser.add_argument('--embed_dim', default=512, type=int, help="the total num of k_flod")
    parser.add_argument('--margin', default=0.2, type=float)
    parser.add_argument('--max_violation', default=0, type=int)
    parser.add_argument('--grad_clip', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--il_measure', default=False, help='Similarity measure used (cosine|l1|l2|msd)')
    # RNN laguage model parameter
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.(e.g. 300, 512)')
    parser.add_argument('--use_bidirectional_rnn', default=True, type=str)
    parser.add_argument('--num_layers', default=1, type=int, help='Number of GRU layers.')
    parser.add_argument('--is_finetune', default=False, type=str, help='Finetune resnet or not')
    ## no set setting
    parser.add_argument('--logger_name', default='logs/', type=str, help="the path of logs")
    parser.add_argument('--print_freq', default=10, type=int)
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
    # choose dataset
    args.data_path = args.data_path + args.data_name + '_precomp/'
    args.image_path = args.image_path + args.data_name + '/images/'
    args.vocab_path = args.vocab_path + args.data_name + '_splits_vocab.json'
    # Print arguments
    print('-------------------------')
    print('# Hyper Parameters setting')
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('-------------------------')
    print('')
    return args


def main(args):
    # choose model
    if args.model_name == "SWAN":
        from layers import SWAN as models
    else:
        raise NotImplementedError

    # make vocab
    vocab = deserialize_vocab(args.vocab_path)

    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(args, vocab)

    model = models.factory(args,
                           vocab.word2idx,
                           cuda=True,
                           data_parallel=args.distributed)

    print("Total Params: ", sum(p.numel() for p in model.parameters()))
    print("Total Requires_grad Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    best_rsum = 0
    best_epoch = 0
    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        best_rsum = checkpoint['best_rsum']
        best_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        print('====== best rsum is : {:.2f}, best epoch is {} ======='.format(best_rsum, best_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # evaluate on test set
    sims = engine.test(args, test_loader, model)

    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n" \
                "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n" \
                "mR:{:.2f}\n".format(r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore)

    print(all_score)

    curr_kf = args.k_fold_current_num
    return [curr_kf, best_epoch, best_rsum, r1i, r5i, r10i, r1t, r5t, r10t, currscore]


def update_options_savepath(args, k):
    args_new = copy.deepcopy(args)
    args_new.k_fold_current_num = k
    args_new.resume = args.ckpt_save_path + args.data_name + '/' + args.experiment_name + "/" + str(k) \
                      + "/" + args.model_name + '_best.pth.tar'
    return args_new


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    args = parser_options()

    # calc ave k results
    last_score = []
    for k in range(args.k_fold_nums):
        print("Start evaluate {}th fold".format(k))

        # update save path
        args = update_options_savepath(args, k)
        print(args.ckpt_save_path)
        # run experiment
        one_score = main(args)
        last_score.append(one_score)
        print("Complete evaluate {}th fold".format(k))

    # caculate average score
    print("===================== Ave Score ({}-fold verify) =================".format(args.k_fold_nums))
    last_score_ = np.average(last_score, axis=0)
    last_score.append([])
    last_score.append(list(last_score_))
    names = ['curr_kf', 'best_epoch', 'best_rsum', 'r1i', 'r5i', 'r10i', 'r1t', 'r5t', 'r10t', 'mr']
    # print avg result
    for name, score in zip(names, last_score_):
        print("{}:{:.2f}".format(name, score))

    # save result to excel_file
    file_name = args.data_name + '_' +  args.model_name + '_' + args.experiment_name
    utils.write_excel_file(args.ckpt_save_path, file_name, names, last_score)

