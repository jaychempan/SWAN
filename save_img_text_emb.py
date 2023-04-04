import numpy as np
import os
import torch
import argparse
import utils
import data
import engine
from vocab import deserialize_vocab

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    ##################################################################################
    ## training path setting
    parser.add_argument('-g','--gpuid', default=0, type=int, help="which gpu to use")
    parser.add_argument('-e', '--experiment_name', default='MSSF_V2', type=str, help="the file name of ckpt save")
    parser.add_argument('-m', '--model_name', default='MSSF_V2_save', type=str, help="Model Name")
    parser.add_argument('--data_name', default='rsitmd', type=str, help="Dataset Name.(eg: rsitmd or rsicd)")
    parser.add_argument('--k_fold_current_num', default=0, type=int, help="current num of k_fold")
    parser.add_argument('--ckpt_save_path', default='./checkpoint_fix_data/', type=str, help="the path of ckpt save")
    # if not use resume, the path also be generated
    parser.add_argument('--resume', default='./checkpoint_fix_data/rsitmd/MSSF_V2/0/MSSF_V2_best.pth.tar', type=str,
                        help="the model path.(eg:./checkpoint/rsitmd/MSSF/0/MSSF_best.pth.tar)")
    #--------------------------------------------------------------------------------------------------------
    parser.add_argument('--data_path', default='./fix_data/', type=str, help=" Preprocessed data")
    parser.add_argument('--image_path', default='../rs_data/', type=str, help="images data path")
    parser.add_argument('--vocab_path', default='./vocab/', type=str, help="vocab data path")
    parser.add_argument('--resnet_ckpt', default='./layers/aid_28-rsp-resnet-50-ckpt.pth', type=str,
                        help="restnet pre model path")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch train size")
    parser.add_argument('--batch_size_val', default=100, type=int, help="Batch val size")
    parser.add_argument('--shard_size', default=128, type=int, help="Batch shard size")
    parser.add_argument('--workers', default=3, type=int, help="the worker num of dataloader")
    parser.add_argument('--k_fold_nums', default=3, type=int, help="the total num of k_flod")
    # GPU setting
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--init_method', default='tcp://localhost:18888', help="init-method")
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')
    parser.add_argument('--world_size', default=2, type=int, help="world size")
    parser.add_argument('--use_mix_precision', default=False, action='store_true', help="whether to use mix precision")
    # Model parameter setting
    parser.add_argument('--embed_dim', default=512, type=int, help="the total num of k_flod")
    parser.add_argument('--margin', default=0.2, type=float)
    parser.add_argument('--max_violation', default=False, action='store_true')
    parser.add_argument('--grad_clip', default=0.0, type=float)
    # RNN laguage model parameter
    parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding.(e.g. 300, 512)')
    parser.add_argument('--use_bidirectional_rnn', default=True, type=str)
    parser.add_argument('--num_layers', default=1, type=int, help='Number of GRU layers.')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--il_measure', default=False, help='Similarity measure used (cosine|l1|l2|msd)')
    parser.add_argument('--is_finetune', default=False, type=str, help='Finetune resnet or not')
    args = parser.parse_args()
    # choose dataset
    args.data_path = args.data_path + args.data_name + '_precomp/'
    args.image_path = args.image_path + args.data_name + '/images/'
    args.vocab_path = args.vocab_path + args.data_name + '_splits_vocab.json'
    if not args.resume:
        args.resume = args.ckpt_save_path + args.data_name + '/' + args.experiment_name + '/' \
                      + str(args.k_fold_current_num) + '/' + args.model_name + '_best.pth.tar'
    print(args.resume)
    # Print arguments
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

    # choose model
    if args.model_name == "MSSF":
        from layers import MSSF as models
    elif args.model_name == "MSSF_V2":
        from layers import MSSF_V2 as models
    elif args.model_name == "MSSF_V2_wo_SFGS":
        from layers import MSSF_V2_wo_SFGS as models
    elif args.model_name == "MSSF_V2_wo_TCGE":
        from layers import MSSF_V2_wo_TCGE as models
    elif args.model_name == "MSSF_V2_wo_VMSF":
        from layers import MSSF_V2_wo_VMSF as models
    elif args.model_name == "MSSF_V2_rp_WSA_SA":
        from layers import MSSF_V2_rp_WSA_SA as models
    elif args.model_name == "MSSF_V2_save":
        from layers import MSSF_V2_save as models
    else:
        raise NotImplementedError

    # make vocab
    vocab = deserialize_vocab(args.vocab_path)

    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(args, vocab)

    model = models.factory(args,
                           vocab.word2idx,
                           cuda=True,
                           data_parallel=False)

    print("Total Params: ", sum(p.numel() for p in model.parameters()))
    print("Total Requires_grad Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpuid))
        model.load_state_dict(checkpoint['model'])
        print('====== best rsum is : {:.2f},   epoch is {} ======='.format(checkpoint['best_rsum'],checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    img_emb, text_emb = engine.save(args, test_loader, model)

    return img_emb, text_emb



if __name__ == '__main__':
    args = parser_options()

    # run experiment
    img_emb, text_emb = main(args)

    import mytools
    mytools.save_to_npy(img_emb, "img_emb.npy")
    mytools.save_to_npy(text_emb, "text_emb.npy")




