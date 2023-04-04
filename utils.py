import os
import random
import openpyxl as xl
import torch
import numpy as np
import math
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import time
import torch.distributed as dist
import seaborn as sns
from matplotlib import pyplot as plt


# 从npy中读取
def load_from_npy(filename):
    info = np.load(filename, allow_pickle=True)
    return info


# 从txt文件中读取数据
def load_from_txt(filename, encoding="utf-8"):
    f = open(filename, 'r', encoding=encoding)
    contexts = f.readlines()
    return contexts


# 保存结果到txt文件
def log_to_txt(contexts=None, filename="save.txt", mark=False, encoding='UTF-8', mode='a'):
    f = open(filename, mode, encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c) + " | " + str(contexts[c]) + "\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts, list):
            tmp = ""
            for c in contexts:
                tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)
    f.close()


def collect_match(args, input):
    """change the model output to the match matrix"""
    image_size = input.size(0)
    text_size = input.size(1)

    # match_v = torch.zeros(image_size, text_size, 1)
    # match_v = match_v.view(image_size*text_size, 1)
    input_ = nn.LogSoftmax(2)(input)
    output = torch.index_select(input_, 2, Variable(torch.LongTensor([1])).cuda(args.gpuid))

    return output


def collect_neg(args, input):
    """"collect the hard negative sample"""
    if input.dim() != 2:
        return ValueError

    batch_size = input.size(0)
    mask = Variable(torch.eye(batch_size) > 0.5).cuda(args.gpuid)
    output = input.masked_fill_(mask, 0)
    output_r = output.max(1)[0]
    output_c = output.max(0)[0]
    loss_n = torch.mean(output_r) + torch.mean(output_c)
    return loss_n


# 计算对比损失函数
def calcul_contraloss(args, scores, size, margin, max_violation=False):
    diagonal = scores.diag().view(size, 1)

    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda(args.gpuid)
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    sum_cost_s = cost_s.sum()
    sum_cost_im = cost_im.sum()

    return sum_cost_s + sum_cost_im


# ========================================================================================================

# 计算内部损失函数
def calcul_intraloss(args, scores, up=0.5, down=0.05, lamb=1.0):
    if args.il_measure == 'cosine':
        diagonal = scores.diag()
        scores = scores.cuda(args.gpuid)
        eye = torch.eye(scores.size(0)).float().cuda(args.gpuid)
        scores_non_self = scores - eye
        # scores_non_self.gt_(self.up).lt_(1 - self.down)
        scores_non_self = scores_non_self * (
            scores_non_self.gt(up).float())
        scores_non_self = scores_non_self * (
            scores_non_self.lt(1 - down).float())
        scores_norm = scores_non_self.sum() / scores.size(0)
        # print(scores_norm.item())

    elif args.il_measure == 'msd' or args.il_measure == 'l1' or args.il_measure == 'l2':
        scores_non_self = torch.nn.functional.normalize(scores).cuda(args.gpuid)

        idx_up = round(up * scores.size(0))
        idx_down = round(down * scores.size(0))
        _, s_index = scores_non_self.sort()
        s_mean = scores_non_self.mean()

        s_up = scores_non_self[0, s_index[0, idx_up]]
        s_down = scores_non_self[0, s_index[0, idx_down]]

        scores_non_self = scores_non_self * (
            scores_non_self.gt(s_down).float())
        scores_non_self = scores_non_self * (
            scores_non_self.lt(s_up).float())
        scores_norm = scores_non_self.sum() / scores.size(0)

    return lamb * scores_norm


# ========================================================================================================
def acc_train(input):
    predicted = input.squeeze().numpy()
    batch_size = predicted.shape[0]
    predicted[predicted > math.log(0.5)] = 1
    predicted[predicted < math.log(0.5)] = 0
    target = np.eye(batch_size)
    recall = np.sum(predicted * target) / np.sum(target)
    precision = np.sum(predicted * target) / np.sum(predicted)
    acc = 1 - np.sum(abs(predicted - target)) / (target.shape[0] * target.shape[1])

    return acc, recall, precision


def acc_i2t(input):
    """Computes the precision@k for the specified values of k of i2t"""
    # input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i(input):
    """Computes the precision@k for the specified values of k of t2i"""
    # input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5 * image_size)
    top1 = np.zeros(5 * image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)
# 计算同类映射
def cal_class_idxs(class_):
    all_class_idxs = []
    for i in range(len(class_)):
        now_class = class_[i].strip('\n')
        one_class_idxs = []
        for j in range(len(class_)):
            search_class = class_[j].strip('\n')
            if now_class == search_class:
                one_class_idxs.append(j)
        all_class_idxs.append(one_class_idxs)
    return all_class_idxs

# 计算同类场景检索排序指标-i2t
def srr_i2t(sim, all_class_idxs, r):
    """Computes the scene retrieval ranking of k of i2t"""
    image_size = sim.shape[0]

    cnt_pro = []
    cnt_idxs = []
    for index in range(image_size):
        cnt_i = 0
        inds_r = np.argsort(sim[index])[::-1][:r]
        # Score
        for i in all_class_idxs[index * 5]:
            for j in inds_r:
                if i == j:
                    cnt_i = cnt_i + 1
        cnt_pro.append(cnt_i / r)
        cnt_idxs.append(list(inds_r))
    # return cnt_pro, cnt_idxs
    return np.average(cnt_pro)

# 计算同类场景检索排序指标-t2i
def srr_t2i(sim, all_class_idxs, r):
    """Computes the scene retrieval ranking of k of t2i"""
    sim = sim.T
    cap_size = sim.shape[0]
#     print(sim.shape)
    cnt_pro = []
    cnt_idxs = []
    for index in range(cap_size):
        cnt_i = 0
        inds_r = np.argsort(sim[index])[::-1][:r]
        inds_r = [x*5 for x in inds_r]
        # Score
        for i in all_class_idxs[index]:
            for j in inds_r:
                if i == j:
                    cnt_i = cnt_i + 1
        cnt_pro.append(cnt_i / r)
        cnt_idxs.append(list(inds_r))
    # return cnt_pro,cnt_idxs
    return np.average(cnt_pro)
# 分块计算距离
def shard_dis_SWAN(args, images, captions, model, lengths):
    """compute image-caption pairwise distance during validation and test"""
    # l1 = len(images)
    # l2 = len(captions)
    n_im_shard = (len(images) - 1) // args.shard_size + 1
    n_cap_shard = (len(captions) - 1) // args.shard_size + 1

    d = np.zeros((len(images), len(captions)))
    all = []
    print("==> start to compute image-caption pairwise distance <==")
    for i in range(n_im_shard):
        im_start, im_end = args.shard_size * i, min(args.shard_size * (i + 1), len(images))

        print("Calculate the similarity in batches: [{}/{}]".format(i, n_im_shard))

        for j in range(n_cap_shard):
            cap_start, cap_end = args.shard_size * j, min(args.shard_size * (j + 1), len(captions))
            with torch.no_grad():
                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda(args.gpuid)

                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda(args.gpuid)
                l = lengths[cap_start:cap_end]
                t1 = time.time()
                if args.il_measure:
                    sim,_,_ = model(im, s, l)
                else:
                    sim = model(im, s, l)

                t2 = time.time()
                all.append(t2 - t1)

                sim = sim.squeeze()

                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()

    print("infer time:{:.2f}".format(np.average(all)))
    print("==> end to compute image-caption pairwise distance <==")
    return d

# 导出图像向量和文本向量
def save_img_text_emb(args, images, captions, model, lengths):
    """compute image-caption pairwise distance during validation and test"""
    n_im_shard = (len(images) - 1) // args.shard_size + 1
    n_cap_shard = (len(captions) - 1) // args.shard_size + 1

    img_emb_all = np.zeros((1,512))
    text_emb_all = np.zeros((1, 512))

    for i in range(n_im_shard):
        im_start, im_end = args.shard_size * i, min(args.shard_size * (i + 1), len(images))
        with torch.no_grad():
            im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda(args.gpuid)

            img_emb = model.get_img_emb(im)
            img_emb = img_emb.cpu().numpy()
            img_emb_all = np.concatenate((img_emb_all, img_emb), axis=0)

    for j in range(n_cap_shard):
        cap_start, cap_end = args.shard_size * j, min(args.shard_size * (j + 1), len(captions))
        with torch.no_grad():
            s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda(args.gpuid)
            l = lengths[cap_start:cap_end]

            text_emb = model.get_text_emb(s, l)
            text_emb = text_emb.cpu().numpy()
            text_emb_all = np.concatenate((text_emb_all, text_emb), axis=0)

    return img_emb_all[1:,:], text_emb_all[1:,:]

# 保存模型文件
def save_checkpoint(state, is_best, filename, prefix='', model_name=None):
    tries = 15
    error = None
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + model_name + '_best.pth.tar')

        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


# 动态调整学习率
def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

        if epoch % args.lr_update_epoch == args.lr_update_epoch - 1:
            lr = lr * args.lr_decay_param

        param_group['lr'] = lr

    print("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))


# ====================================================================

# 并行计算时用来统计平均loss
def reduce_value(args, value, average=True):
    world_size = args.world_size
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)  # 对不同设备之间的value求和
        if average:  # 如果需要求平均，获得多块GPU计算loss的均值
            value /= world_size

    return value


# 从文件名列表中生成对应的场景类别
def gen_class_from_list(lis):
    # 1-first proess
    class_1 = [name.split('_')[0] for name in lis]

    for i in range(len(class_1)):
        if class_1[i][0] == '0':
            class_1[i] = 'noneclass'
    #     print("====================================")
    #     print('=>total len is {}； total class is {}\n=> The class is {}'.format(len(class_1),len(set(class_1)),set(class_1)))
    #     print("====================================")
    return class_1


# 统计场景类别
def cnt_class(class_):
    cnt = {}
    for value in class_:
        cnt[value] = cnt.get(value, 0) + 1
    return cnt


# 计算val/train 比例
def cal_pro_cnt(bb, aa):
    cc = {}
    for key in aa.keys():
        try:
            cc[key] = round(bb[key] / aa[key], 2)
        except:
            continue
    return cc


# 绘制train+val场景分布图
def vis_cal_data_info(args, data_info_path, train_fnames, val_fnames):
    # 不同数据集，场景不一样
    if args.data_name == 'rsicd':
        xticklabels = ['airport', 'bareland', 'baseballfield', 'beach', 'bridge', 'center', 'church', 'commercial',
                       'denseresidential', 'desert', 'farmland', 'forest', 'industrial', 'meadow', 'mediumresidential',
                       'mountain', 'park', 'parking', 'playground', 'pond', 'port', 'railwaystation', 'resort', 'river',
                       'school', 'sparseresidential', 'square', 'stadium', 'storagetanks', 'viaduct', 'noneclass']
    elif args.data_name == 'rsitmd':
        xticklabels = ['airport', 'bareland', 'baseballfield', 'beach', 'bridge', 'center', 'church', 'commercial',
                       'denseresidential', 'desert', 'farmland', 'forest', 'industrial', 'meadow', 'mediumresidential',
                       'mountain', 'park', 'parking', 'playground', 'pond', 'port', 'railwaystation', 'resort', 'river',
                       'school', 'sparseresidential', 'square', 'stadium', 'storagetanks', 'viaduct', 'intersection',
                       'plane', 'boat']

    class_train = gen_class_from_list(train_fnames)
    class_val = gen_class_from_list(list(set(val_fnames)))
    cnt_train = cnt_class(class_train)
    cnt_val = cnt_class(class_val)
    pro = cal_pro_cnt(cnt_val, cnt_train)
    log_to_txt(contexts=pro, filename=data_info_path + 'data.txt')
    plt.figure(figsize=[12, 8], dpi=100)
    plt.subplot(211)
    ax = sns.barplot(x=list(cnt_train.keys()), y=list(cnt_train.values()), order=xticklabels)
    ax.set_xticklabels([i[:5] for i in xticklabels], rotation=50, fontsize=10)

    for p in ax.patches:
        # get the height of each bar
        height = p.get_height()
        #         print(height)
        try:
            ax.text(x=p.get_x() + (p.get_width() / 2), y=height + 10, s=int(height), ha="center")
        except:
            continue

    plt.ylim(0, 1000)
    plt.tight_layout()

    plt.subplot(212)
    ax = sns.barplot(x=list(cnt_val.keys()), y=list(cnt_val.values()), order=xticklabels)
    ax.set_xticklabels([i[:5] for i in xticklabels], rotation=50, fontsize=10)

    for p in ax.patches:
        # get the height of each bar
        height = p.get_height()
        #         print(height)
        try:
            ax.text(x=p.get_x() + (p.get_width() / 2), y=height + 10, s=int(height), ha="center")
        except:
            continue
    plt.ylim(0, 1000)
    plt.tight_layout()
    plt.savefig(data_info_path + 'data.png')

def write_excel_file(folder_path, file_name, headers, result):
    result_path = os.path.join(folder_path, file_name + ".xlsx")

    print(result_path)
    print('***** start to write excel file ' + result_path + ' *****')

    if os.path.exists(result_path):
        print('***** excel exist, add data in tail ' + result_path + ' *****')
        workbook = xl.load_workbook(result_path)
    else:
        print('***** excel note exist，create excel ' + result_path + ' *****')
        workbook = xl.Workbook()
        workbook.save(result_path)

    sheet = workbook.active

    sheet.append(headers)

    for data in result:
        sheet.append(data)
        workbook.save(result_path)
        print('***** generate excel file ' + result_path + ' *****')


# Random seed
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    #torch.backends.cudnn.enabled = False

# ===============================================================================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count
