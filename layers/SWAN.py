import math

import numpy as np
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.init
from torch.nn.parallel.distributed import DistributedDataParallel
import copy
import torch
import torch.nn as nn
import torch.nn.init
from .resnet import resnet50

#============
# Main Model
#============
class SWAN(nn.Module):
    def __init__(self, args, word2idx):
        super(SWAN, self).__init__()
        self.Eiters = 0

        # Image Encoder
        self.image_encoder =  ImageExtractFeature(args)
        # Text Encoder
        self.text_encoder = TextExtractFeature(args, word2idx)
        # Scene Fine-Grained Sensing Module
        self.sfgs = SFGS(args)
        # Vsion Global-Local Features Fusion
        self.agg = Aggregation(args)
        # Text Coarse-Grained Enhancement Module
        self.tcge = TCGE(args)

    def forward(self, img , text, lengths):
        # Visual Part
        vl_fea, vg_emb = self.image_encoder(img)

        vl_emb = self.sfgs(vl_fea)

        img_emb = self.agg(vl_emb, vg_emb)

        # Textual Part
        cap_fea = self.text_encoder(text, lengths)

        text_emb = self.tcge(cap_fea, lengths)

        # Calculating similarity
        sims = cosine_sim(img_emb, text_emb)

        return sims
#=========================
# Image feature extraction
#========================
class ImageExtractFeature(nn.Module):
    def __init__(self, args):
        super(ImageExtractFeature, self).__init__()
        self.embed_dim = args.embed_dim
        self.is_finetune = args.is_finetune
        # load resnet50
        self.resnet = resnet50(args, num_classes = 30,pretrained=True)
        # Vision Multi-Scale Fusion Module
        self.vmsf = VMSF(args)

        # Filtering local features
        self.conv_filter = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

        for param in self.resnet.parameters():
            param.requires_grad = self.is_finetune

    def forward(self, img):
        # Shallow features
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Deep features
        deep_fea_1 = self.resnet.layer2(self.resnet.layer1(x))
        deep_fea_2 = self.resnet.layer3(deep_fea_1)
        deep_fea_3 = self.resnet.layer4(deep_fea_2)

        shallow_fea = self.conv_filter(x)

        deep_feas = (deep_fea_1, deep_fea_2, deep_fea_3)
        vg_emb = self.vmsf(deep_feas)
        return shallow_fea, vg_emb

# =========================
# Text feature extraction
# =========================
class TextExtractFeature(nn.Module):
    def __init__(self, args, word2idx):
        super(TextExtractFeature, self).__init__()
        self.gpuid = args.gpuid
        self.embed_dim = args.embed_dim
        self.word_dim = args.word_dim
        self.vocab_size = 8590
        self.num_layers = args.num_layers
        self.use_bidirectional_rnn = args.use_bidirectional_rnn
        # word embedding
        self.embed = nn.Embedding(self.vocab_size, self.word_dim)

        # caption embedding
        self.use_bidirectional_rnn = self.use_bidirectional_rnn
        print('=> using bidirectional rnn:{}'.format(self.use_bidirectional_rnn))
        self.rnn = nn.GRU(self.word_dim, self.embed_dim, self.num_layers,
                          batch_first=True, bidirectional=self.use_bidirectional_rnn)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(0.4)

        self.init_weights(word2idx, self.word_dim)

    def init_weights(self, word2idx, word_dim):
        # Load pretrained word embedding
        wemb = torchtext.vocab.GloVe()

        assert wemb.vectors.shape[1] == word_dim

        # quick-and-dirty trick to improve word-hit rate
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace(
                    '-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        # Embed word ids to vectors
        x = self.dropout(self.embed(x))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bidirectional_rnn:
            cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] +
                       cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        return cap_emb

# ================================
# Vision Multi-Scale Fusion Module
# ================================
class VMSF(nn.Module):
    def __init__(self, args):
        super(VMSF, self).__init__()
        self.embed_dim = args.embed_dim
        self.dropout_r = 0.2
        self.use_relu = True

        self.conv_512 = nn.Conv2d(in_channels=512, out_channels=self.embed_dim, kernel_size=1, stride=1)
        self.conv_1024 = nn.Conv2d(in_channels=1024, out_channels=self.embed_dim, kernel_size=1, stride=1)
        self.conv_2048 = nn.Conv2d(in_channels=2048, out_channels=self.embed_dim * 2, kernel_size=1, stride=1)

        self.up_sample_double = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample_half = nn.Upsample(scale_factor=0.5, mode='nearest')

        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dim * 4, out_channels=self.embed_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.channel_filter = MLP(self.embed_dim, self.embed_dim * 2, self.embed_dim, self.dropout_r, self.use_relu)

    def forward(self, deep_feas):
        d1, d2, d3 = deep_feas

        p_2 = self.conv_1024(d2)
        up_4 = self.up_sample_double(self.conv_2048(d3))
        up_2 = self.up_sample_half(self.conv_512(d1))

        # Depth concat && channel attention
        ms_fea = self.channel_att(torch.cat([up_2, p_2, up_4], dim=1))

        # Mean fsuion && chanel filter
        high_emb = self.channel_filter(ms_fea.mean(-1).mean(-1))

        return high_emb

#=================================
# Scene Fine-Grained Sensing Module
#=================================
class SFGS(nn.Module):
    def __init__(self ,args, dim = 32):
        super(SFGS,self).__init__()
        self.args = args
        self.embed_dim = args.embed_dim
        self.dim = dim
        self.dropout_r = 0.1
        self.use_relu = True

        self.conv2d_block_11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16,16))
        )
        self.conv2d_block_33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=3, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        self.conv2d_block_55 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=5, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.fc = FC(self.embed_dim // 2, self.embed_dim , self.dropout_r, self.use_relu)

        self.wsa = WSA(args, num_dim=128, is_weighted=True)

    def forward(self, vl_fea):
        bs, dim, _, _ = vl_fea.size()

        vl_1 = self.conv2d_block_11(vl_fea).view(bs, dim, -1)
        vl_2 = self.conv2d_block_33(vl_fea).view(bs, dim, -1)
        vl_3 = self.conv2d_block_55(vl_fea).view(bs, dim * 2, -1)

        vl_depth = torch.cat([vl_1,vl_2,vl_3], dim=1)

        return self.wsa(self.fc(vl_depth)).mean(1)

#=======================================
# Global-Local embeddings Aggregation Module
#=======================================
class Aggregation(nn.Module):
    def __init__(self, args):
        super(Aggregation, self).__init__()
        self.args = args
        self.embed_dim = args.embed_dim

        self.fc_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.fc_2 = nn.Linear(self.embed_dim,self.embed_dim)

    def forward(self, vl_emb, gl_emb):

        # Depth concat
        v_emb = torch.cat([vl_emb,gl_emb],dim=1)

        return self.fc_2(torch.relu(self.fc_1(v_emb)))

#========================================
# Text Coarse-Grained Enhancement Module
#========================================
class TCGE(nn.Module):
    def __init__(self,args):
        super(TCGE, self).__init__()
        self.embed_dim = args.embed_dim
        self.gpuid = args.gpuid

        self.bn_1d = nn.BatchNorm1d(self.embed_dim)
        self.ga = GA(args)

        self.dropout = nn.Dropout(0.2)

        self.mlp = MLP(self.embed_dim, self.embed_dim * 2, self.embed_dim, 0.1, True)

        self.conv1d_block_22 = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, stride = 2, kernel_size= 2),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.conv1d_block_33 = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, stride = 3, kernel_size= 3),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, cap_emb, lengths):
        # GA Embeddings
        bs, dim, emb_dim = cap_emb.size()
        ga_emb = cap_emb + self.dropout(self.bn_1d(self.ga(cap_emb).view(bs * dim, -1)).view(bs, dim, -1))
        # Joint Wrod Embeddings
        f2 = self.conv1d_block_22(cap_emb.permute(0, 2, 1)).permute(0, 2, 1)
        f3 = self.conv1d_block_33(cap_emb.permute(0, 2, 1)).permute(0, 2, 1)
        jw_emb = torch.cat([f2, f3],dim=1)

        # GA-JW Fusion
        ga_jw = torch.cat([ga_emb, jw_emb], dim=1)
        tex_emb = self.mlp(ga_jw) + ga_jw

        I = torch.LongTensor(lengths).view(-1, 1, 1) # 100, 1, 1
        I = Variable(I.expand(tex_emb.size(0), 1, self.embed_dim)-1).cuda(self.gpuid) # 100, 1, 512
        out = torch.gather(tex_emb, 1, I).squeeze(1)

        return l2norm(out, dim=-1)

#======================
# Multi-Head Attention
#======================
class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.embed_dim = args.embed_dim
        self.dropout_r = 0.1
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_merge = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(self.dropout_r)

    def forward(self, v, k, q, mask=None):
        bs = q.size(0)

        v = self.linear_v(v).view(bs, -1, 8, 64).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, 8, 64).transpose(1, 2)
        q = self.linear_q(q).view(bs, -1, 8, 64).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)

        atted = self.linear_merge(atted)

        return atted

    def att(self, k, q, v, mask=None):
        d_k = q.shape[-1]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = torch.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, v)

#============================
# Weighted Self Attention
#============================
class WSA(nn.Module):
    def __init__(self,args, num_dim = 128, is_weighted = False):
        super(WSA, self).__init__()
        self.num_dim = num_dim
        self.embed_dim = args.embed_dim
        self.is_weighted = is_weighted
        self.dropout_r = 0.1

        self.mhatt = MHAtt(args)
        self.ffn = FeedForward(self.embed_dim, self.embed_dim * 2)

        self.dropout1 = nn.Dropout(self.dropout_r)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        self.dropout2 = nn.Dropout(self.dropout_r)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Learnable weights
        if is_weighted:
            self.fmp_weight = nn.Parameter(torch.randn(1, self.num_dim, self.embed_dim))
    def forward(self, x, x_mask=None):
        bs = x.shape[0]

        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        if self.is_weighted:
            # feature map fusion
            x = self.fmp_weight.expand(bs, x.shape[1], x.shape[2]).transpose(1, 2).bmm(x)

        return x

#===================
# Gated Attention
#===================
class GA(nn.Module):
    def __init__(self, args):
        super(GA, self).__init__()
        self.h = 2
        self.embed_dim = args.embed_dim
        self.d_k = self.embed_dim // self.h

        self.linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 3)

        self.fc_q = nn.Linear(self.d_k, self.d_k)
        self.fc_k = nn.Linear(self.d_k, self.d_k)
        self.fc_g = nn.Linear(self.d_k, self.d_k*2)

    def forward(self, cap_emb):
        bs = cap_emb.shape[0]

        q, k, v = [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (cap_emb, cap_emb, cap_emb))]

        # gate
        G = self.fc_q(q) * self.fc_k(k)
        M = torch.sigmoid(self.fc_g(G)) # (bs, h, num_region, d_k*2)
        q = q * M[:, :, :, :self.d_k]
        k = k * M[:, :, :, self.d_k:]

        scores = torch.div(torch.matmul(q, k.transpose(-2, -1)), math.sqrt(self.d_k), rounding_mode='floor')

        p_attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)

        return x

#==================
# Some Reuse Module
#==================
# full connection layer
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


# Feed Forward Nets
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# mlp
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout=dropout, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.linear(self.fc(x))
        return out

#====================
# Some Reuse Function
#====================
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12

def clones(module, N):
    """Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def factory(args, word2idx, cuda=True, data_parallel=False):
    args_new = copy.copy(args)

    model_without_ddp = SWAN(args_new, word2idx)

    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    if data_parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model = DistributedDataParallel(model, device_ids=[args.gpuid],find_unused_parameters=False)
        model_without_ddp = model.module
        if not cuda:
            raise ValueError

    return model_without_ddp