import time
import torch
import numpy as np
from torch.autograd import Variable
import utils
import tensorboard_logger as tb_logger
import logging
from torch.nn.utils.clip_grad import clip_grad_norm


# ==============================================================
def train(args, train_loader, model, optimizer, epoch):

    # extract value
    grad_clip = args.grad_clip
    max_violation = args.max_violation
    margin = args.margin
    # loss_name = args.model_name + "_" + args.data_name
    print_freq = args.print_freq
    if args.distributed:
        mean_loss = torch.zeros(1).to(args.gpuid)
    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())

    for i, train_data in enumerate(train_loader):
        images, captions, lengths, ids= train_data

        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)

        input_text = Variable(captions)

        if torch.cuda.is_available():
            input_visual = input_visual.cuda(args.gpuid)
            input_text = input_text.cuda(args.gpuid)

        torch.cuda.synchronize(device=args.gpuid)

        if not args.il_measure:
            # ONE
            scores = model(input_visual, input_text, lengths)
            loss = utils.calcul_contraloss(args, scores, input_visual.size(0), margin, max_violation=max_violation)

        else:
            scores,scores_intra_img,scores_intra_cap = model(input_visual, input_text, lengths)
            intra_loss = utils.calcul_intraloss(args,scores_intra_img) + utils.calcul_intraloss(args,scores_intra_cap)
            loss = utils.calcul_contraloss(args, scores, input_visual.size(0), margin, max_violation=max_violation) + intra_loss

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        optimizer.zero_grad()
        loss.backward()

        if args.distributed:
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses

            train_logger.update('Loss', round(mean_loss.item(),3))
        else:
            if args.il_measure:
                train_logger.update('IntraLoss', intra_loss.cpu().data.numpy())
            train_logger.update('Loss', loss.cpu().data.numpy())

        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and args.rank == 0:
            logging.info(
                'Epoch [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                args.ckpt_save_path+ args.model_name + "_" + args.data_name + ".txt"
            )

        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)


def validate(args, val_loader, model):
    print('')
    print("--------------------- start val on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()

    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))

    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0] * len(val_loader.dataset)

    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids = val_data

        for (id, img, cap, l) in zip(ids, (images.numpy().copy()),  (captions.numpy().copy()), lengths):
            input_visual[id] = img
            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis_MSSF(args, input_visual, input_text, model,
                             lengths=input_text_lengeth)

    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)

    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)

    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n" \
                "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n" \
                "mR:{:.2f}".format(r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore)

    print("--------------------- end val on training ---------------------")
    print('')

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, all_score



def validate_test(args, test_loader, model):
    print('')
    print("--------------------- start test on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))


    input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0] * len(test_loader.dataset)

    embed_start = time.time()
    for i, val_data in enumerate(test_loader):

        images, captions, lengths, ids = val_data

        for (id, img,cap, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), lengths):
            input_visual[id] = img

            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    embed_end = time.time()
    print("## test embedding time: {:.2f} s".format(embed_end-embed_start))

    d = utils.shard_dis_MSSF(args, input_visual, input_text, model, lengths=input_text_lengeth)

    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)

    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)

    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n" \
                "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n" \
                "mR:{:.2f}".format(r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore)

    print("--------------------- end test on training ---------------------")
    print('')

    tb_logger.log_value('r1i_test', r1i, step=model.Eiters)
    tb_logger.log_value('r5i_test', r5i, step=model.Eiters)
    tb_logger.log_value('r10i_test', r10i, step=model.Eiters)
    tb_logger.log_value('medri_test', medri, step=model.Eiters)
    tb_logger.log_value('meanri_test', meanri, step=model.Eiters)
    tb_logger.log_value('r1t_test', r1t, step=model.Eiters)
    tb_logger.log_value('r5t_test', r5t, step=model.Eiters)
    tb_logger.log_value('r10t_test', r10t, step=model.Eiters)
    tb_logger.log_value('medrt_test', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt_test', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum_test', currscore, step=model.Eiters)

    return currscore, all_score


def test(args, test_loader, model):
    print('')
    print("--------------------- start test ---------------------")
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))

    input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0] * len(test_loader.dataset)

    embed_start = time.time()
    for i, val_data in enumerate(test_loader):

        images, captions, lengths, ids = val_data

        for (id, img,cap, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), lengths):
            input_visual[id] = img

            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    embed_end = time.time()
    print("## embedding time: {:.2f} s".format(embed_end-embed_start))

    d = utils.shard_dis_SWAN(args, input_visual, input_text, model, lengths=input_text_lengeth)

    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))
    print("--------------------- end test ---------------------")
    print('')
    return d

def save(args, test_loader, model):
    print('')
    print("--------------------- start test ---------------------")
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))

    input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0] * len(test_loader.dataset)

    for i, val_data in enumerate(test_loader):

        images, captions, lengths, ids = val_data

        for (id, img,cap, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), lengths):
            input_visual[id] = img

            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    img_emb, text_emb = utils.save_img_text_emb(args, input_visual, input_text, model, lengths=input_text_lengeth)

    return img_emb, text_emb