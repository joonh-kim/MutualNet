import importlib
import os
import time
import random

import torch
import torch.nn.functional as F
import numpy as np
import ComputePostBN
from utils.setlogger import get_logger

from utils.model_profiling import model_profiling
from utils.config import FLAGS
from utils.datasets import get_dataset

# set log files
saved_path = os.path.join("logs", '{}-{}'.format(FLAGS.dataset, FLAGS.model[7:]))
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, '{}_div1optimizer.log'.format('test' if FLAGS.test_only else 'train')))

def set_random_seed():
    """set random seed"""
    if hasattr(FLAGS, 'random_seed'):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    return model

def get_optimizer(model):
    """get optimizer"""
    # all depthwise convolution (N, 1, x, x) has no weight decay
    # weight decay only on normal conv and fc
    if FLAGS.dataset == 'imagenet1k':
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:  # normal conv
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:  # fc
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                                    momentum=FLAGS.momentum, nesterov=FLAGS.nesterov,
                                    weight_decay=FLAGS.weight_decay)
    return optimizer

def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
        model.apply(
            lambda m: setattr(m, 'width_mult', width_mult))
        print('Model profiling with width mult {}x:'.format(width_mult))
        verbose = width_mult == max(FLAGS.width_mult_list)
        model_profiling(
            model, FLAGS.image_size, FLAGS.image_size,
            verbose=getattr(FLAGS, 'model_profiling_verbose', verbose))

def train(epoch, warm_up, joint, loader, model, criterion, optimizer, lr_scheduler, MFLOPS_table):
    t_start = time.time()
    model.train()
    if epoch < warm_up:
        for param in model.module.attn.parameters():
            param.requires_grad = False
        for param in model.module.linear.parameters():
            param.requires_grad = False
    elif warm_up <= epoch < (warm_up + joint):
        for param in model.module.attn.parameters():
            param.requires_grad = True
        for param in model.module.linear.parameters():
            param.requires_grad = True
    else:
        for param in model.module.attn.parameters():
            param.requires_grad = False
        for param in model.module.linear.parameters():
            param.requires_grad = False
    tau = FLAGS.tau * np.exp(FLAGS.exp_decay_factor * (epoch - 1))

    for batch_idx, (input_list, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()

        # do max width and resolution
        max_width = FLAGS.width_mult_range[-1]
        model.apply(lambda m: setattr(m, 'width_mult', max_width))
        max_output = model(input_list[0])
        loss = criterion(max_output, target)
        loss.backward()
        max_output_detach = max_output.detach()

        # policy selection
        policy_mask = model(input_list[-1], tau=tau, policy=True)
        mask = policy_mask[:, 1:].clone()

        # do other widths and resolution
        output_list = torch.zeros(len(FLAGS.width_mult_list) - 1, target.size()[0], FLAGS.num_classes).cuda(non_blocking=True)
        for idx, width_mult in enumerate(sorted(FLAGS.width_mult_list[:-1], reverse=True)):
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            output_i = model(input_list[idx + 1])
            output_list[idx] = output_i
        output = mask.permute(1, 0).unsqueeze(-1) * output_list
        output = torch.cat((max_output_detach.unsqueeze(0), output), dim=0).sum(dim=0)
        loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1),
                                                         F.softmax(max_output_detach, dim=1))
        loss += 1e-5 * (policy_mask * MFLOPS_table.unsqueeze(0)).mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # print training log
        if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader)-1:
            with torch.no_grad():
                loss = criterion(output, target).cpu().numpy()
                indices = torch.max(output, dim=1)[1]
                acc = (indices == target).sum().cpu().numpy() / indices.size()[0]
                mean_MFLOPS = (policy_mask * MFLOPS_table.unsqueeze(0)).mean()

                logger.info('TRAIN {:.1f}s LR:{:.4f} tau:{:.3f} MFLOPS(avg):{:.2f} Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                    time.time() - t_start, optimizer.param_groups[0]['lr'], tau, mean_MFLOPS, epoch,
                    FLAGS.num_epochs, batch_idx, len(loader), loss, acc))


def validate(epoch, loader, model, criterion, postloader, MFLOPS_table):
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        tau = FLAGS.tau * np.exp(FLAGS.exp_decay_factor * (epoch - 1))
        loss, acc, cnt, MFLOPS = 0, 0, 0, 0
        for policy_idx, width_mult in enumerate(sorted(FLAGS.width_mult_list, reverse=True)):
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            model = ComputePostBN.ComputeBN(model, postloader, FLAGS.resolution_list[policy_idx])
            for batch_idx, (input_list, target) in enumerate(loader):
                target = target.cuda(non_blocking=True)
                policy_mask = model(input_list[-1], tau=tau, policy=True).permute(1, 0)[policy_idx]
                target = (target + 1) * policy_mask.type(torch.LongTensor).cuda(non_blocking=True) - 1
                output = model(input_list[policy_idx])
                loss += criterion(output, target).cpu().numpy() * policy_mask.sum().item()
                indices = torch.max(output, dim=1)[1]
                acc += ((indices == target) * policy_mask).sum().cpu().numpy()
                cnt += policy_mask.sum().item()
                MFLOPS += (policy_mask * MFLOPS_table[policy_idx]).sum().cpu().numpy()
        logger.info('VAL {:.1f}s tau:{:.3f} MFLOPS(avg):{:.2f} Epoch:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
            time.time() - t_start, tau, MFLOPS/cnt, epoch,
            FLAGS.num_epochs, loss/cnt, acc/cnt))


def train_val_test():
    """train and val"""
    # seed
    set_random_seed()

    # model
    model = get_model()
    model_wrapper = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_val = torch.nn.CrossEntropyLoss(ignore_index=-1)
    train_loader, val_loader = get_dataset()
    MFLOPS_table = torch.tensor(FLAGS.MFLOPS_table).cuda(non_blocking=True)

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        new_keys = [key for key in new_keys if 'running' not in key]
        new_keys = [key for key in new_keys if 'tracked' not in key]
        old_keys = [key for key in old_keys if 'running' not in key]
        old_keys = [key for key in old_keys if 'tracked' not in key]
        if not FLAGS.test_only:
            old_keys = old_keys[:-2]
            new_keys = new_keys[:-2]

        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = checkpoint[key_old]
        model_wrapper.load_state_dict(new_checkpoint, strict=False)
        print('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = get_optimizer(model_wrapper)
    # check resume training
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*FLAGS.num_epochs)
        lr_scheduler.last_epoch = last_epoch
        print('Loaded checkpoint {} at epoch {}.'.format(
            FLAGS.resume, last_epoch))
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*FLAGS.num_epochs)
        last_epoch = lr_scheduler.last_epoch
        # print model and do profiling
        print(model_wrapper)
        if FLAGS.profiling:
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)

    if FLAGS.test_only:
        logger.info('Start testing.')
        validate(last_epoch, val_loader, model_wrapper, criterion_val, train_loader, MFLOPS_table)
        return

    logger.info('Start training.')
    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        # train
        train(epoch, FLAGS.warm_up, FLAGS.joint, train_loader, model_wrapper, criterion, optimizer, lr_scheduler, MFLOPS_table)

        # val
        validate(epoch, val_loader, model_wrapper, criterion_val, train_loader, MFLOPS_table)

        # lr_scheduler.step()
        torch.save(
            {
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
            },
            os.path.join(saved_path, 'checkpoint_{}.pt'.format(epoch)))
    return


def main():
    """train and eval model"""
    train_val_test()


if __name__ == "__main__":
    main()