'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import numpy as np
import random
import math
import os
from argument import parser, print_args, create_logger
import time
from tqdm import tqdm
from custom_models import model, wrn
from utils import *
from build import Encoder, Reconstructor, Classifier, Discriminator
from solver import Solver

def attack_pgd_val(config, extractor, encoder_r, classifier, X, y, norm='l_inf', early_stop=False, mixup=False, y_a=None, y_b=None, lam=None):
    epsilon = (config['epsilon'] / 255.)
    alpha = (config['step_size'] / 255.)
    attack_iters = config['num_steps']
    restarts = config['random_start']

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = classifier(encoder_r(extractor(normalize(X + delta))))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, classifier(encoder_r(extractor(normalize(X+delta)))), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, classifier(encoder_r(extractor(normalize(X+delta)))), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(classifier(encoder_r(extractor(normalize(X+delta)))), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def test(epoch, loader):
    global best_acc, best_adv_acc
    logger.info('\nEpoch: %d' % epoch)
    extractor.eval()
    encoder_r.eval()
    classifier.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, batch in enumerate(iterator):
            inputs, targets = batch['input'], batch['target']
            with torch.no_grad():
                outputs = classifier(encoder_r(extractor(normalize(inputs))))
                loss = criterion_ori(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    # Save checkpoint.
    acc = 100.*correct/total
    logger.info('Test acc: %.3f' % acc)
    logger.info('Test loss: %.3f' % test_loss)
    
    if acc > best_acc:
        logger.info('Saving..')
        state = {
            'ext': extractor.state_dict(),
            'enc_r': encoder_r.state_dict(),
            'enc_nr': encoder_nr.state_dict(),
            'enc_ds': encoder_ds.state_dict(),
            'rec': reconstruct.state_dict(),
            'cls': classifier.state_dict(),
            'dis': discriminator.state_dict(),
            'best_acc': acc,
            'best_adv_acc': best_adv_acc,
            'epoch': epoch+1,
        }
        if not os.path.isdir(args.ckpt_root):
            os.mkdir(args.ckpt_root)
        test_root = os.path.join(args.ckpt_root, 'ckpt.t7')
        torch.save(state, test_root)
        best_acc = acc

def val(epoch, loader):
    global best_acc, best_adv_acc
    logger.info('\nEpoch: %d' % epoch)
    extractor.eval()
    encoder_r.eval()
    classifier.eval()

    val_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(loader, ncols=0, leave=False)
    for batch_idx, batch in enumerate(iterator):
        inputs, targets = batch['input'], batch['target']
        delta = attack_pgd_val(config, extractor=extractor, encoder_r=encoder_r, classifier=classifier, X=inputs, y=targets)
        delta = delta.detach()
        adv_inputs = normalize(torch.clamp(inputs + delta[:inputs.size(0)], min=lower_limit, max=upper_limit))
        outputs = classifier(encoder_r(extractor(adv_inputs)))
        loss = criterion_ori(outputs, targets)
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    # Save checkpoint.
    acc = 100.*correct/total
    logger.info('Val acc: %.3f' % acc)
    logger.info('Val loss: %.3f' % val_loss)
    
    if acc > best_adv_acc:
        logger.info('Saving..')
        state = {
            'ext': extractor.state_dict(),
            'enc_r': encoder_r.state_dict(),
            'enc_nr': encoder_nr.state_dict(),
            'enc_ds': encoder_ds.state_dict(),
            'rec': reconstruct.state_dict(),
            'cls': classifier.state_dict(),
            'dis': discriminator.state_dict(),
            'best_acc': best_acc,
            'best_adv_acc': acc,
            'epoch': epoch+1,
        }
        if not os.path.isdir(args.ckpt_root):
            os.mkdir(args.ckpt_root)
        test_root = os.path.join(args.ckpt_root, 'ckpt_best.t7')
        torch.save(state, test_root)
        best_adv_acc = acc


if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)
    log_name = 'adv-trip'
    logger = create_logger(args.log_root, log_name, 'info')
    print_args(args, logger)

    config = {
        'epsilon': 8.0,
        'num_steps': 20,
        'step_size': 2.0,
        'random_start': 1,
    }

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0  # best test accuracy
    best_adv_acc = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    logger.info('==> Preparing data..')

    transforms = [Crop(32, 32), FlipLR()]
    if args.val:
        try:
            print(os.path.join(args.data_path, f"cifar10_validation_split.pth"))
            dataset = torch.load(os.path.join(args.data_path, f"cifar10_validation_split.pth"))
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        dataset = cifar10(args.data_path)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.), dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=args.workers)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=args.workers)

    enc_out = 640
    rec_in = enc_out*3
    rec_out = 320
    num_classes = 10

    logger.info('==> Building model..')
    extractor = model.WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
    encoder_r = Encoder(depth=34, widen_factor=10, dropRate=0.0)
    encoder_nr = Encoder(depth=34, widen_factor=10, dropRate=0.0)
    encoder_ds = Encoder(depth=34, widen_factor=10, dropRate=0.0)
    reconstruct = Reconstructor(in_planes=rec_in, out_planes=rec_out)
    classifier = Classifier(enc_out, num_classes)
    discriminator = Discriminator(dis_in=enc_out, dis_mid=enc_out)

    extractor = extractor.to(device)
    encoder_r = encoder_r.to(device)
    encoder_nr = encoder_nr.to(device)
    encoder_ds = encoder_ds.to(device)
    reconstruct = reconstruct.to(device)
    classifier = classifier.to(device)
    discriminator = discriminator.to(device)

    logger.info(config) 
    if device == 'cuda':
        extractor = torch.nn.DataParallel(extractor)
        encoder_r = torch.nn.DataParallel(encoder_r)
        encoder_nr = torch.nn.DataParallel(encoder_nr)
        encoder_ds = torch.nn.DataParallel(encoder_ds)
        reconstruct = torch.nn.DataParallel(reconstruct)
        classifier = torch.nn.DataParallel(classifier)
        discriminator = torch.nn.DataParallel(discriminator)
        cudnn.benchmark = True

    solver = Solver(args, trainloader=train_batches, extractor=extractor, encoder_r=encoder_r, encoder_nr=encoder_nr, encoder_ds=encoder_ds, reconstruct=reconstruct, classifier=classifier, discriminator=discriminator, device=device, logger=logger)

    criterion_ori = nn.CrossEntropyLoss()

    if args.resume:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        resume_dir = args.ckpt_root + '0'
        assert os.path.isdir(resume_dir), 'Error: no checkpoint directory found!'
        resume_root = os.path.join(resume_dir, 'ckpt.t7')
        checkpoint = torch.load(resume_root)
        best_acc = checkpoint['best_acc']
        resume_root = os.path.join(resume_dir, 'ckpt_best.t7')
        checkpoint = torch.load(resume_root)
        best_adv_acc = checkpoint['best_adv_acc']
        resume_root = os.path.join(resume_dir, 'ckpt_latest.t7')
        checkpoint = torch.load(resume_root)
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, args.num_epoches):
        solver.train_epoch(epoch)
        if args.val:
            val(epoch, val_batches)
