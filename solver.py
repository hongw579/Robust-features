from __future__ import print_function
import torch
import sys
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from argument import print_args
from utils import *

def attack_pgd(args, extractor, encoder_r, classifier, X, y, norm='l_inf', early_stop=False, mixup=False, y_a=None, y_b=None, lam=None):
    epsilon = (random.randint(args.epsilon_min, args.epsilon_max) / 255.)
    alpha = (random.randint(args.step_size_min, args.step_size_max) / 255.)
    attack_iters = random.randint(args.num_steps_min, args.num_steps_max)*8
    restarts = args.random_start

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

class Solver():
    def __init__(self, args, trainloader, extractor, encoder_r, encoder_nr, encoder_ds, reconstruct, classifier, discriminator, device, logger):

        self.args = args
        self.logger = logger
        self.device = device
        self.trainloader = trainloader

        self.ext = extractor
        self.enc_r = encoder_r
        self.enc_nr = encoder_nr
        self.enc_ds = encoder_ds
        self.rec = reconstruct
        self.cls = classifier
        self.dis = discriminator
        self.kl = nn.KLDivLoss(reduction = 'none')
        self.kl_coeff = 1.
        self.cls_r_coeff = 1.

        # All modules in the same dict
        self.modules = nn.ModuleDict({
            'ext': self.ext, 'enc_r': self.enc_r, 'enc_nr': self.enc_nr, 'enc_ds': self.enc_ds, 'rec': self.rec, 'cls': self.cls, 'dis': self.dis
        })

        self.xent_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.set_optimizer()
        self.set_scheduler()

        if args.resume:
            # Load checkpoint.
            logger.info('==> Resuming from checkpoint..')
            resume_dir = args.ckpt_root + '0'
            assert os.path.isdir(resume_dir), 'Error: no checkpoint directory found!'
            resume_root = os.path.join(resume_dir, 'ckpt_latest.t7')
            checkpoint = torch.load(resume_root)
            for k in self.modules.keys():
                self.modules[k].load_state_dict(checkpoint[k])

    def set_optimizer(self):
        self.logger.info('optimizer has momentum!')
        self.opt_main = optim.SGD(list(self.ext.parameters())+list(self.enc_r.parameters())+list(self.cls.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.opt = {
            'ext': optim.SGD(self.ext.parameters(), lr=self.args.lr/100., momentum=0.9, weight_decay=5e-4),
            'enc_r': optim.SGD(self.enc_r.parameters(), lr=self.args.lr/100., momentum=0.9, weight_decay=5e-4),
            'enc_nr': optim.SGD(self.enc_nr.parameters(), lr=self.args.lr/100., momentum=0.9, weight_decay=5e-4),
            'enc_ds': optim.SGD(self.enc_ds.parameters(), lr=self.args.lr/100., momentum=0.9, weight_decay=5e-4),
            'rec': optim.SGD(self.rec.parameters(), lr=self.args.lr/100., momentum=0.9, weight_decay=5e-4),
            'cls': optim.SGD(self.cls.parameters(), lr=self.args.lr/100., momentum=0.9, weight_decay=5e-4),
            'dis': optim.SGD(self.dis.parameters(), lr=self.args.lr/5., momentum=0.9, weight_decay=5e-4),
            'dis_ext': optim.SGD(self.ext.parameters(), lr=self.args.lr/10., momentum=0.9, weight_decay=5e-4),
            'dis_enc_r': optim.SGD(self.enc_r.parameters(), lr=self.args.lr/10., momentum=0.9, weight_decay=5e-4),
            'dis_enc_nr': optim.SGD(self.enc_nr.parameters(), lr=self.args.lr/10., momentum=0.9, weight_decay=5e-4),
            'dis_enc_ds': optim.SGD(self.enc_ds.parameters(), lr=self.args.lr/10., momentum=0.9, weight_decay=5e-4),
        }

    def set_scheduler(self):
        self.scheduler = dict()
        for key in ['ext', 'enc_r', 'enc_nr', 'enc_ds', 'rec', 'cls', 'dis', 'dis_ext', 'dis_enc_r', 'dis_enc_nr', 'dis_enc_ds']:
            self.scheduler[key] = optim.lr_scheduler.MultiStepLR(self.opt[key], milestones = [self.args.lr_decay1, self.args.lr_decay2, self.args.lr_decay3], gamma = 0.1)
        self.scheduler_main = optim.lr_scheduler.MultiStepLR(self.opt_main, milestones = [self.args.lr_decay1, self.args.lr_decay2, self.args.lr_decay3], gamma = 0.1)

    def reset_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()
        self.opt_main.zero_grad()

    def group_opt_step(self, opt_keys):
        for k in opt_keys:
            self.opt[k].step()
        self.reset_grad()

    def main_opt_step(self):
        self.opt_main.step()
        self.reset_grad()

    def get_lr(self):
        lr = []
        for _, opt in self.opt.items():
            for param_group in opt.param_groups:
                lr.append(param_group['lr'])
        for param_group in self.opt_main.param_groups:
             lr.append(param_group['lr'])
        return lr

    def optimize_classifier_r(self, image, target):
        logits = self.cls(self.enc_r(self.ext(image)))
        loss = self.cls_r_coeff*self.xent_loss(logits, target)
        loss.backward()
        self.main_opt_step()
        return loss, logits
    
    def optimize_classifier_nr(self, image, target):
        logits = self.cls(self.enc_nr(self.ext(image)))
        if self.args.lb_smooth:
            loss = softmax_crossentropy_labelsmooth(logits, target, lb_smooth=self.args.lbs)
        else:
            loss = self.xent_loss(logits, target)
        loss.backward()
        self.group_opt_step(['enc_nr'])
        return loss
    
    def optimize_classifier_nr_adv(self, image, target):
        logits = self.cls(self.enc_nr(self.ext(image), grl=True, constant=self.args.grl_const))
        if self.args.lb_smooth:
            loss = softmax_crossentropy_labelsmooth(logits, target, lb_smooth=self.args.lbs)
        else:
            loss = self.xent_loss(logits, target)
        loss.backward()
        self.group_opt_step(['enc_nr'])
        return loss
    
    def robust_feat_distance(self, img_nat, img_adv):
        feat_nat_r = self.enc_r(self.ext(img_nat))
        feat_adv_r = self.enc_r(self.ext(img_adv))
        loss_cos = (1-F.cosine_similarity(feat_nat_r, feat_adv_r)).mean()
        loss_cos.backward()
        self.group_opt_step(['ext', 'enc_r'])
        return loss_cos

    def kl_maximizer(self, img_nat, img_adv):
        r_nat, r_adv = self.enc_r(self.ext(img_nat)), self.enc_r(self.ext(img_adv))
        nr_nat, nr_adv = self.enc_nr(self.ext(img_nat)), self.enc_nr(self.ext(img_adv))
        ds_nat, ds_adv = self.enc_ds(self.ext(img_nat)), self.enc_ds(self.ext(img_adv))

        kl_adv_r_nr = torch.exp(-self.kl(F.log_softmax(r_adv, dim=1), F.softmax(nr_adv, dim=1)).sum(dim=(1,2,3))).mean()
        kl_adv_nr_ds = torch.exp(-self.kl(F.log_softmax(nr_adv, dim=1), F.softmax(ds_adv, dim=1)).sum(dim=(1,2,3))).mean()
        kl_adv_ds_r = torch.exp(-self.kl(F.log_softmax(ds_adv, dim=1), F.softmax(r_adv, dim=1)).sum(dim=(1,2,3))).mean()

        kl_nat_r_nr = torch.exp(-self.kl(F.log_softmax(r_nat, dim=1), F.softmax(nr_nat, dim=1)).sum(dim=(1,2,3))).mean()
        kl_nat_nr_ds = torch.exp(-self.kl(F.log_softmax(nr_nat, dim=1), F.softmax(ds_nat, dim=1)).sum(dim=(1,2,3))).mean()
        kl_nat_ds_r = torch.exp(-self.kl(F.log_softmax(ds_nat, dim=1), F.softmax(r_nat, dim=1)).sum(dim=(1,2,3))).mean()

        loss_kl = (kl_adv_r_nr+kl_adv_nr_ds+kl_adv_ds_r+kl_nat_r_nr+kl_nat_nr_ds+kl_nat_ds_r) * self.kl_coeff / 6
        loss_kl.backward()
        self.group_opt_step(['enc_r', 'enc_nr', 'enc_ds'])

        return kl_adv_r_nr, kl_adv_nr_ds, kl_adv_ds_r, kl_nat_r_nr, kl_nat_nr_ds, kl_nat_ds_r

    def discriminator_adversarial(self, img_nat, img_adv):

        self.nat_domain_code = np.repeat(np.array([[*([1]), *([0])]]), img_nat.shape[0], axis=0)
        self.adv_domain_code = np.repeat(np.array([[*([0]), *([1])]]), img_adv.shape[0], axis=0)
        self.nat_domain_code = torch.FloatTensor(self.nat_domain_code).to(self.device)
        self.adv_domain_code = torch.FloatTensor(self.adv_domain_code).to(self.device)

        self.neg_domain_code = np.repeat(np.array([[*([0.5]), *([0.5])]]), img_nat.shape[0], axis=0)
        self.neg_domain_code = torch.FloatTensor(self.neg_domain_code).to(self.device)

        # FD should guess if the features extracted f_di = DI(G(im))
        # are from target or source domain. To win this game and fool FD,
        # DI should extract domain invariant features.

        # Loss measures features' ability to fool the discriminator
        nat_ds_domain = self.dis(self.enc_ds(self.ext(img_nat)))
        adv_ds_domain = self.dis(self.enc_ds(self.ext(img_adv)))
        ds_dis_nat = self.adv_loss(nat_ds_domain, self.nat_domain_code)
        ds_dis_adv = self.adv_loss(adv_ds_domain, self.adv_domain_code)

        nat_r_domain = self.dis(self.enc_r(self.ext(img_nat)))
        adv_r_domain = self.dis(self.enc_r(self.ext(img_adv)))
        r_dis_nat = self.adv_loss(nat_r_domain, self.nat_domain_code)
        r_dis_adv = self.adv_loss(adv_r_domain, self.adv_domain_code)

        nat_nr_domain = self.dis(self.enc_nr(self.ext(img_nat)))
        adv_nr_domain = self.dis(self.enc_nr(self.ext(img_adv)))
        nr_dis_nat = self.adv_loss(nat_nr_domain, self.nat_domain_code)
        nr_dis_adv = self.adv_loss(adv_nr_domain, self.adv_domain_code)

        alignment_loss1 = (ds_dis_nat+ds_dis_adv+r_dis_nat+r_dis_adv+nr_dis_nat+nr_dis_adv)/6.
        alignment_loss1.backward()
        self.group_opt_step(['dis'])

        # Measure discriminator's ability to classify source from target samples
        nat_ds_domain_pred = self.dis(self.enc_ds(self.ext(img_nat)))
        adv_ds_domain_pred = self.dis(self.enc_ds(self.ext(img_adv)))
        ds_loss_nat = self.adv_loss(nat_ds_domain_pred, self.nat_domain_code)
        ds_loss_adv = self.adv_loss(adv_ds_domain_pred, self.adv_domain_code)

        nat_r_domain_pred = self.dis(self.enc_r(self.ext(img_nat)))
        adv_r_domain_pred = self.dis(self.enc_r(self.ext(img_adv)))
        r_loss_nat = self.adv_loss(nat_r_domain_pred, self.neg_domain_code)
        r_loss_adv = self.adv_loss(adv_r_domain_pred, self.neg_domain_code)

        nat_nr_domain_pred = self.dis(self.enc_nr(self.ext(img_nat)))
        adv_nr_domain_pred = self.dis(self.enc_nr(self.ext(img_adv)))
        nr_loss_nat = self.adv_loss(nat_nr_domain_pred, self.neg_domain_code)
        nr_loss_adv = self.adv_loss(adv_nr_domain_pred, self.neg_domain_code)

        alignment_loss2 = (ds_loss_nat+ds_loss_adv+r_loss_nat+r_loss_adv+nr_loss_nat+nr_loss_adv)/6.
        alignment_loss2.backward()
        self.group_opt_step(['dis_ext', 'dis_enc_r', 'dis_enc_nr', 'dis_enc_ds'])
        return ds_dis_nat, ds_dis_adv, r_dis_nat, r_dis_adv, nr_dis_nat, nr_dis_adv, ds_loss_nat, ds_loss_adv, r_loss_nat, r_loss_adv, nr_loss_nat, nr_loss_adv

    def optimize_rec(self, img_nat, img_adv):
        _feat_nat = self.ext(img_nat)
        _feat_adv = self.ext(img_adv)

        feat_nat_r = self.enc_r(_feat_nat)
        feat_adv_r = self.enc_r(_feat_adv)

        feat_nat_nr = self.enc_nr(_feat_nat)
        feat_adv_nr = self.enc_nr(_feat_adv)

        feat_nat_ds = self.enc_ds(_feat_nat)
        feat_adv_ds = self.enc_ds(_feat_adv)

        rec_nat = self.rec(torch.cat([feat_nat_r, feat_nat_nr, feat_nat_ds], 1))
        rec_adv = self.rec(torch.cat([feat_adv_r, feat_adv_nr, feat_nat_ds], 1))

        recon_loss_nat = F.l1_loss(_feat_nat, rec_nat)
        recon_loss_adv = F.l1_loss(_feat_adv, rec_adv)

        recon_loss = recon_loss_nat + recon_loss_adv

        recon_loss.backward()
        self.group_opt_step(['enc_r', 'enc_nr', 'enc_ds', 'rec'])
        return recon_loss_nat, recon_loss_adv

    def train_epoch(self, epoch):
        self.logger.info('\nEpoch: %d' % epoch)
        correct = 0
        total = 0
        iterator = tqdm(self.trainloader, ncols=0, leave=False)

        self.logger.info(self.get_lr())

        for batch_idx, batch in enumerate(iterator):
            input1, target1 = batch['input'], batch['target']
            delta = attack_pgd(self.args, extractor=self.ext, encoder_r=self.enc_r, classifier=self.cls, X=input1, y=target1)
            delta = delta.detach()
            adv_input1 = normalize(torch.clamp(input1 + delta[:input1.size(0)], min=lower_limit, max=upper_limit))
            nat_input1 = normalize(input1)
            # set training
            for k in self.modules.keys():
                self.modules[k].train()
            self.reset_grad()

            class_loss_r, r_logits = self.optimize_classifier_r(adv_input1, target1)
            class_loss_nr_nat  = self.optimize_classifier_nr(nat_input1, target1)
            class_loss_nr_adv = self.optimize_classifier_nr_adv(adv_input1, target1)
            cosine_loss = self.robust_feat_distance(nat_input1, adv_input1)
            kl_loss = self.kl_maximizer(nat_input1, adv_input1)
            dis_loss_ds  = self.discriminator_adversarial(nat_input1, adv_input1)
            rec_loss_nat, rec_loss_adv = self.optimize_rec(nat_input1, adv_input1)

            _, predicted = r_logits.max(1)
            total += target1.size(0)
            correct += predicted.eq(target1).sum().item()
            iterator.set_description(str(predicted.eq(target1).sum().item()/target1.size(0)))

        self.scheduler_main.step()
        for _, sched in self.scheduler.items():
            sched.step()
        acc = 100.*correct/total
        self.logger.info('Train acc: %.3f' % acc)

        state_latest = dict()
        for k in self.modules.keys():
            state_latest[k] = self.modules[k].state_dict()
        state_latest['epoch'] = epoch+1

        opt_latest = dict()
        for k in self.opt.keys():
            opt_latest[k] = self.opt[k].state_dict()
        opt_latest['main'] = self.opt_main.state_dict()

        if not os.path.isdir(self.args.ckpt_root):
            os.mkdir(self.args.ckpt_root)
        train_root = os.path.join(self.args.ckpt_root, 'ckpt_latest.t7')
        torch.save(state_latest, train_root)
        train_root = os.path.join(self.args.ckpt_root, 'opt_latest.t7')
        torch.save(opt_latest, train_root)

        return batch_idx

