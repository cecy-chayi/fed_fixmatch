import argparse
import copy
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

class Client:
    def __init__(self, args, client_id, model, labeled_trainloader, unlabeled_trainloader):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.model.to(args.device)
        self.labeled_trainloader = labeled_trainloader
        self.unlabeled_trainloader = unlabeled_trainloader

    def train(self, args, global_model):
        if args.amp:
            from apex import amp

        end = time.time()

        if args.world_size > 1:
            labeled_epoch = 0
            unlabeled_epoch = 0
            self.labeled_trainloader.sampler.set_epoch(labeled_epoch)
            self.unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

        self.model.load_state_dict(global_model.state_dict())
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, args.warmup, args.total_steps)

        for epoch in range(args.local_ep):
            self.model.train()
            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            mask_probs = AverageMeter()
            if not args.no_progress:
                p_bar = tqdm(range(args.eval_step),
                             disable=args.local_rank not in [-1, 0])
            for batch_idx in range(args.eval_step):
                try:
                    # inputs_x, targets_x = labeled_iter.next()
                    # error occurs ↓
                    inputs_x, targets_x = next(labeled_iter)
                except:
                    if args.world_size > 1:
                        labeled_epoch += 1
                        self.labeled_trainloader.sampler.set_epoch(labeled_epoch)
                    labeled_iter = iter(self.labeled_trainloader)
                    # inputs_x, targets_x = labeled_iter.next()
                    # error occurs ↓
                    inputs_x, targets_x = next(labeled_iter)

                try:
                    # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                    # error occurs ↓
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except:
                    if args.world_size > 1:
                        unlabeled_epoch += 1
                        self.unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                    # error occurs ↓
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]
                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
                targets_x = targets_x.to(args.device)
                logits = self.model(inputs)
                logits = de_interleave(logits, 2 * args.mu + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()

                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                      reduction='none') * mask).mean()

                loss = Lx + args.lambda_u * Lu

                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                batch_time.update(time.time() - end)
                end = time.time()
                mask_probs.update(mask.mean().item())

                if not args.no_progress:
                    p_bar.set_description(
                        "Client{client_id} Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                            client_id=self.client_id,
                            epoch=epoch + 1,
                            epochs=args.local_ep,
                            batch=batch_idx + 1,
                            iter=args.eval_step,
                            lr=scheduler.get_last_lr()[0],
                            data=data_time.avg,
                            bt=batch_time.avg,
                            loss=losses.avg,
                            loss_x=losses_x.avg,
                            loss_u=losses_u.avg,
                            mask=mask_probs.avg))
                    p_bar.update()

            if not args.no_progress:
                p_bar.close()

        return self.model.state_dict(), losses.avg, losses_x.avg, losses_u.avg, mask_probs.avg
