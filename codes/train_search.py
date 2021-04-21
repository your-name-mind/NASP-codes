import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
# import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
# from tensorboard_logger import configure, log_value
import pdb



def main():

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.greedy, args.l2)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=args.num_worker)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=args.num_worker)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    # lr = scheduler.get_lr()[0]
    lr = scheduler.get_last_lr()[0]

    # log_value("lr", lr, epoch)
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
    logging.info(model.alphas_normal)
    logging.info(model.alphas_reduce)

    # training
    start_time = time.time()
    train_acc, train_obj, alphas_time, forward_time, backward_time = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    end_time = time.time()
    logging.info("train time %f", end_time - start_time)
    logging.info("alphas_time %f ", alphas_time)
    logging.info("forward_time %f", forward_time)
    logging.info("backward_time %f", backward_time)
    # log_value('train_acc', train_acc, epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    start_time2 = time.time()
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    end_time2 = time.time()
    logging.info("inference time %f", end_time2 - start_time2)
    # log_value('valid_acc', valid_acc, epoch)
    logging.info('valid_acc %f', valid_acc)
    logging.info('alphas_normal = %s', model.alphas_normal)
    logging.info('alphas_reduce = %s', model.alphas_reduce)

    scheduler.step()

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  alphas_time = 0
  forward_time = 0
  backward_time = 0
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()
    begin1 = time.time()
    architect.step(input, target, input_search, target_search, lr, optimizer)
    model.clip()
    end1 = time.time()
    alphas_time += end1 - begin1
    optimizer.zero_grad()
    model.binarization()
    begin2 = time.time()
    logits = model(input)
    end2 = time.time()
    forward_time += end2 - begin2
    loss = criterion(logits, target)
    
    begin3 = time.time()
    loss.backward()
    end3 = time.time()
    backward_time += end3 - begin3

    model.restore()

    # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # only update selected op's conv_para
    optimizer.step()
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg, alphas_time, forward_time, backward_time


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  model.binarization()
  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda()
      logits = model(input)
      loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  model.restore()
  return top1.avg, objs.avg


if __name__ == '__main__':
  '''第一、 主脚本中的执行函数要放在 if _ name _ =='_ main '：代码块中，以防止多进程启动时主脚本不会再次运行（很可能会产生错误）'''
  parser = argparse.ArgumentParser("cifar") 
  parser.add_argument('--data', type=str, default='/home/ltf/another_code/data', help='location of the data corpus')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
  parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
  parser.add_argument('--layers', type=int, default=8, help='total number of layers')
  parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
  parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
  parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
  parser.add_argument('--save', type=str, default='EXP', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--num_worker', type=int, default=4, help='data loader number of worker')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
  parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--name', type=str, default="runs", help='name for log')
  parser.add_argument('--debug', action='store_true', default=False, help='debug or not')
  parser.add_argument('--greedy', type=float, default=0, help='explore and exploitation')
  parser.add_argument('--l2', type=float, default=0, help='additional l2 regularization for alphas')
  args = parser.parse_args()

  args.save = 'result/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  if args.debug:
    args.save += "_debug"
  if not os.path.exists(args.data):
    args.data = '/automl/dataset/cifar'
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  # configure(args.save + "/%s"%(args.name))

  CIFAR_CLASSES = 10
  main() 

