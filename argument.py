import os
import argparse
import logging

def parser():

    parser = argparse.ArgumentParser(description='PyTorch tiny-imagenet Training')
    parser.add_argument('--data_path', default='', type=str, help='path for input data')
    parser.add_argument('--num_epoches', default=200, type=int, help='number of total epoches')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of workers in dataloader')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay1', default=100, type=int, help='first learning rate decay point')
    parser.add_argument('--lr_decay2', default=150, type=int, help='second learning rate decay point')
    parser.add_argument('--lr_decay3', default=200, type=int, help='third learning rate decay point')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--adversarial_test', '-t', action='store_true', help='adversarial test')
    parser.add_argument('--val', default=False, type=bool, help='if use validation set')
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--log_root', default='./log', help='the directory to save the logs')
    parser.add_argument('--ckpt_root', default='./checkpoint', help='the directory to save the ckeckpoints')

    parser.add_argument('--grl_const', default=0.5, type=float, help='gradient reversal layer')
    parser.add_argument('--loss_func', default='xent', help='loss used during generating PGD attack')
    parser.add_argument('--epsilon_min', default=8, type=int, help='minimum epsilon used in PGD attack')
    parser.add_argument('--epsilon_max', default=12, type=int, help='maximum epsilon used in PGD attack')
    parser.add_argument('--num_steps_min', default=1, type=int, help='minimum number of steps in PGD attack')
    parser.add_argument('--num_steps_max', default=4, type=int, help='maximum number of steps in PGD attack')
    parser.add_argument('--step_size_min', default=2, type=int, help='minimum step size in PGD attack')
    parser.add_argument('--step_size_max', default=4, type=int, help='maximum step size in PGD attack')
    parser.add_argument('--random_start', default=1, type=int, help='if use random start in PGD attack')
    parser.add_argument('--lb_smooth', default=0, type=float, help='label smooth')

    args = parser.parse_args()

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

