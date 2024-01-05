import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger:logging.Logger = None
tb_logger:SummaryWriter = None
args = {}

def setup():
    # 项目路径
    root_path = Path('./')
    data_path = root_path.joinpath('dataset')

    # 结果记录
    result_path = root_path.joinpath('result')
    experiment_path = result_path.joinpath(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    saved_model_path = experiment_path.joinpath('saved_model')
    tb_path = experiment_path.joinpath('tb_logs')
    tb_path.mkdir(exist_ok=True, parents=True)
    saved_model_path.mkdir(exist_ok=True, parents=True)

    resume = None

    # 设置log
    global logger 
    logger = logging.getLogger()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(experiment_path.joinpath('log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    # 设置tensorboard
    global tb_logger
    tb_logger = SummaryWriter(log_dir=tb_path, purge_step=1)
    
    global args
    args = {
        'data_path': data_path,
        'experiment_path': experiment_path,
        'saved_model_path': saved_model_path,
        'seed' : 3407,
        'test_interval' : 5,
        'DEVICE': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'EPOCH' : 200,
        'BATCH_SIZE': 128,
        'LR' : 1e-4,
        'd_model' : 512,
        'd_hidden' : 1024,
        'q' : 8,
        'v' : 8,
        'h' : 8,
        'N' : 6,
        'dropout' : 0.2,
        'pe' : True,
        'mask' : True,
        'use_smote' : False,
        'loss_function' : 'cross_entropy',
        'optimizer_name' : 'Adam',
        'weight_decay' : 1e-4,
        'logit_adj_train' : True,
        'tro_train' : 1,
        'resume' : resume,
        'get_submission' : False,
        'k_folds' : 5,
        'patience' : 40,
        'GPU NUM' : 2,
    }

    if args['GPU NUM'] > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args['local_rank'] = local_rank
        args['DEVICE'] = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    
    if local_rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logger.info(f'args: {args}')
    setup_seed(args['seed']) 


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ListMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = []
        self.reset()

    def reset(self):
        self.val = []
        self.max = 0
        self.min = 0
    
    def restore(self, val):
        self.val = val
        self.max = max(self.val)
        self.min = min(self.val)

    def update(self, val, n=1):
        self.val.append(val)
        self.max = max(self.val)
        self.min = min(self.val)


setup()