import numpy as np
import math
import pandas as pd
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset_process.dataset_process_csv import get_dataset, get_val_dataset
from module.cnn import CNN
from module.loss import *
from module.schedule import *
from module.transformer import Transformer
from utils.utils import *


def train():
    models = []
    best_metrics = []
    for train_dataset, test_dataset in get_dataset():
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if args['GPU NUM'] > 1 else None
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if args['GPU NUM'] > 1 else None
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args['BATCH_SIZE'], shuffle=True if train_sampler is None else None, sampler=train_sampler)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args['BATCH_SIZE'], sampler=test_sampler)

        d_input = train_dataset.input_len  # 时间步数量
        d_channel = train_dataset.channel_len  # 时间序列维度
        d_output = train_dataset.output_len  # 分类类别

        # 创建loss函数 
        if args['loss_function'] == 'focal_loss':
            criterion = focal_loss
        elif args['loss_function'] == 'cross_entropy':
            criterion = Myloss(args['logit_adj_train'], train_dataloader, args['tro_train'])
        
        # for model_type in ['transformer', 'cnn']:
        for model_type in ['cnn', 'transformer']:
            
            if model_type == 'transformer':
                model = Transformer(d_model=args['d_model'], d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=args['d_hidden'], q=args['q'], v=args['v'], h=args['h'], N=args['N'], dropout=args['dropout'], pe=args['pe'], mask=args['mask'], device=args['DEVICE']).to(args['DEVICE'])
            else:
                model = CNN().to(args['DEVICE'])
            
            if args['GPU NUM'] > 1:
                model = DDP(model, device_ids=[args['local_rank']], output_device=args['local_rank'])

            index = len(models)
            models.append(model)
            best_metrics.append(1e9)

            logger.info(f'_____________model type: {model_type}\tindex: {index}_____________')

            if args['resume'] is not None:
                if f'{model_type}_{index}_best.pkl' in os.listdir(args['resume']):
                        model.load_state_dict(torch.load(args['resume'].joinpath(f'{model_type}_{index}_best.pkl'))['model'])
                        logger.info(f'load {model_type}_{index}_best.pkl successfully')
                        models[index] = model
                        current_accuracy, current_metric = accuracy(model, test_dataloader, 0, 'test_set')
                        best_metrics[index] = current_metric
                        logger.info(f'当前最大accuracy\t测试集:{current_accuracy:.2%}')
                        logger.info(f'当前最小metric\t测试集:{current_metric:.02f}')
                        if args['local_rank'] == 0:
                            torch.save({'model': model.state_dict()}, args['saved_model_path'].joinpath(f'{model_type}_{index}_best.pkl'))
                        model.to('cpu')
                        continue

            # 创建优化器
            optimizer = getattr(optim, args['optimizer_name'])(model.parameters(), lr=args['LR'], weight_decay=args['weight_decay'])

            # 创建学习率调整器
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, t_total=100, cycles=0.5)

            # 用于记录准确率变化
            correct_on_test = ListMeter('Correct_on_test', ':.2e')
            metrics_on_test = ListMeter('Metrics_on_test', ':.2e')
            start_epoch = 0

            no_up = 0
            for epoch in range(start_epoch, args['EPOCH']):
                output, loss = train_epoch(model, train_dataloader, optimizer, criterion)

                logger.info(f'Epoch:{epoch + 1}:\tloss: {loss:.04f}\tlr: {scheduler.get_last_lr()[0]:.06f}')
                tb_logger.add_scalar(f'train_loss', loss, epoch + 1)
                tb_logger.add_scalar(f'lr', scheduler.get_last_lr()[0], epoch + 1)

                scheduler.step()

                current_accuracy, current_metric = accuracy(model, test_dataloader, epoch, 'test_set')

                correct_on_test.update(current_accuracy)
                metrics_on_test.update(current_metric)

                logger.info(f'当前最大accuracy\t测试集:{correct_on_test.max:.2%}')
                logger.info(f'当前最小metric\t测试集:{metrics_on_test.min:.02f}')

                if math.isclose(current_metric, metrics_on_test.min):
                    no_up = 0
                    if args['local_rank'] == 0:
                        torch.save({'model': model.state_dict()}, args['saved_model_path'].joinpath(f'{model_type}_{index}_best.pkl'))
                        models[index] = model
                        best_metrics[index] = current_metric
                else:
                    no_up += 1
                    if no_up == args['patience'] and model_type == 'transformer':
                        logger.info(f'Early Stopping at Epoch:{epoch + 1}')
                        break
            model.to('cpu')

    if args['local_rank'] == 0:
        get_submission(models, best_metrics)
    


def train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()

    losses = AverageMeter('Loss', ':.4e')
    for (x, y) in train_dataloader:
        output = model(x, 'train')

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.shape[0])
    
    return output, losses.avg

@torch.no_grad()
def get_submission(models, best_metrics):
    val_dataset = get_val_dataset()
    dataloader = DataLoader(dataset=val_dataset, batch_size=args['BATCH_SIZE'], shuffle=False)
    results = []

    for model in models:
        model.to(args['DEVICE'])
        result = None
        model.eval()
        for x, y in dataloader:
            y_pre = model(x, 'test')

            y_pre = torch.softmax(y_pre, dim=1)
            if result is None:
                result = y_pre
            else:
                result = torch.cat((result, y_pre), dim=0)

        results.append(result)
        model.to('cpu')
    
    # best_metrics将作为权重，越小权重越大，权重和为1
    weights = [1 / i for i in best_metrics]
    weights = [i / sum(weights) for i in weights]
    result = results[0] * weights[0]
    for i in range(1, len(results)):
        result += results[i] * weights[i]
    
    max_indices = torch.argmax(result, axis=1)
    y_nor = torch.zeros_like(result, device=args['DEVICE'])
    mask = result.max(dim=1).values <= 0.5
    y_nor[mask] = result[mask]
    y_nor[~mask, max_indices[~mask]] = 1
    y = range(80000, 100000)
    result = list(zip(y, *y_nor.T.data.tolist()))

    # 保存到文件
    result = pd.DataFrame(np.array(result), columns=['id','label_0','label_1','label_2','label_3'])
    result = result.set_index('id', drop=True)
    result.to_csv(args["experiment_path"].joinpath('submission.csv'), mode='w')


# 测试函数
def accuracy(model, dataloader, epoch, flag='test_set'):
    correct = AverageMeter('Correct', ':.2e')
    metric = AverageMeter('Metric', ':.2e')
    correct_class = [AverageMeter(f'Correct_{i}', ':.2e') for i in range(4)]
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            y_pre = model(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            batch_size = x.shape[0]
            correct.update((label_index == y.long()).sum().item(), batch_size)

            for (i, j) in zip(label_index, y.long()):
                if i == j:
                    correct_class[j].update(1, 1)
                else:
                    correct_class[j].update(0, 1)
            
            y_label = torch.softmax(y_pre, dim=1)
            y = torch.nn.functional.one_hot(y, 4).float()
            metric.update((y_label - y).abs().sum().item(), batch_size)
        
        correct = correct.avg
        correct_class = [i.avg for i in correct_class]
        metric = metric.sum

        metric = torch.tensor(metric, dtype=torch.float32, device=args['DEVICE'])
        torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
        metric = metric.item()
            
        logger.info(f'Epoch:{epoch + 1}:\tAccuracy on {flag}: {correct:.2%}')
        logger.info(f'Epoch:{epoch + 1}:\tAccuracy on {flag} for class 0, 1, 2, 3: {correct_class[0]:.2%}, {correct_class[1]:.2%}, {correct_class[2]:.2%}, {correct_class[3]:.2%}')
        logger.info(f'Epoch:{epoch + 1}:\tMetric on {flag}: {metric:.2f}')
        tb_logger.add_scalars(f'acc_{flag}', {'a':correct, '0':correct_class[0], '1':correct_class[1], '2':correct_class[2], '3':correct_class[3]}, epoch)
        tb_logger.add_scalar(f'metric_{flag}', metric, epoch)

        return correct, metric


def main():
    if args['get_submission']:
        get_submission()
    else:
        train()


if __name__ == '__main__':
    main()