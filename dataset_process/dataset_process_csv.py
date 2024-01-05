from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from os import path
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import pdist, squareform
from utils.utils import logger, args

def get_dataset():
    data_path = args['data_path']
    use_smote = args['use_smote']
    device = args['DEVICE']

    # 读取训练数据和测试数据并展示头部信息
    data = pd.read_csv(path.join(data_path, 'train.csv'))
    # 将数据简单处理：数据切割以及打标签
    train_list = []
    for items in data.values:
        train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])
    data = pd.DataFrame(np.array(train_list))
    data.columns = ['id'] + [str(i) for i in range(len(train_list[0]) - 2)] + ['label']

    if use_smote:
        smote = SMOTE(random_state=args['seed'])
        k_x_train, k_y_train = smote.fit_resample(data.iloc[:, 1:data.shape[1] - 1].values, data['label'].values) 
        logger.info(f"after smote, k_x_train.shape: {k_x_train.shape}, k_y_train.shape: {k_y_train.shape}")
        new_data = pd.DataFrame(k_x_train, columns=data.columns[1:data.shape[1] - 1])
        new_data = new_data.reset_index()
        new_data = new_data.rename(columns={'index': 'id'})
        new_data['label'] = k_y_train
        data = new_data
    # 将数据打乱 为了划分数据集的随机性
    data = data.sample(frac=1.0).reset_index(drop=True)

    # 数据增强
    fussion_data_1 = my_data_augmentation(data, k=3, thread=0.95, label=1, diff=False)
    fussion_data_2 = my_data_augmentation(data, k=3, thread=0.95, label=1, diff=True)

    # ori_data_len = data.shape[0]

    data = pd.concat([data, fussion_data_1, fussion_data_2], axis=0, ignore_index=True)
    data = data.sample(frac=1.0).reset_index(drop=True) #XXX 应不应该让增强数据进入测试集？
    data['id'] = range(data.shape[0])
    
    for k in range(args['k_folds']):
        # 将data分割为train 和 test
        frac = 1 / args['k_folds']
        test_begin = int(data.shape[0] * k * frac)
        test_end = int(data.shape[0] * (k + 1) * frac)
        test = data.iloc[test_begin:test_end]
        train = pd.concat([data.iloc[:test_begin], data.iloc[test_end:]], axis=0)
        
        logger.info(f"test_begin: {test_begin}, test_end: {test_end}")
        logger.info(f"train_begin_1: {0}, train_end_1: {test_begin}, train_begin_2: {test_end}, train_end_2: {data.shape[0]}")

        train_dataset, test_dataset = MyDataset(train, device), MyDataset(test, device)
        yield train_dataset, test_dataset


def get_val_dataset():
    data_path = args['data_path']
    device = args['DEVICE']

    data = pd.read_csv(path.join(data_path, 'test.csv'))
    # 将数据简单处理：数据切割以及打标签
    train_list = []
    for items in data.values:
        train_list.append([items[0]] + [float(i) for i in items[1].split(',')])
    data = pd.DataFrame(np.array(train_list))
    data.columns = ['id'] + [str(i) for i in range(len(train_list[0]) - 1)]

    val_dataset = MyDataset(data, device)

    return val_dataset


def my_data_augmentation(data, k = 3, thread = 0.95, label = 1, diff = False):
    data_1 = data[data['label'] == label]
    data_1 = data_1.iloc[:, 1:data_1.shape[1]-1]
    if diff == True:
        data_1 = data_1.diff(axis=1).dropna(axis=1)

    cosine_distances = pdist(data_1, metric='cosine')
    cosine_similarity = 1 - squareform(cosine_distances)
    similarity_df = pd.DataFrame(cosine_similarity, index=data_1.index, columns=data_1.index)

    fussion_df = []
    set_diffsion = set()
    for i in similarity_df.index:
        most_similarity_df = similarity_df[i].sort_values(ascending=False).iloc[1:1 + k].index
        for j in most_similarity_df:
            if (i, j) in set_diffsion:
                continue
            if similarity_df[i][j] < thread:
                break
            new_row = (data.loc[i].iloc[1:data.shape[1]-1] + data.loc[j].iloc[1:data.shape[1]-1]) / 2
            fussion_df.append(new_row.values.tolist())
            set_diffsion.add((j, i))
    logger.info(f"label: {label}, diff: {diff}, k: {k}, thread:{thread}, data_augmentation number: {len(fussion_df)}")
    fussion_df = pd.DataFrame(fussion_df)
    fussion_df.columns = [str(i) for i in range(len(fussion_df.columns))]
    fussion_df['label'] = label
    return fussion_df



class MyDataset(Dataset):
    def __init__(self, data, device):
        """
        训练数据集与测试数据集的Dataset对象
        :param path: 数据集路径
        :param dataset: 区分是获得训练集还是测试集
        """
        super(MyDataset, self).__init__()
        self.device = device
        self.dataset_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.dataset, \
        self.label = self.pre_option(data)

    def __getitem__(self, index):
        return self.dataset[index], self.label[index]

    def __len__(self):
        return self.dataset_len


    # 数据预处理
    def pre_option(self, data: pd.DataFrame):
        """
        :param path: 数据集路径
        :return: 训练集样本数量，测试集样本数量，时间步维度，通道数，分类数，训练集数据，训练集标签，测试集数据，测试集标签，测试集中时间步最长的样本列表，没有padding的训练集数据
        """

        # row, time_len, channel
        if 'label' in data.columns:
            dataset = torch.tensor(data.iloc[:, 1:data.shape[1] - 1].values, dtype=torch.float32, device=self.device).unsqueeze(2)
            label = torch.tensor(data['label'].values, dtype=torch.long, device=self.device)
        else:
            label = torch.tensor(data['id'].values, dtype=torch.long, device=self.device)
            dataset = torch.tensor(data.iloc[:, 1:data.shape[1]].values, dtype=torch.float32, device=self.device).unsqueeze(2)

        dataset_len, input_len, channel_len = dataset.shape
        output_len = 4

        return dataset_len, input_len, channel_len, output_len, dataset, label