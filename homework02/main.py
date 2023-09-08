
if __name__ == '__main__':
    import os
    import random
    import pandas as pd
    import torch
    from tqdm import tqdm


    # 用于从文件中加载特征数据
    def load_feat(path):
        feat = torch.load(path)
        return feat


    # 将特征数据 x 在时间轴上平移 n 个帧的操作
    def shift(x, n):
        if n < 0:
            left = x[0].repeat(-n, 1)
            right = x[:n]

        elif n > 0:
            right = x[-1].repeat(n, 1)
            left = x[n:]
        else:
            return x

        return torch.cat((left, right), dim=0)


    # 将特征数据 x 进行拼接，以增加时间上的上下文信息
    def concat_feat(x, concat_n):
        assert concat_n % 2 == 1  # n must be odd
        if concat_n < 2:
            return x
        seq_len, feature_dim = x.size(0), x.size(1)
        x = x.repeat(1, concat_n)
        x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
        mid = (concat_n // 2)
        for r_idx in range(1, mid + 1):
            x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
            x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

        return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


    # 数据预处理函数
    def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
        class_num = 41  # NOTE: pre-computed, should not need change
        mode = 'train' if (split == 'train' or split == 'val') else 'test'
        label_dict = {}
        if mode != 'test':
            phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

            for line in phone_file:
                line = line.strip('\n').split(' ')
                label_dict[line[0]] = [int(p) for p in line[1:]]

        if split == 'train' or split == 'val':
            # split training and validation data
            usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
            random.seed(train_val_seed)
            random.shuffle(usage_list)
            percent = int(len(usage_list) * train_ratio)
            usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
        elif split == 'test':
            usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
        else:
            raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

        usage_list = [line.strip('\n') for line in usage_list]
        print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
            len(usage_list)))

        max_len = 3000000
        X = torch.empty(max_len, 39 * concat_nframes) #没有初始化的张量
        if mode != 'test':
            y = torch.empty(max_len, dtype=torch.long)

        idx = 0
        for i, fname in tqdm(enumerate(usage_list)):
            feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
            cur_len = len(feat)
            feat = concat_feat(feat, concat_nframes)
            if mode != 'test':
                label = torch.LongTensor(label_dict[fname])

            X[idx: idx + cur_len, :] = feat
            if mode != 'test':
                y[idx: idx + cur_len] = label

            idx += cur_len

        X = X[:idx, :]
        if mode != 'test':
            y = y[:idx]

        print(f'[INFO] {split} set')
        print(X.shape)
        if mode != 'test':
            print(y.shape)
            return X, y
        else:
            return X


    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader


    class LibriDataset(Dataset):
        def __init__(self, x, y=None):
            self.data = x
            if y is not None:
                self.label = torch.LongTensor(y)
            else:
                self.label = None

        def __getitem__(self, idx):
            if self.label is not None:
                return self.data[idx], self.label[idx]
            else:
                return self.data[idx]

        def __len__(self):
            return len(self.data)


    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class BasicBlock(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(BasicBlock, self).__init__()

            self.block = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            x = self.block(x)
            return x


    class Classifier(nn.Module):
        # hidden_layers是隐藏层的数量
        # hidden_dim 表示每个隐藏层中的神经元数量（维度）
        def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
            super(Classifier, self).__init__()
            # fc是一个全神经网络
            self.fc = nn.Sequential(
                BasicBlock(input_dim, hidden_dim),
                # 创建了一个包含多个隐藏层的神经网络模型，每个隐藏层都由一个 BasicBlock 构成
                # 隐藏层数由 hidden_layers 决定，每个隐藏层的输入和输出维度都是 hidden_dim。这种结构通常用于构建深度神经网络。
                *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
                nn.Linear(hidden_dim, output_dim)
            )

        # 定义前向传播方式，fc表示的是各个全联接层进行前向传播
        def forward(self, x):
            x = self.fc(x)
            return x


    # data prarameters
    concat_nframes = 1  # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
    train_ratio = 0.8  # the ratio of data used for training, the rest will be used for validation

    # training parameters
    seed = 0  # random seed
    batch_size = 512  # batch size
    num_epoch = 5  # the number of training epoch
    learning_rate = 0.0001  # learning rate
    model_path = './model.ckpt'  # the path where the checkpoint will be saved

    # model parameters
    input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
    hidden_layers = 1  # the number of hidden layers
    hidden_dim = 256  # the hidden dim

    import gc

    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./feat', phone_path='.',
                                       concat_nframes=concat_nframes, train_ratio=train_ratio)
    val_X, val_y = preprocess_data(split='val', feat_dir='./feat', phone_path='.',
                                   concat_nframes=concat_nframes, train_ratio=train_ratio)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    # remove raw feature to save memory
    del train_X, train_y, val_X, val_y
    gc.collect() #进行垃圾回收
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')
    import numpy as np


    # fix seed
    def same_seeds(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    # fix random seed
    same_seeds(seed)

    # create model, define a loss function, and optimizer
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    #之前是使用MSE，现在换了一种方式，叫做CrossEntropy，交叉熵增
    criterion = nn.CrossEntropyLoss()
    # 这段代码的作用是创建一个AdamW优化器，将其配置为根据给定的学习率来更新模型中的所有参数。在训练过程中，将使用这个优化器来执行反向传播和参数更新，以不断优化模型，使其在训练数据上达到更好的性能
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # get dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train()  # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 在outputs的每一行（每一个样本）上寻找最大值，并且返回这些最大值构成的张量以及最大值对应的索引的张量
            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        # validation
        if len(val_set) > 0:
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)

                    loss = criterion(outputs, labels)

                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (
                            val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                    val_acc / len(val_set), val_loss / len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')
