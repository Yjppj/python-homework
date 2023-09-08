# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 把date_set转化成train_set和valid_set
# valid_ration是比例，此处设置成了0.2，我们使用0.2的数据进行验证，0.8的数据进行训练
def train_valid_split(date_set, valid_ratio, seed):
    valid_set_size = int(valid_ratio * len(date_set))
    train_set_size = len(date_set) - valid_set_size
    train_set, valid_set = random_split(date_set, [train_set_size, valid_set_size], torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

#继承了Dataset类
class COVID19Dataset(Dataset):
    #x,y 是这个类的两个属性，python不同于其他语言的语法，是可以这样声明的
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


# 神经网络模型
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        #layers是模型的层次结构 ，Sequential 是一个容器，可以放置多个激活函数
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16), #这个是一个线性层
            nn.ReLU(),#这个就是课堂上面讲的一种激活函数
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # 输入数据经过一系列线性变换和非线性激活函数后，生成了模型的中间表示（通常是特征表示）
        x = self.layers(x)
        x = x.squeeze(1)
        return x

"""
返回值分别是：
raw_x_train[:,feat_idx]   ： 训练数据集的特征数值，经过特征选择，只包含了选定的特征列
raw_y_valid[:,feat_idx] : 验证数据集的特征数据，也经过了特征选择，只包含了选定的特征列
raw_y_test[:,feat_idx] : 测试数据集的特征数据，同样经过了特征选择，只包含了选定的特征列
y_train 训练数据集的目标变量，即模型要学习的真实输出值
y_train 验证数据集的目标变量，也是模型要预测的真实输出值

"""
def select_feat(train_data, valid_data, test_data, select_all=True):
    #此处牵扯到了一些二维数组的操作
    # y_train, y_valid 分别代表训练数据集和结果数据集的目标变量，是机器要学习的输出值
    # train_data[:,-1] 表示从train_data数组中选择所有行，但是只取值最后一列，通常最后一列，最后一列是存储目标变量（标签）的列
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    # train_data[:, :-1] 返回了训练数据集中的所有特征数据，而不包括目标变量
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    # shape[0] 通常用于获取数组的第一个维度的大小，即样本的数量
    # shape[1] 通常用于获取数组的第二个维度的大小，也就是特征的数量
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


def trainer(train_loader, valid_loader, model, config, device):
    #使用均方误差损失
    criterion = nn.MSELoss(reduction='mean')
    # 此处是视频中降到的optimization中的方式之一：Gradient Descent(梯度下降)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=0.9)  # momentum是动量参数的设置，是一种加速收敛的技巧，它有助于在梯度下降中客服一些局部最小值的问题
    # 用来检测日志
    writer = SummaryWriter()
    if not os.path.isdir('models'):
        os.mkdir("models")
    # n_epochs 用于控制深度学习模型的训练周期数量问题
    # best_loss是用来跟踪训练记录过程中的最佳损失值问题。
    # step 是一个计数器，用来跟踪训练过程中的步数迭代问题
    # early_stop_count:也是一个计数器，用来跟踪连续多个训练周期中损失值没有改变的情况，用来实现早停策略，如果在一定数量周期的训练过程中，模型的性能没有明显的改善，就需要停止训练，防止过拟合
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    #训练周期
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        # 一个库，可以用来在终端中创建进度条，tqdm将迭代train_loader中的批次数据，并显示进度条用来跟踪数据加载的速度； position是进度条在终端中的位置，leave是指当前进度条完成后是否要保留在终端中。
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            #x是输入的特征数据，y是目标数据，通常是标签
            optimizer.zero_grad()  # 把梯度设置为0，以准备新的梯度
            x, y = x.to(device), y.to(device)
            # 通过定义模型的forward方法，您可以指定如何处理输入数据以生成输出。在PyTorch中，当您调用模型实例时（例如，model(
            #     x)），实际上是在调用模型的forward方法
            #将输入的数据x通过神经网络的各个层次进行变换，这个过程包括了输入数据的线性变换、非线性激活函数的应用等等其他操作，是受my_model定义的影响
            pred = model(x)
            # 计算loss
            loss = criterion(pred, y)
            #计算梯度（也就是视频中的斜率）、反向传播
            loss.backward()
            # 更新参数、训练迭代
            #用来执行优化器中的一次参数更新
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())
            train_pbar.set_description(f'Epoch[{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        model.eval() #切换到评估模式
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): #在验证过程中，我们不需要计算梯度
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch[{epoch + 1}/{n_epochs}] : Train loss:{mean_valid_loss:.4f},Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # 保存最好的模型
            print('Saving mode with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > config['early_stop']:
            print('\n模型没有提升，停止训练')
            return


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed': 5201314,
        "select_all": True,
        "valid_ratio": 0.2,
        "n_epochs": 3000, # 表示模型训练的周期
        "batch_size": 256, #批次大小
        "learning_rate": 1e-5,
        "early_stop": 400,
        "save_path": './models/model.ckpt'
    }

    same_seed(config['seed'])
    # NumPy 数组
    train_data, test_data = pd.read_csv('covid.train.csv').values, pd.read_csv('covid.test.csv').values
    # 把训练数据集传递进来进行分割，获取训练的数据，和验证训练的数据
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
    print(f"""train_data size: {train_data.shape}
    valid_data size :{valid_data.shape}
    test_data size:{test_data.shape}
    """)

    # select features ,类比到 如果模型是 y = b + wx ,b和w是参数，x 就是feature，特征选择是一个重要的步骤，因为模型的好坏是会收到特征的影响，如果我们缺失了某些重要的特征，或者是选择某些无用的特征，但会影响最终的下降过
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

    print(f'number of features:{x_train.shape[1]}')
    train_dataset, valid_dataset, test_dataset = (COVID19Dataset(x_train, y_train),
                                                  COVID19Dataset(x_valid, y_valid),
                                                  COVID19Dataset(x_test))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
    #训练代码
    trainer(train_loader, valid_loader, model, config, device)