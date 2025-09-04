import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
# from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import joblib

from dataset_split import loaddata,loaddata_preproc,loaddata_preproc_v2,extract_window_data
from user_utilities import count_parameters
from load_dataset_nn import load_dataset_zzm_train_valid_test,load_dataset_wdh_train_valid_test,load_dataset_lyl_train_valid_test
from load_dataset_nn import load_dataset_szh_train_valid_test,load_dataset_kdl_train_valid_test,load_dataset_ly_train_valid_test

# 20250816
# 相比lstm，多了一个卷积层
# gpu版本

'''
1 按文件导入数据
2 按文件划分数据集
3 数据整体归一化
4 滑窗
5 整理组合数据集
'''

# BNN层，类似于BP网络的Linear层，一层BNN层由weight和bias组成，weight和bias都具有均值和方差
class bnn_layer(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.):
        # prior_var 先验分布,方差(默认为1)
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features)) # 权重均值 torch.Size([output_features, input_features])
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features)) # 权重标准差(使用softplus保证正值)
        self.b_mu =  nn.Parameter(torch.zeros(output_features)) # 偏置均值
        self.b_rho = nn.Parameter(torch.zeros(output_features)) # 偏置标准差
        # self.register_buffer('w', None)# 使用register_buffer存储不需要梯度但需要在设备间传输的变量
        # self.register_buffer('b', None)
        self.w = None # 权重
        self.b = None # 偏置
        self.prior = torch.distributions.Normal(0,prior_var) # 设置先验分布为标准正态分布N(0, prior_var)

    def forward(self, input):
        # sample weights  权重样本
        # 从标准正态分布中采样权重
        w_epsilon = Normal(0,1).sample(self.w_mu.shape).to('cuda') #采样一组(个)随机数，尺寸同zeros(output_features, input_features)
        # 获得服从均值为mu，方差为delta的正态分布的样本
        # 计算权重的标准差（保证始终为正）,使用Softplus函数的变体（通过log(1+exp(ρ))实现）确保标准差为正数,直接使用exp(ρ)可能导致数值不稳定，这种形式更鲁棒
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon # 重参数化技巧（Reparameterization Trick）z=mu+sigma*epsilon
        # sample bias  偏置样本   # 与sample weights同理
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to('cuda')
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon
        # record log prior by evaluating log pdf of prior at sampled weight and bias 计算权重和偏置的对数先验概率
        # 计算log p(w)，用于后续计算loss
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values 计算变分后验分布的对数概率
        # 计算 log p(w|\theta)，用于后续计算loss
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        # 权重确定后，和BP网络层一样使用。 使用采样的权重和偏置执行标准线性变换
        return nn.functional.linear(input, self.w, self.b)

class CNNBiLSTM(nn.Module):
    def __init__(self,feature_number, window_size, num_hidden_units, num_classes):
        super(CNNBiLSTM,self).__init__()#父类构造函数
        self.batch_size = 32
        self.feature_number = feature_number
        self.window_size = window_size
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3),padding=1)#输入batch_size*1*20*4 输出输入batch_size*32*20*4
        self.bn1 = nn.BatchNorm2d(32)#归一化
        self.relu1 = nn.ReLU()
        # self.act1 = nn.Sigmoid()#引入非线性，帮助模型捕捉复杂的特征
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)#(batch, 32, window_size//2, feature_number//2) 128,32,10,2
        # self.drop = nn.Dropout(p=0.01)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1)#输入输入batch_size*32*20*4 输出输入batch_size*64*20*4
        self.bn2 = nn.BatchNorm2d(64)#归一化
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)#(batch, 64, window_size//4, feature_number//4) 128 64 5,1

        # self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1)
        # self.bn3 = nn.BatchNorm2d(128)#归一化
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.flatten = nn.Flatten()  # 展平层 (128, 64, 5, 1) 128 * 64 * 5 * 1
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(input_size=64*(window_size//4)*(feature_number//4),#CNN池化层下采样,每次//2  
                            hidden_size=self.num_hidden_units,
                            num_layers=2,
                            batch_first=True,#输入数据的形状为(batch_size, seq_len, input_size)
                            bidirectional=True)#双向LSTM（前向 + 后向，输出维度会翻倍）

        self.fc = nn.Linear(in_features=2*self.num_hidden_units,out_features=self.num_classes)
        self.bnn_classfier = bnn_layer(input_features=2*self.num_hidden_units, output_features=self.num_classes)

        # self.softmax = nn.Softmax(dim=1)
        # 交叉熵损失已经包含Softmax，不需要额外添加
        # 可以移除self.softmax = nn.Softmax(dim=1)
        # 或者修改为LogSoftmax配合NLLLoss  ?????????????

        print("\n===== 模型结构 =====")
        print(f"输入形状: (batch, 1, {window_size}, {feature_number})")
        print(f"Conv1输出: {32}x{window_size//2}x{feature_number//2}")
        print(f"LSTM输入大小: {32*(window_size//2)*(feature_number//2)}")
        print(f"LSTM隐藏层: {num_hidden_units} (双向: {2*num_hidden_units})")
        print(f"输出类别数: {num_classes}\n")

    # def forward(self, x):
    def forward(self, x,y,criterion):
        self.batch_size = x.size(0)
        # 重塑为四维张量(batch_size, channels, height, width)
        x = x.view(x.size(0),1,self.window_size,self.feature_number)#保持原始数据的batch_size(第一维度不变)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.act1(x)
        x = self.pool1(x)
        # print(x.shape)
        # x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.pool3(x)

        x = self.flatten(x)
        x = self.dropout(x) #测试时需调用model.eval()
        # 重塑为 (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), -1, 64 * (self.window_size // 4) * (self.feature_number // 4))#permute

        out,(h_n, c_n) = self.bilstm(x)#双向LSTM输出维度翻倍(batch_size, sequence_length, hidden_size*2)
        #h_n (最终隐藏状态):形状: (num_layers * 2, batch_size, hidden_size)双向LSTM
        #c_n (最终细胞状态):形状同 h_n，但存储的是LSTM的细胞状态
        out = out[:, -1, :]# 取LSTM最后一个时间步的隐藏状态，输出至全连接层

        # out = self.fc(out)

        # 进行多次前向传播（如10次），每次采样不同的权重，得到一组输出样本
        samples = 10
        # outputs = torch.zeros(samples,device='cuda')
        outputs = torch.zeros(samples, self.batch_size, self.num_classes, device='cuda')
        log_priors = torch.zeros(samples,device='cuda')
        log_posts = torch.zeros(samples,device='cuda')
        log_likes = torch.zeros(samples, self.batch_size,device='cuda')  # 每个样本的似然
        for s in range(samples):
            outputs[s] = self.bnn_classfier(out)
            # 获取当前采样的先验和后验概率
            log_priors[s] = self.bnn_classfier.log_prior # get log prior 计算先验概率
            log_posts[s] = self.bnn_classfier.log_post # get log variational posterior 计算后验概率
            # 计算似然（假设分类问题使用交叉熵）
            if isinstance(criterion, nn.CrossEntropyLoss):
                log_likes[s] = -criterion(outputs[s], y)  # 交叉熵的负值即对数似然
            # 或者如果是回归问题（使用均方误差）
            else:
                log_likes[s] = Normal(outputs[s], self.noise_tol).log_prob(y).sum(dim=-1)

        # 计算均值
        out = outputs.mean(dim=0)  # 平均所有样本的输出
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()  # 平均所有样本和batch的似然

        # 计算ELBO损失（Evidence Lower BOund）
        # 注意：这里使用负ELBO作为损失（因为优化器最小化损失）
        loss = (log_post - log_prior) / self.batch_size - log_like # 计算蒙特卡洛估计值并返回负ELBO作为损失

        # 冻结不需要训练的子网络
        # for param in bnn_layer.parameters():
        #     param.requires_grad = False

        # out_sample_elbo_loss = self.sample_elbo_loss(input, target, 5)
        # out = self.act4(out)
        # out = self.softmax(out) #如果模型训练，该行注销
        return out,loss
    
        # 计算loss 计算证据下界(ELBO)损失函数
    def sample_elbo_loss(self, input, model_output, target, criterion_fun, samples):
        # we calculate the negative elbo, which will be our loss function
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples

        # 蒙特卡罗近似
        for i in range(samples):
            outputs[i] = self(input).reshape(-1) # make predictions 生成预测输出
            log_priors[i] = self.log_prior() # get log prior 计算先验概率
            log_posts[i] = self.log_post() # get log variational posterior 计算后验概率
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum() # calculate the log likelihood 计算似然概率
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        loss = log_post - log_prior - log_like # 计算蒙特卡洛估计值并返回负ELBO作为损失
        return loss

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备设置
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='CNN_LSTM_bnn motion recognization')
    # parser.add_argument('--train_newmodel', type=bool, required=True, help='flag train new model') #required=True, 参数用户必须提供
    parser.add_argument('--train_newmodel', 
                   action='store_true',  # 用户提供时设为 True，否则为 False
                   help='flag to train a new model')
    # python train_CNN_LSTM1.py                  # args.train_newmodel == False
    # python train_CNN_LSTM1.py --train_newmodel # args.train_newmodel == True
    parser.add_argument('--dataset', type=str, default='wdh', help='dataset')
    args = parser.parse_args()

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/209.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/210.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//211.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//212.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//220.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//222.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//221.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//223.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//218.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//219.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'wdh'

    # 不同工况测试数据
    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/209.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/210.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/87.bin',startpoint=500,endpoint=3000,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/89.bin',startpoint=500,endpoint=3000,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/90.bin',startpoint=500,endpoint=2500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/92.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/91.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/93.bin',startpoint=500,endpoint=2500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/94.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/95.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'wdh'

    stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/234.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/235.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//236.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//237.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//238.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//240.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//239.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//241.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//242.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq//243.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    args.dataset = 'cpq'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/75.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/76.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/77.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/78.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/79.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/81.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/80.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/82.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/83.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/86.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'lys'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/65.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/66.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/67.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/68.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/69.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/71.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/70.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/72.bin',startpoint=500,endpoint=2500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/73.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/74.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'yhl'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/224.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/225.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/226.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/227.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/228.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/230.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/229.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/231.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/232.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/233.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'ly'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/244.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/245.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/246.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/247.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/249.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/253.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/251.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/254.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/1.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/jdn/1.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'jdn'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/3.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/4.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/5.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/6.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/7.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/9.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/8.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/10.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/11.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/12.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'lyl'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/13.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/14.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/16.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/17.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/18.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/20.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/19.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/21.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/22.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/23.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'szh'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/24.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/25.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/26.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/27.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/28.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/30.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/29.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/31.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/32.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/mcy/33.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'mcy'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/34.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/35.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/36.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/37.bin',startpoint=500,endpoint=3500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/38.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/40.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/39.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/41.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/42.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/43.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'kdl'

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/45.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/46.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/49.bin',startpoint=2000,endpoint=7000,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/50.bin',startpoint=500,endpoint=5000,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/51.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/53.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/52.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/55.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/56.bin',startpoint=500,endpoint=4000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/59.bin',startpoint=500,endpoint=4000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'zzm'

    print(f"dataset:{args.dataset}")
    dataset_all = np.concatenate((stand_raw,stand1_raw,#stand2_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,#downstairs1_raw,downstairs2_raw,downstairs3_raw,downstairs4_raw,downstairs5_raw,downstairs6_raw,downstairs7_raw
                                     ),axis=0)
    
    # #应该对训练集进行归一化
    # #数据标准化，并标准化参数保存至文件
    # #scaler = StandardScaler()  # StandardScaler (Z-score 标准化)
    # scaler = MinMaxScaler(feature_range=(-1, 1))  # (归一化到 [-1,1])
    # scaler.fit_transform(dataset_all)#计算均值、标准差等参数
    # # print("均值:", scaler.mean_)
    # # print("标准差:", scaler.scale_)  # 对于 StandardScaler
    # print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    # print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler
    # # joblib.dump(scaler, "cnn_bilstm_bnn_scaler.save")  # 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    # # 保存最大值，最小值，缩放比例和偏移量
    # # 保存每个特征均值，标准差，方差，已处理的样本总数

    batch_size = 32
    window_size = 50
    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    # train_data_stand2,train_label_stand2,validate_data_stand2,validate_label_stand2,test_data_stand2,test_label_stand2 = extract_window_data(dataset=stand2_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,#train_data_stand2,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,#train_data_downstairs1,train_data_downstairs2,train_data_downstairs3,train_data_downstairs4,train_data_downstairs5,train_data_downstairs6,train_data_downstairs7
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,#train_label_stand2,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,#train_label_downstairs1,train_label_downstairs2,train_label_downstairs3,train_label_downstairs4,train_label_downstairs5,train_label_downstairs6,train_label_downstairs7
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,#validate_data_stand2,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,#validate_data_downstairs1,validate_data_downstairs2,validate_data_downstairs3,validate_data_downstairs4,validate_data_downstairs5,validate_data_downstairs6,validate_data_downstairs7
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,#validate_label_stand2,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,#validate_label_downstairs1,validate_label_downstairs2,validate_label_downstairs3,validate_label_downstairs4,validate_label_downstairs5,validate_label_downstairs6,validate_label_downstairs7
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,#test_data_stand2,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,#test_data_downstairs1,test_data_downstairs2,test_data_downstairs3,test_data_downstairs4,test_data_downstairs5,test_data_downstairs6,test_data_downstairs7
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,#test_label_stand2,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,#test_label_downstairs1,test_label_downstairs2,test_label_downstairs3,test_label_downstairs4,test_label_downstairs5,test_label_downstairs6,test_label_downstairs7
                                     ),axis=0)

    
    # 应用标准化
    scaler = joblib.load("data_scaler.save")
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    train_X = torch.tensor(np.array(train_data_norm), dtype=torch.float32)
    train_Y = torch.tensor(train_label_raw, dtype=torch.long)
    validate_X = torch.tensor(np.array(validate_data_norm), dtype=torch.float32)
    validate_Y = torch.tensor(validate_label_raw, dtype=torch.long)
    test_X = torch.tensor(np.array(test_data_norm), dtype=torch.float32)
    test_Y = torch.tensor(test_label_raw, dtype=torch.long)
    
    print("\n===== 数据集统计 =====")
    print(f"Train X shape: {train_X.shape}, Train Y shape: {train_Y.shape}")
    print(f"Validate X shape: {validate_X.shape}, Validate Y shape: {validate_Y.shape}")
    print(f"Test X shape: {test_X.shape}, Test Y shape: {test_Y.shape}")
    print(f"训练集: {len(train_X)}个样本 | 验证集: {len(validate_X)} | 测试集: {len(test_X)}")
    print(f"输入形状: {train_X.shape} | 标签形状: {train_Y.shape}")
    print(f"类别分布: {np.bincount(train_Y.numpy())}\n")

    # 创建数据集 TensorDataset 
    train_dataset = TensorDataset(train_X, train_Y)
    validate_dataset = TensorDataset(validate_X, validate_Y)
    test_dataset = TensorDataset(test_X, test_Y)

    # gnn_dataset = IMUGraphDataset(train_dataset, train_Y).shuffle()

    # 数据加载器 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)#丢弃最后一个，保证所有batch_size相同
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 初始化模型、损失函数和优化器
    feature_number = train_X.shape[2]
    window_size = train_X.shape[1]
    num_hidden_units = 64  # 假设双向LSTM隐藏单元数量为64
    num_classes = 6  # 6个类别
    patience = 10
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes).to(device)
    if args.train_newmodel == False:
        try:
            print("load existing cnn_bilstm_bnn_model")
            model.load_state_dict(torch.load('cnn_bilstm_bnn_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
        except FileNotFoundError as e:
            print(f"未找到cnn_bilstm_bnn_model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)#L2正则话，但应设置小一些，避免与batchNorm冲突
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)# 使用学习率调度器

    print(f"模型参数量: {count_parameters(model)}")

    writer = SummaryWriter('./logs/cnn_bilstm_bnn_train') # 日志保存目录
    #训练完成后，在命令行运行 tensorboard --logdir=logs --port=6006
    # 然后在浏览器访问 http://localhost:6006

    best_val_loss = float('inf')
    no_improve = 0
    # 训练模型
    num_epochs = 100
    # for epoch in range(num_epochs):
    #     epoch_start_time = time.time()
    #     model.train()
    #     train_loss = 0
    #     all_preds_train = torch.tensor([],device=device)
    #     all_labels_train = torch.tensor([],device=device)
    #     # for batch_x, batch_y in train_loader:
    #     for i, (batch_x, batch_y) in enumerate(train_loader): #为迭代对象添加计数
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
    #         batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
    #         optimizer.zero_grad()
    #         outputs,loss = model(batch_x,batch_y,criterion)
    #         # outputs = model(batch_x)
    #         # loss = criterion(outputs, batch_y)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         all_preds_train = torch.cat([all_preds_train, predicted])# 直接在 GPU 上拼接张量
    #         all_labels_train = torch.cat([all_labels_train, batch_y])
    #         if i % 10 == 0:
    #             writer.add_scalar('Loss/train_batch', loss.item(), epoch*len(train_loader)+i)# 每10个batch记录一次 参数：[指标名称和分类, 要记录的标量值, 全局步数]
    #     train_loss /= len(train_loader)
    #     train_accuracy = (all_preds_train == all_labels_train).float().mean().item()#GPU操作

    #     # ========== 新增TensorBoard记录 ==========
    #     writer.add_scalar('Loss/train', train_loss, epoch)
    #     writer.add_scalar('Time/epoch', time.time()-epoch_start_time, epoch)
    #     writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    #     model.eval()
    #     validate_loss = 0
    #     all_preds_validate = torch.tensor([],device=device)
    #     all_labels_validate = torch.tensor([],device=device)
    #     with torch.no_grad():
    #         for batch_x,batch_y in validate_loader:
    #             batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
    #             batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
    #             outputs,loss = model(batch_x,batch_y,criterion)
    #             # outputs = model(batch_x)
    #             # loss = criterion(outputs, batch_y)
    #             validate_loss+=loss.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             all_preds_validate = torch.cat([all_preds_validate, predicted])# 直接在 GPU 上拼接张量
    #             all_labels_validate = torch.cat([all_labels_validate, batch_y])
    #     validate_loss /= len(validate_loader)
    #     validate_accuracy = (all_preds_validate == all_labels_validate).float().mean().item()#GPU操作
    #     # validate_accuracy = np.sum(np.array(all_preds_validate) == np.array(all_labels_validate)) / len(all_labels_validate) #验证集准确率
    #     scheduler.step(validate_loss)

    #     # ========== 新增TensorBoard记录 ==========
    #     writer.add_scalar('Loss/validate', validate_loss, epoch)
    #     writer.add_scalar('Accuracy/validate', validate_accuracy, epoch)
    #     # 记录学习率
    #     # for i, param_group in enumerate(optimizer.param_groups):
    #     #     writer.add_scalar(f'LR/group_{i}', param_group['lr'], epoch)
    #     # ========================================

    #     # 保存模型
    #     if validate_loss < best_val_loss:
    #         best_val_loss = validate_loss
    #         no_improve = 0
    #         # 保存模型 # 保存最佳模型而不是最后模型
    #         torch.save(model, "phycs_cnn_bilstm_bnn_model_all.pth")#保存完整模型
    #         torch.save(model.state_dict(), "phycs_cnn_bilstm_bnn_model_params.pth")#只保存模型参数(state_dict)
    #     else:
    #         no_improve += 1
    #         if no_improve >= patience:
    #             print(f"Early stopping at epoch {epoch}")
    #             break
    #     epoch_time = time.time()-epoch_start_time
    #     print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time_perbatch {epoch_time/len(train_loader)*1000:.2f}ms, Train iter {len(train_loader)}')
    
    writer.close()# 关闭TensorBoard writer

    # 测试模型
    model_test = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes).to(device)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('cnn_bilstm_bnn_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到cnn_bilstm_bnn model文件: {e}")
    model_test.eval()
    all_preds_test = torch.tensor([],device=device)#GPU
    all_labels_test = torch.tensor([],device=device)
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
            batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)#重塑形状
            outputs,loss = model_test(batch_x,batch_y,criterion)
            # outputs = model_test(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_test = torch.cat([all_preds_test, predicted])# 直接在 GPU 上拼接张量
            all_labels_test = torch.cat([all_labels_test, batch_y])
    
    # 打印一些预测和真实标签
    print("Predicted labels:", all_preds_test[:10])
    print("True labels:", all_labels_test[:10])

    # 计算评估指标
    all_preds_test_np = all_preds_test.cpu().numpy()
    all_labels_test_np = all_labels_test.cpu().numpy()
    conf_matrix = confusion_matrix(all_labels_test_np, all_preds_test_np)
    precision = precision_score(all_labels_test_np, all_preds_test_np, average='macro')
    recall = recall_score(all_labels_test_np, all_preds_test_np, average='macro')
    f1 = f1_score(all_labels_test_np, all_preds_test_np, average='macro')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 计算准确率
    accuracy = np.sum(np.array(all_preds_test_np) == np.array(all_labels_test_np)) / len(all_labels_test_np)
    print(f'Accuracy: {accuracy:.4f}')

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()


