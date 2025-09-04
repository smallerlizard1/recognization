import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import joblib

from dataset_split import loaddata,loaddata_preproc,loaddata_preproc_v2,extract_window_data
from user_utilities import count_parameters
'''
1 按文件导入数据
2 按文件划分数据集
3 数据整体归一化
4 滑窗
5 整理组合数据集
'''

class CNNBiLSTM(nn.Module):
    def __init__(self,feature_number, window_size, num_hidden_units, num_classes):
        super(CNNBiLSTM,self).__init__()#父类构造函数
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
        # self.act2 = nn.Tanh()

        self.fc = nn.Linear(in_features=2*self.num_hidden_units,out_features=self.num_classes)

        # self.act4 = nn.Tanh()

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

    def forward(self, x):
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
        # x = self.dropout(x)

        # 重塑为 (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), -1, 64 * (self.window_size // 4) * (self.feature_number // 4))#permute

        out,(h_n, c_n) = self.bilstm(x)#双向LSTM输出维度翻倍(batch_size, sequence_length, hidden_size*2)
        #h_n (最终隐藏状态):形状: (num_layers * 2, batch_size, hidden_size)双向LSTM
        #c_n (最终细胞状态):形状同 h_n，但存储的是LSTM的细胞状态
        out = out[:, -1, :]# 取LSTM最后一个时间步的隐藏状态，输出至全连接层

        out = self.fc(out)
        # out = self.act4(out)
        # out = self.softmax(out) #如果模型训练，该行注销
        return out


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备设置
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='CNN_SLTM1 motion recognization')
    # parser.add_argument('--train_newmodel', type=bool, required=True, help='flag train new model') #required=True, 参数用户必须提供
    parser.add_argument('--train_newmodel', 
                   action='store_true',  # 用户提供时设为 True，否则为 False
                   help='flag to train a new model')
    # python train_CNN_LSTM1.py                  # args.train_newmodel == False
    # python train_CNN_LSTM1.py --train_newmodel # args.train_newmodel == True
    parser.add_argument('--dataset', type=str, default='wdh', help='dataset')
    args = parser.parse_args()
    
    # stand_raw = loaddata(filepath='./data20250513/bt_msg_38.bin',startpoint=0,endpoint=2000,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata(filepath='./data20250513/bt_msg_40.bin',startpoint=0,endpoint=230,data_png_name='01.png')#静止站立label=0
    # stand2_raw = loaddata(filepath='./data20250513/bt_msg_40.bin',startpoint=2494,endpoint=3362,data_png_name='02.png')#静止站立label=0
    # level_raw = loaddata(filepath='./data20250513/bt_msg_39.bin',startpoint=167,endpoint=1645,data_png_name='10.png')#平地,label=1
    # level1_raw = loaddata(filepath='./data20250513/bt_msg_40.bin',startpoint=230,endpoint=2493,data_png_name='11.png')#平地,label=1
    # upstairs_raw = loaddata(filepath='./data20250513/bt_msg_41.bin',startpoint=244,endpoint=3400,data_png_name='20.png')#上楼梯,label=2
    # downstairs_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=219,endpoint=582,data_png_name='30.png')#下楼梯,label=3
    # downstairs1_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=714,endpoint=987,data_png_name='31.png')#下楼梯,label=3
    # downstairs2_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1087,endpoint=1373,data_png_name='32.png')#下楼梯,label=3
    # downstairs3_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1485,endpoint=1706,data_png_name='33.png')#下楼梯,label=3
    # downstairs4_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1790,endpoint=2022,data_png_name='34.png')#下楼梯,label=3
    # downstairs5_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2154,endpoint=2373,data_png_name='35.png')#下楼梯,label=3
    # downstairs6_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2461,endpoint=2678,data_png_name='36.png')#下楼梯,label=3
    # downstairs7_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2815,endpoint=3052,data_png_name='37.png')#下楼梯,label=3

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
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/87.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/89.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/90.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/92.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/91.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/93.bin',startpoint=500,endpoint=2500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/94.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/new_test/95.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'wdh'

    stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/234.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/235.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/236.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/237.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/238.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/240.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/239.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/241.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/242.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/243.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
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
    # # joblib.dump(scaler, "cnn_lstm1_scaler.save")  # 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    # # 保存最大值，最小值，缩放比例和偏移量
    # # 保存每个特征均值，标准差，方差，已处理的样本总数

    window_size = 50
    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
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
    
    print("\n===== 数据统计 =====")
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

    # 数据加载器 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    feature_number = train_X.shape[2]
    window_size = train_X.shape[1]
    num_hidden_units = 64  # 假设双向LSTM隐藏单元数量为64
    num_classes = 6  # 6个类别
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes).to(device)
    if args.train_newmodel == False:
        try:
            print("load existing cnn_bilstm1_model")
            model.load_state_dict(torch.load('cnn_bilstm1_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
        except FileNotFoundError as e:
            print(f"未找到cnn_lstm model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,verbose=True)

    print(f"模型参数量: {count_parameters(model)}")

    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    # 训练模型
    num_epochs = 100
    # for epoch in range(num_epochs):
    #     epoch_start_time = time.time()
    #     model.train()
    #     train_loss = 0
    #     for batch_x, batch_y in train_loader:
    #         batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
    #         optimizer.zero_grad()
    #         outputs = model(batch_x)
    #         loss = criterion(outputs, batch_y)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #     train_loss /= len(train_loader)

    #     model.eval()
    #     validate_loss = 0
    #     with torch.no_grad():
    #         for batch_x,batch_y in validate_loader:
    #             batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
    #             outputs = model(batch_x)
    #             loss = criterion(outputs, batch_y)
    #             validate_loss+=loss.item()
    #     validate_loss /= len(validate_loader)
    #     # scheduler.step(validate_loss)

    #     # 保存模型
    #     if validate_loss < best_val_loss:
    #         best_val_loss = validate_loss
    #         no_improve = 0
    #         # 保存模型 # 保存最佳模型而不是最后模型
    #         torch.save(model, "cnn_bilstm1_model_all.pth")#保存完整模型
    #         torch.save(model.state_dict(), "cnn_bilstm1_model_params.pth")#只保存模型参数(state_dict)
    #     else:
    #         no_improve += 1
    #         if no_improve >= patience:
    #             print(f"Early stopping at epoch {epoch}")
    #             break
    #     epoch_time = time.time()-epoch_start_time
    #     print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time {epoch_time:.2f}s, Train iter {len(train_loader)}')
    
    # 测试模型
    model_test = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes).to(device)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('cnn_bilstm1_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到cnn_lstm model文件: {e}")
    model_test.eval()
    all_preds_torch = torch.tensor([],device=device)#GPU
    all_labels_torch = torch.tensor([],device=device)
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
            batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)#重塑形状
            outputs = model_test(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_torch = torch.cat([all_preds_torch, predicted])# 直接在 GPU 上拼接张量
            all_labels_torch = torch.cat([all_labels_torch, batch_y])
    
    # 打印一些预测和真实标签
    print("Predicted labels:", all_preds_torch[:10])
    print("True labels:", all_labels_torch[:10])

    # 计算评估指标
    all_labels_np = all_labels_torch.cpu().numpy()
    all_preds_np = all_preds_torch.cpu().numpy()
    conf_matrix = confusion_matrix(all_labels_np, all_preds_np)
    precision = precision_score(all_labels_np, all_preds_np, average='macro')
    recall = recall_score(all_labels_np, all_preds_np, average='macro')
    f1 = f1_score(all_labels_np, all_preds_np, average='macro')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 计算准确率
    accuracy = np.sum(np.array(all_preds_np) == np.array(all_labels_np)) / len(all_labels_np)
    print(f'Accuracy: {accuracy:.4f}')

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()

