import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import joblib

from dataset_split import loaddata,loaddata_preproc,loaddata_preproc_v2,extract_window_data
from user_utilities import count_parameters
from load_dataset_nn import load_dataset_zzm_train_valid_test,load_dataset_wdh_train_valid_test,load_dataset_lyl_train_valid_test
from load_dataset_nn import load_dataset_szh_train_valid_test,load_dataset_kdl_train_valid_test,load_dataset_ly_train_valid_test

'''
1 按文件导入数据
2 按文件划分数据集
3 数据整体归一化
4 滑窗
5 整理组合数据集
'''

# --- 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)# 初始化位置编码矩阵（max_len行，d_model列）
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)# 生成位置序列（0到max_len-1），转为列向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))# 计算除数项（用于频率调节）
        pe[:, 0::2] = torch.sin(position * div_term)# 偶数位置用sin，奇数位置用cos
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) if batch_first else pe.unsqueeze(1)
        # (1, max_len, d_model)                  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # 注册为不参与训练的缓冲区

    def forward(self, x):
        if self.batch_first:
            return x + self.pe[:,:x.size(1)]# 将位置编码加到输入上（自动广播到batch维度）# x形状: (batch, seq_len, d_model)
        else:
            return x + self.pe[:x.size(0)]# x形状: (seq_len, batch, d_model)


# --- 标准Transformer ---
class StandardTransformer(nn.Module):
    def __init__(self, input_dim=51, num_classes=6, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)# 将输入特征维度（51）投影到模型维度（128）
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)# 定义Transformer编码层（含多头注意力+前馈网络）
        self.transformer = TransformerEncoder(encoder_layers, num_layers)# 堆叠num_layers个编码层
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model//2), 
                                        nn.ReLU(), 
                                        nn.Linear(d_model//2, num_classes))
        self.pos_encoder = PositionalEncoding(d_model, batch_first=True)# 位置编码器

# (32,50,51) →投影→ (32,50,128) →输出→ (32,6)
    def forward(self, x):
        x = self.input_proj(x) * math.sqrt(self.d_model)#(32,50,51) →投影→ (32,50,128)
        # x = x.transpose(0, 1)  # 调整维度：(batch, seq_len, d_model) -> (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化 # 沿序列维度平均池化 -> (batch, d_model)
        return self.classifier(x) # -> (batch, num_classes)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备设置
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='transformer motion recognization')
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

    # stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/234.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    # stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/235.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    # level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/236.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    # level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/237.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    # upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/238.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    # upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/240.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    # downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/239.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
    # downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/241.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
    # upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/242.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    # downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/cpq/243.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    # args.dataset = 'cpq'

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

    stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/65.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
    stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/66.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
    level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/67.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
    level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/68.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
    upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/69.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
    upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/71.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
    downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/70.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
    downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/72.bin',startpoint=500,endpoint=2500,data_png_name='31.png')#下坡,label=3
    upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/73.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
    downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/74.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
    args.dataset = 'yhl'

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
    batch_size = 32
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 初始化模型、损失函数和优化器
    feature_number = train_X.shape[2]
    window_size = train_X.shape[1]
    num_classes = 6  # 6个类别
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = StandardTransformer(input_dim=feature_number,num_classes=num_classes).to(device)
    if args.train_newmodel == False:
        try:
            print("load existing transformer_model")
            model.load_state_dict(torch.load('transformer_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
        except FileNotFoundError as e:
            print(f"未找到transformer model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)

    print(f"模型参数量: {count_parameters(model)}")
    writer = SummaryWriter('./logs/transformer_train') # 日志保存目录
    #训练完成后，在命令行运行 tensorboard --logdir=logs --port=6006
    # 然后在浏览器访问 http://localhost:6006

    model = model.to(device)# 模型在正确设备上
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    # 训练模型
    num_epochs = 100
    # for epoch in range(num_epochs):
    #     epoch_start_time = time.time()
    #     model.train()
    #     train_loss = 0
    #     all_preds_train = torch.tensor([],device=device)
    #     all_labels_train = torch.tensor([],device=device)
    #     for i, (batch_x, batch_y) in enumerate(train_loader): #为迭代对象添加计数
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
    #         optimizer.zero_grad()
    #         outputs = model(batch_x)
    #         loss = criterion(outputs, batch_y)
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
    #             outputs = model(batch_x)
    #             loss = criterion(outputs, batch_y)
    #             validate_loss+=loss.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             all_preds_validate = torch.cat([all_preds_validate, predicted])# 直接在 GPU 上拼接张量
    #             all_labels_validate = torch.cat([all_labels_validate, batch_y])
    #     validate_loss /= len(validate_loader)
    #     validate_accuracy = (all_preds_validate == all_labels_validate).float().mean().item()#GPU操作
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
    #         torch.save(model, "transformer_model_all.pth")#保存完整模型
    #         torch.save(model.state_dict(), "transformer_model_params.pth")#只保存模型参数(state_dict)
    #     else:
    #         no_improve += 1
    #         if no_improve >= patience:
    #             print(f"Early stopping at epoch {epoch}")
    #             break
    #     epoch_time = time.time()-epoch_start_time
    #     print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time_perbatch {epoch_time*1000/len(train_loader):.2f}ms, Train iter {len(train_loader)}')
    
    writer.close()# 关闭TensorBoard writer

    # 测试模型
    model_test = StandardTransformer(input_dim=feature_number,num_classes=num_classes).to(device)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('transformer_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到transformer model文件: {e}")
    model_test.eval()
    all_preds_test = torch.tensor([],device=device)#GPU
    all_labels_test = torch.tensor([],device=device)
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
            outputs = model_test(batch_x)
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

