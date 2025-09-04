import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import to_networkx
# from torch_geometric.loader import DataLoader
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

class GCNBiLSTM(nn.Module):
    def __init__(self,batch_size, feature_number, window_size, num_lstm_hidden_units, gcn_dim_h, num_classes):
        super(GCNBiLSTM,self).__init__()#父类构造函数
        self.batch_size = batch_size
        self.feature_number = feature_number # 所有列 num_nodes * num_features_pernode
        self.window_size = window_size
        self.num_lstm_hidden_units = num_lstm_hidden_units
        self.num_classes = num_classes
        self.gnn_num_node = 5*self.window_size # 时间窗口序列都作为节点
        self.gnn_num_nodefeatures = feature_number//5
        self.gnn_dim_h = gcn_dim_h

        self.conv1 = GCNConv(self.gnn_num_nodefeatures, gcn_dim_h)
        self.conv2 = GCNConv(gcn_dim_h, gcn_dim_h)
        self.conv3 = GCNConv(gcn_dim_h, gcn_dim_h)

        self.bilstm = nn.LSTM(input_size=160,#CNN池化层下采样,每次//2  
                            hidden_size=self.num_lstm_hidden_units,
                            num_layers=2,
                            batch_first=True,#输入数据的形状为(batch_size, seq_len, input_size)
                            bidirectional=True)#双向LSTM（前向 + 后向，输出维度会翻倍）

        self.fc = nn.Linear(in_features=2*self.num_lstm_hidden_units,out_features=self.num_classes)

        self.lin1 = Linear(self.gnn_dim_h*3, self.gnn_dim_h*3)#融合多尺度特征（输入是3层输出的拼接，维度dim_h*3）
        self.lin2 = Linear(self.gnn_dim_h*3, self.num_classes)#输出分类结果（维度为类别数self.num_classes）

        # self.softmax = nn.Softmax(dim=1)
        # 交叉熵损失已经包含Softmax，不需要额外添加
        # 可以移除self.softmax = nn.Softmax(dim=1)
        # 或者修改为LogSoftmax配合NLLLoss  ?????????????

        print("\n===== 模型结构 =====")
        print(f"输入形状: (batch, 1, {window_size}, {feature_number})")
        print(f"Conv1输出: {32}x{window_size//2}x{feature_number//2}")
        print(f"gin输出: ")
        print(f"LSTM输入大小: {32*(window_size//2)*(feature_number//2)}")
        print(f"LSTM隐藏层: {num_lstm_hidden_units} (双向: {2*num_lstm_hidden_units})")
        print(f"输出类别数: {num_classes}\n")

    def forward(self, gcn_batch_x):
        # 重塑为四维张量(batch_size, channels, height, width)
        # x = x.view(x.size(0),1,self.window_size,self.feature_number)#保持原始数据的batch_size(第一维度不变)
        
        # Node embeddings 
        h = self.conv1(gcn_batch_x.x, gcn_batch_x.edge_index)
        h = h.relu()
        h = self.conv2(h, gcn_batch_x.edge_index)
        h = h.relu()
        h = self.conv3(h, gcn_batch_x.edge_index)
        h=h.view(self.batch_size,self.window_size,-1)#32,20,160
        # print(f"{gcn_batch_x} {h.shape}") #torch.Size([3200, 32])  torch.Size([32, 100, 32])
        # Graph-level readout
        # hG = global_mean_pool(h, gcn_batch_x.batch)
        # print(hG.shape) #torch.Size([32,32])
        # Classifier
        # h = F.dropout(hG, p=0.5, training=self.training)
        # h = self.lin(h)
        # return hG, F.log_softmax(h, dim=1)


        # 特征融合
        # combined_features = torch.cat([cnn_out, gin_h], dim=1)
        # # 添加序列维度用于LSTM (batch_size, seq_len=1, features)
        # feature_x = combined_features.unsqueeze(1)
        
        # 重塑为 (batch_size, sequence_length, input_size)
        # feature_x = feature_x.view(feature_x.size(0), self.batch_size, -1)#permute
        out,(h_n, c_n) = self.bilstm(h)#双向LSTM输出维度翻倍(batch_size, sequence_length, hidden_size*2)
        #h_n (最终隐藏状态):形状: (num_layers * 2, batch_size, hidden_size)双向LSTM
        #c_n (最终细胞状态):形状同 h_n，但存储的是LSTM的细胞状态
        out = out[:, -1, :]# 取LSTM最后一个时间步的隐藏状态，输出至全连接层

        # Classifier
        out = self.fc(out)
        # out = self.act4(out)
        # out = self.softmax(out) #如果模型训练，该行注销

        # out = self.lin1(out)#通过lin1线性变换 + ReLU激活
        # out = out.relu()
        # out = F.dropout(out, p=0.5, training=self.training)#Dropout防止过拟合（仅在训练时启用）
        # out = self.lin2(out)#最终线性层lin2输出分类得分
        # # 原始输出h：可用于损失计算（如CrossEntropyLoss自动处理log_softmax）。
        # # log_softmax：概率化输出（对数空间，数值稳定）。
        # return out, F.log_softmax(out, dim=1)
        return out


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备设置
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='GCN_SLTM motion recognization')
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
    #应该对训练集进行归一化
    #数据标准化，并标准化参数保存至文件
    #scaler = StandardScaler()  # StandardScaler (Z-score 标准化)
    scaler = MinMaxScaler(feature_range=(-1, 1))  # (归一化到 [-1,1])
    scaler.fit_transform(dataset_all)#计算均值、标准差等参数
    # print("均值:", scaler.mean_)
    # print("标准差:", scaler.scale_)  # 对于 StandardScaler
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler
    # joblib.dump(scaler, "cnn_lstm1_scaler.save")  # 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    # 保存最大值，最小值，缩放比例和偏移量
    # 保存每个特征均值，标准差，方差，已处理的样本总数

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
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True) #丢弃最后一个，保证所有batch_size相同
    validate_loader = DataLoader(validate_dataset, batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    # imu_data = np.random.rand(batch_size, window_size, 5, 5)  # [样本数, 5个IMU, 6个特征]
    # labels = np.random.randint(0, 6, size=batch_size)  # 分类标签
    edges = [(0, 1),   # 躯干0-左大腿1
                (1, 2),   # 左大腿1-左小腿2
                (0, 3),   # 躯干0-右大腿3
                (3, 4)   # 右大腿3-右小腿4
            ]
    edges1 = []
    for i in range(window_size):
        edges1.append((0+i*5, 1+i*5))  # 躯干0-左大腿1
        edges1.append((1+i*5, 2+i*5))  # 左大腿1-左小腿2
        edges1.append((0+i*5, 3+i*5))   # 躯干0-右大腿3
        edges1.append((3+i*5, 4+i*5))   # 右大腿3-右小腿4
        if i<(window_size-1):
            edges1.append((0+i*5, 0+(i+1)*5))   # 时刻点0躯干0-时刻点1躯干0
            edges1.append((1+i*5, 1+(i+1)*5))   # 时刻点0左大腿1-时刻点1左大腿1
            edges1.append((2+i*5, 2+(i+1)*5))   # 时刻点0左小腿2-时刻点1左小腿2
            edges1.append((3+i*5, 3+(i+1)*5))   # 时刻点0右大腿3-时刻点1右大腿3
            edges1.append((4+i*5, 4+(i+1)*5))   # 时刻点0右小腿4-时刻点1右小腿4
    edges1_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 初始化模型、损失函数和优化器
    feature_number = train_X.shape[2]
    window_size = train_X.shape[1]
    num_lstm_hidden_units = 64  # 假设双向LSTM隐藏单元数量为64
    gnn_dim_h = 32
    num_classes = 6  # 6个类别
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = GCNBiLSTM(batch_size, feature_number, window_size, num_lstm_hidden_units, gnn_dim_h, num_classes).to(device)
    if args.train_newmodel == False:
        try:
            print("load existing gcn_bilstm_model")
            model.load_state_dict(torch.load('gcn_bilstm_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
        except FileNotFoundError as e:
            print(f"未找到gcn_bilstm model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)

    print(f"模型参数量: {count_parameters(model)}")

    writer = SummaryWriter('./logs/gcn_bilstm_train') # 日志保存目录
    #训练完成后，在命令行运行 tensorboard --logdir=logs --port=6006
    # 然后在浏览器访问 http://localhost:6006

    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    # 训练模型
    num_epochs = 100
    # for epoch in range(num_epochs):
    #     model.train()
    #     train_loss = 0
    #     all_preds_train = torch.tensor([],device=device)
    #     all_labels_train = torch.tensor([],device=device)
    #     for iter, (batch_x, batch_y) in enumerate(train_loader): #为迭代对象添加计数
    #         epoch_start_time = time.time()
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #         # batch_x形状: [batch_size, window_size, feature_number]32,20,25
    #         # batch_y形状: [batch_size]
    #         # gnn_imu_data = Data(batch_x,edges1_index,batch_y) #torch.Size([32, 20, 25]) batch_size, window_size, feature_number
    #         # 创建gcn批处理
    #         data_list = []
    #         for i in range(batch_x.size(0)):
    #             # 使用当前样本的所有时间步的特征作为节点特征
    #             # 假设每个时间步是一个节点，所以节点数量=window_size
    #             graph_data = Data(
    #                 x=batch_x[i].view(-1,5).to(device), # 形状: [window_size * num_node, feature_number]  
    #                 edge_index=edges1_index,  # 需要预先定义好的边连接关系
    #                 y=batch_y[i].unsqueeze(0).to(device)  # 保持标签形状一致
    #             )
    #             data_list.append(graph_data)
    #         gcn_batch = Batch.from_data_list(data_list).to(device)
    #         # 准备CNN输入数据 (时序数据)
    #         # 添加通道维度 (CNN期望的形状: [batch_size, channels, height, width]), 将window_size视为高度，feature_number视为宽度
    #         # cnn_input = batch_x.unsqueeze(1)  # 形状变为 [batch_size, 1, window_size, feature_number]
    #         # batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
    #         # print(batch_x.shape) #torch.Size([32, 1, 20, 25])
    #         optimizer.zero_grad()
    #         outputs = model(gcn_batch)
    #         loss = criterion(outputs, batch_y)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         all_preds_train = torch.cat([all_preds_train, predicted])# 直接在 GPU 上拼接张量
    #         all_labels_train = torch.cat([all_labels_train, batch_y])
    #         if iter % 10 == 0:
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
    #             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #             # 创建gcn批处理
    #             data_list = []
    #             for i in range(batch_x.size(0)):
    #                 # 使用当前样本的所有时间步的特征作为节点特征
    #                 # 假设每个时间步是一个节点，所以节点数量=window_size
    #                 graph_data = Data(
    #                     x=batch_x[i].view(-1,5).to(device), # 形状: [window_size * num_node, feature_number]  3200*5
    #                     edge_index=edges1_index,  # 需要预先定义好的边连接关系
    #                     y=batch_y[i].unsqueeze(0).to(device)  # 保持标签形状一致
    #                 )
    #                 data_list.append(graph_data)
    #             gcn_batch = Batch.from_data_list(data_list).to(device)
    #             outputs = model(gcn_batch)
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
    #         torch.save(model, "gcn_bilstm_model_all.pth")#保存完整模型
    #         torch.save(model.state_dict(), "gcn_bilstm_model_params.pth")#只保存模型参数(state_dict)
    #     else:
    #         no_improve += 1
    #         if no_improve >= patience:
    #             print(f"Early stopping at epoch {epoch}")
    #             break
    #     epoch_time = time.time()-epoch_start_time
    #     print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time_perbatch {epoch_time/len(train_loader)*1000:.2f}ms, Train iter {len(train_loader)}')
    
    writer.close()# 关闭TensorBoard writer

    # 测试模型
    model_test = GCNBiLSTM(batch_size, feature_number, window_size, num_lstm_hidden_units, gnn_dim_h, num_classes).to(device)
    try:
        model_test.load_state_dict(torch.load('gcn_bilstm_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到gcn_bilstm model文件: {e}")
    model_test.eval()
    all_preds_test = torch.tensor([],device=device)
    all_labels_test = torch.tensor([],device=device)
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 创建gcn批处理
            data_list = []
            for i in range(batch_x.size(0)):
                # 使用当前样本的所有时间步的特征作为节点特征
                # 假设每个时间步是一个节点，所以节点数量=window_size
                graph_data = Data(
                    x=batch_x[i].view(-1,5).to(device), # 形状: [window_size * num_node, feature_number]  
                    edge_index=edges1_index,  # 需要预先定义好的边连接关系
                    y=batch_y[i].unsqueeze(0).to(device)  # 保持标签形状一致
                )
                data_list.append(graph_data)
            gcn_batch = Batch.from_data_list(data_list).to(device)
            outputs = model_test(gcn_batch)
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


# 程序运行结果：

# 正在读取文件: 115.bin
# 文件大小: 144384 字节
# 文件总帧数: 282
# 读取 282帧数据
# ./data_v2_20250726/115.bin
# 结果已保存至: /home/wdh/wdh_usr/test/ex_nn/cnn_lstm/data_v2_20250726/results_temp/00.png
# 正在读取文件: 116.bin
# 文件大小: 248832 字节
# 文件总帧数: 486
# 读取 486帧数据
# ./data_v2_20250726/116.bin
# 结果已保存至: /home/wdh/wdh_usr/test/ex_nn/cnn_lstm/data_v2_20250726/results_temp/10.png
# 正在读取文件: 127.bin
# 文件大小: 712704 字节
# 文件总帧数: 1392
# 读取 1392帧数据
# ./data_v2_20250726/127.bin
# 结果已保存至: /home/wdh/wdh_usr/test/ex_nn/cnn_lstm/data_v2_20250726/results_temp/11.png
# 正在读取文件: 125.bin
# 文件大小: 568320 字节
# 文件总帧数: 1110
# 读取 1110帧数据
# ./data_v2_20250726/125.bin
# 结果已保存至: /home/wdh/wdh_usr/test/ex_nn/cnn_lstm/data_v2_20250726/results_temp/20.png
# 正在读取文件: 126.bin
# 文件大小: 645120 字节
# 文件总帧数: 1260
# 读取 1260帧数据
# ./data_v2_20250726/126.bin
# 结果已保存至: /home/wdh/wdh_usr/test/ex_nn/cnn_lstm/data_v2_20250726/results_temp/30.png
# 正在读取文件: 120.bin
# 文件大小: 1824768 字节
# 文件总帧数: 3564
# 读取 3564帧数据
# ./data_v2_20250726/120.bin
# 结果已保存至: /home/wdh/wdh_usr/test/ex_nn/cnn_lstm/data_v2_20250726/results_temp/40.png
# 正在读取文件: 121.bin
# 文件大小: 1499136 字节
# 文件总帧数: 2928
# 读取 2928帧数据
# ./data_v2_20250726/121.bin
# 结果已保存至: /home/wdh/wdh_usr/test/ex_nn/cnn_lstm/data_v2_20250726/results_temp/50.png
# [归一化参数]：最小值 [-1.49804688e+00 -3.30029297e+00 -4.21765137e+01 -4.00634766e+02
#  -1.52319421e+04 -8.88671875e-01 -1.83789062e+00 -4.47692871e+01
#  -1.79504395e+02 -8.96617116e+03 -1.45800781e+00 -2.73144531e+00
#  -2.37030029e+01 -3.74145508e+02 -1.19074077e+04 -9.71679688e-01
#  -2.88232422e+00 -4.61920166e+01 -1.77978516e+02 -5.65253838e+03
#  -1.49804688e+00 -3.60937500e+00 -2.76800537e+01 -4.01062012e+02
#  -1.40619143e+04]
# [归一化参数]：最大值: [2.06347656e+00 1.90380859e+00 8.81707764e+01 2.32971191e+02
#  1.48988014e+04 8.86718750e-01 7.16308594e-01 2.07092285e+01
#  1.07604980e+02 4.85706937e+03 2.06347656e+00 2.35107422e+00
#  9.28619385e+01 2.39318848e+02 1.39174526e+04 1.07568359e+00
#  1.04687500e+00 3.06683350e+01 1.22131348e+02 7.10837150e+03
#  1.39892578e+00 2.04541016e+00 8.81707764e+01 2.33703613e+02
#  1.48004338e+04]
# 0数据集划分结果:
# 训练集: 184,(184, 20, 25) 样本
# 验证集: 52,(52, 20, 25) 样本
# 测试集: 26,(26, 20, 25) 样本
# 1数据集划分结果:
# 训练集: 147,(147, 20, 25) 样本
# 验证集: 42,(42, 20, 25) 样本
# 测试集: 21,(21, 20, 25) 样本
# 1数据集划分结果:
# 训练集: 686,(686, 20, 25) 样本
# 验证集: 196,(196, 20, 25) 样本
# 测试集: 98,(98, 20, 25) 样本
# 2数据集划分结果:
# 训练集: 616,(616, 20, 25) 样本
# 验证集: 176,(176, 20, 25) 样本
# 测试集: 88,(88, 20, 25) 样本
# 3数据集划分结果:
# 训练集: 686,(686, 20, 25) 样本
# 验证集: 196,(196, 20, 25) 样本
# 测试集: 98,(98, 20, 25) 样本
# 4数据集划分结果:
# 训练集: 1036,(1036, 20, 25) 样本
# 验证集: 296,(296, 20, 25) 样本
# 测试集: 148,(148, 20, 25) 样本
# 5数据集划分结果:
# 训练集: 1036,(1036, 20, 25) 样本
# 验证集: 296,(296, 20, 25) 样本
# 测试集: 148,(148, 20, 25) 样本

# ===== 数据集统计 =====
# Train X shape: torch.Size([4391, 20, 25]), Train Y shape: torch.Size([4391])
# Validate X shape: torch.Size([1254, 20, 25]), Validate Y shape: torch.Size([1254])
# Test X shape: torch.Size([627, 20, 25]), Test Y shape: torch.Size([627])
# 训练集: 4391个样本 | 验证集: 1254 | 测试集: 627
# 输入形状: torch.Size([4391, 20, 25]) | 标签形状: torch.Size([4391])
# 类别分布: [ 184  833  616  686 1036 1036]

# feature_number:25,window_size:20

# ===== 模型结构 =====
# 输入形状: (batch, 1, 20, 25)
# Conv1输出: 32x10x12
# gin输出: 
# LSTM输入大小: 3840
# LSTM隐藏层: 64 (双向: 128)
# 输出类别数: 6

# 模型参数量: 228012
# Epoch 0: Train Loss 1.3175, Validate Loss 0.4423
# Epoch 1: Train Loss 0.4595, Validate Loss 0.1929
# Epoch 2: Train Loss 0.4281, Validate Loss 0.1411
# Epoch 3: Train Loss 0.3333, Validate Loss 0.1063
# Epoch 4: Train Loss 0.2950, Validate Loss 0.0928
# Epoch 5: Train Loss 0.2409, Validate Loss 0.0556
# Epoch 6: Train Loss 0.2179, Validate Loss 0.0415
# Epoch 7: Train Loss 0.2049, Validate Loss 0.2496
# Epoch 8: Train Loss 0.1495, Validate Loss 0.0450
# Epoch 9: Train Loss 0.1605, Validate Loss 0.0691
# Epoch 10: Train Loss 0.1506, Validate Loss 0.0347
# Epoch 11: Train Loss 0.1129, Validate Loss 0.0661
# Epoch 12: Train Loss 0.0933, Validate Loss 0.1027
# Epoch 13: Train Loss 0.1385, Validate Loss 0.0249
# Epoch 14: Train Loss 0.0800, Validate Loss 0.0495
# Epoch 15: Train Loss 0.0825, Validate Loss 0.0365
# Epoch 16: Train Loss 0.0766, Validate Loss 0.0840
# Epoch 17: Train Loss 0.0691, Validate Loss 0.0264
# Epoch 18: Train Loss 0.0455, Validate Loss 0.0374
# Epoch 19: Train Loss 0.0801, Validate Loss 0.0330
# Epoch 20: Train Loss 0.0732, Validate Loss 0.0366
# Epoch 21: Train Loss 0.1000, Validate Loss 0.0474
# Epoch 22: Train Loss 0.0328, Validate Loss 0.0326
# Early stopping at epoch 23

# ===== 模型结构 =====
# 输入形状: (batch, 1, 20, 25)
# Conv1输出: 32x10x12
# gin输出: 
# LSTM输入大小: 3840
# LSTM隐藏层: 64 (双向: 128)
# 输出类别数: 6

# Predicted labels: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# True labels: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Confusion Matrix:
# [[ 26   0   0   0   0   0]
#  [  0 119   0   0   0   0]
#  [  0   0  88   0   0   0]
#  [  0   0   0  98   0   0]
#  [  0   0   0   0 148   0]
#  [  0   0   0   0   0 129]]
# Precision: 1.0000
# Recall: 1.0000
# F1 Score: 1.0000
# Accuracy: 1.0000