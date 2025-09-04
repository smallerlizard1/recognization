import numpy as np
import time,math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.data import Data, Batch, Dataset
# from torch_geometric.nn import GCNConv, GINConv
# from torch_geometric.nn import global_mean_pool, global_add_pool
# from torch_geometric.utils import to_networkx
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
from modules_gcnusr import build_symmetric_adj
from modules_pinoc import HFeat,Pin5LinkWithIMU,HumanAnthro,HFeatConfig,PhysicsLosses
from modules_hgcn import LAMessagePassingDown_NoInteraction,LAMessagePassingUp,LAMessagePassingDown,ActivationUpModule,ActivationDownModule

# 采用自定义GCN模块
# 添加hamiltonian

'''
1 按文件导入数据
2 按文件划分数据集
3 数据整体归一化
4 滑窗
5 整理组合数据集
'''

class GraphConvolution(nn.Module):
    def __init__(self,feature_num,hidden_size):
        super(GraphConvolution,self).__init__()
        self.feature_num = feature_num # 行，对接每个节点的特征
        self.hidden_size = hidden_size # 列，对接线性变换后的特征
        self.w=nn.Parameter(torch.FloatTensor(feature_num,hidden_size))#定义线性层权重
        self.b=nn.Parameter(torch.FloatTensor(hidden_size)) # 偏置
        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv,stdv)#初始化
        self.b.data.uniform_(-stdv,stdv)
        # nn.init.kaiming_normal_(self.w,mode="fan_in",nonlinearity="relu")
        # nn.init.zeros_(self.b)
    def forward(self,x,adj):
        """ x: 特征向量，形状可以是:
                - 单个图: [num_nodes, feature_num]
                - 批量图: [batch_size, num_nodes, feature_num]
            adj: 邻接矩阵，形状可以是:
                - 单个图: [num_nodes, num_nodes]
                - 批量图: [batch_size, num_nodes, num_nodes]"""
        input_ndim = x.dim()
        # 处理批量图数据: [batch_size, num_nodes, feature_num]
        if input_ndim == 3:
            batch_size, num_nodes, _ = x.shape
            # 将批量数据展平为 [batch_size * num_nodes, feature_num]
            x = x.view(-1, self.feature_num)
            # 应用线性变换 # 恢复批量维度: [batch_size, num_nodes, hidden_size]
            x = torch.mm(x, self.w).view(batch_size, num_nodes, self.hidden_size)
            # 批量邻接矩阵: [batch_size, num_nodes, num_nodes]
            output = torch.bmm(adj, x)  # 批量矩阵乘法
            # 添加偏置并恢复形状
            output = output + self.b
            return output
        # 单个图: [num_nodes, feature_num]
        elif input_ndim == 2:
            x = torch.mm(x,self.w)
            # print(f"gcn_x{x.shape}")
            # 使用稀疏矩阵乘法(如果adj是稀疏的),或普通矩阵乘法
            output=torch.spmm(adj,x) if adj.is_sparse else torch.mm(adj,x)
            return output + self.b
        else:
            raise ValueError(f"输入x的维度必须是2或3,但得到的是 {input_ndim}")


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
        return F.linear(input, self.w, self.b)


class GCNBiLSTM(nn.Module):
    def __init__(self,cfg,anth,batch_size, feature_number, window_size, num_lstm_hidden_units, gcn_dim_h, num_classes):
        super(GCNBiLSTM,self).__init__()#父类构造函数
        self.batch_size = batch_size
        self.feature_number = feature_number # 所有列 num_nodes * num_features_pernode
        self.window_size = window_size
        self.num_lstm_hidden_units = num_lstm_hidden_units
        self.num_classes = num_classes
        self.gnn_num_node = 5*self.window_size # 时间窗口序列都作为节点, 5个节点，4条边
        self.gnn_num_features_pernode = feature_number//5
        self.gnn_dim_h = gcn_dim_h
        self.cfg = cfg
        self.anth = anth

        # self.conv1 = GraphConvolution(self.gnn_num_features_pernode, gcn_dim_h)
        # self.conv2 = GraphConvolution(gcn_dim_h, gcn_dim_h)
        # self.conv3 = GraphConvolution(gcn_dim_h, gcn_dim_h)

        self.linear_p = torch.nn.Linear(self.gnn_num_features_pernode, self.gnn_dim_h, bias = False)
        self.linear_q = torch.nn.Linear(self.gnn_num_features_pernode, self.gnn_dim_h, bias = False)
        
        self.hfeat = HFeat(cfg, anth) # H-Feat 特征提取
        
        self.layers = torch.nn.ModuleList()
        for _ in range(2):
            self.layers.append(torch.nn.ModuleList([
                LAMessagePassingUp(self.gnn_dim_h),
                LAMessagePassingDown(self.gnn_dim_h),
                ActivationUpModule(self.gnn_dim_h),
                ActivationDownModule(self.gnn_dim_h)
            ]))

        self.layers.append(torch.nn.ModuleList([
            LAMessagePassingUp(self.gnn_dim_h),
            LAMessagePassingDown(self.gnn_dim_h)
            ]))

        self.bilstm = nn.LSTM(input_size=80,#CNN池化层下采样,每次//2  
                            hidden_size=self.num_lstm_hidden_units,
                            num_layers=2,
                            batch_first=True,#输入数据的形状为(batch_size, seq_len, input_size)
                            bidirectional=True)#双向LSTM（前向 + 后向，输出维度会翻倍）

        self.fc = nn.Linear(in_features=2*self.num_lstm_hidden_units,out_features=self.num_classes)

        self.lin1 = Linear(self.gnn_dim_h*3, self.gnn_dim_h*3)#融合多尺度特征（输入是3层输出的拼接，维度dim_h*3）

        self.lin2 = Linear(self.gnn_dim_h*3, self.num_classes)#输出分类结果（维度为类别数self.num_classes）
        self.bnn_classfier = bnn_layer(input_features=2*self.num_lstm_hidden_units, output_features=self.num_classes)

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

    def forward(self, batch_x,batch_y,edge_index,criterion):
        # batch_x = batch_x.view(batch_x.size(0),-1,self.gnn_num_features_pernode)#保持原始数据的batch_size(第一维度不变)
        # # print(f"x:{batch_x.shape} batch_adj:{batch_adj.shape}") #x:torch.Size([32, 250, 5]) batch_adj:torch.Size([32, 250, 250])
        # h = self.conv1(batch_x, batch_adj)
        # h = h.relu()
        # h = self.conv2(h, batch_adj)
        # h = h.relu()
        # h = self.conv3(h, batch_adj)
        # h = h.view(self.batch_size,self.window_size,-1)#32,50,80

        # 物理特征提取
        # M,Tkin,Vg,H = self.hfeat(batch_x)
        # print(f"hp.shape:{hp.shape}")
        # 特征融合
        batch_x = batch_x.view(batch_x.size(0),-1,self.gnn_num_features_pernode)#保持原始数据的batch_size(第一维度不变)
        # combined_features = torch.cat([cnn_out, gin_h], dim=1)
        # # 添加序列维度用于LSTM (batch_size, window_size=1, features)
        # feature_x = combined_features.unsqueeze(1)
        # hamilton coordinate
        p = self.linear_p(batch_x)
        q = self.linear_q(batch_x)

        for layer in self.layers[:-1]:
            la_up, la_down, actvn_up, actvn_down = layer
            p,q = la_up(p,q,edge_index)
            p,q = la_down(p,q,edge_index)
            p,q = actvn_up(p,q)
            p,q = actvn_down(p,q)
        
        la_up,la_down = self.layers[-1]
        p,q = la_up(p,q,edge_index)
        p,q = la_down(p,q,edge_index)
        
        h = q.view(self.batch_size,self.window_size,-1)# 重塑为 (batch_size, window_size, input_size)
        # print(f"h_shape:{h.shape}") # h_shape:torch.Size([32, 50, 80])
        # feature_x = feature_x.view(feature_x.size(0), self.batch_size, -1)#permute
        out,(h_n, c_n) = self.bilstm(h)#双向LSTM输出维度翻倍(batch_size, window_size, hidden_size*2)
        #h_n (最终隐藏状态):形状: (num_layers * 2, batch_size, hidden_size)双向LSTM
        #c_n (最终细胞状态):形状同 h_n，但存储的是LSTM的细胞状态
        out = out[:, -1, :]# 取LSTM最后一个时间步的隐藏状态，输出至全连接层

        # Classifier
        # out = self.fc(out)
        # out = self.act4(out)
        # out = self.softmax(out) #如果模型训练，该行注销

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
                log_likes[s] = -criterion(outputs[s], batch_y)  # 交叉熵的负值即对数似然
            # 或者如果是回归问题（使用均方误差）
            else:
                log_likes[s] = Normal(outputs[s], self.noise_tol).log_prob(batch_y).sum(dim=-1)

        # 计算均值
        out = outputs.mean(dim=0)  # 平均所有样本的输出
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()  # 平均所有样本和batch的似然

        # 计算ELBO损失（Evidence Lower BOund）
        # 注意：这里使用负ELBO作为损失（因为优化器最小化损失）
        loss = (log_post - log_prior) / self.batch_size - log_like # 计算蒙特卡洛估计值并返回负ELBO作为损失

        # out = self.lin1(out)#通过lin1线性变换 + ReLU激活
        # out = out.relu()
        # out = F.dropout(out, p=0.5, training=self.training)#Dropout防止过拟合（仅在训练时启用）
        # out = self.lin2(out)#最终线性层lin2输出分类得分
        # 原始输出h：可用于损失计算（如CrossEntropyLoss自动处理log_softmax）。
        # log_softmax：概率化输出（对数空间，数值稳定）。
        # return out, F.log_softmax(out, dim=1)
        return out,loss
        # return out


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备设置
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='HGCN_SLTM_bnn motion recognization')
    # parser.add_argument('--train_newmodel', type=bool, required=True, help='flag train new model') #required=True, 参数用户必须提供
    parser.add_argument('--train_newmodel', 
                   action='store_true',  # 用户提供时设为 True，否则为 False
                   help='flag to train a new model')
    # python train_CNN_LSTM1.py                  # args.train_newmodel == False
    # python train_CNN_LSTM1.py --train_newmodel # args.train_newmodel == True
    parser.add_argument('--dataset', type=str, default='wdh', help='dataset')
    args = parser.parse_args()

    batch_size = 32
    window_size = 50
    train_datazzm_norm,train_labelzzm_raw,validate_datazzm_norm,validate_labelzzm_raw,test_datazzm_norm,test_labelzzm_raw = load_dataset_zzm_train_valid_test(batch_size,window_size)
    train_datawdh_norm,train_labelwdh_raw,validate_datawdh_norm,validate_labelwdh_raw,test_datawdh_norm,test_labelwdh_raw = load_dataset_wdh_train_valid_test(batch_size,window_size)
    train_datalyl_norm,train_labellyl_raw,validate_datalyl_norm,validate_labellyl_raw,test_datalyl_norm,test_labellyl_raw = load_dataset_lyl_train_valid_test(batch_size,window_size)
    train_dataszh_norm,train_labelszh_raw,validate_dataszh_norm,validate_labelszh_raw,test_dataszh_norm,test_labelszh_raw = load_dataset_szh_train_valid_test(batch_size,window_size)
    train_datakdl_norm,train_labelkdl_raw,validate_datakdl_norm,validate_labelkdl_raw,test_datakdl_norm,test_labelkdl_raw = load_dataset_kdl_train_valid_test(batch_size,window_size)
    train_dataly_norm, train_labelly_raw, validate_dataly_norm, validate_labelly_raw, test_dataly_norm, test_labelly_raw  = load_dataset_ly_train_valid_test(batch_size,window_size)

    train_data_norm = np.concatenate([train_datazzm_norm,train_datawdh_norm,train_datalyl_norm,train_dataszh_norm,train_datakdl_norm,train_dataly_norm], axis=0)# 合并数据集（沿样本维度拼接）
    train_label_raw = np.concatenate([train_labelzzm_raw, train_labelwdh_raw,train_labellyl_raw,train_labelszh_raw,train_labelkdl_raw,train_labelly_raw], axis=0)
    validate_data_norm = np.concatenate([validate_datazzm_norm, validate_datawdh_norm,validate_datalyl_norm,validate_dataszh_norm,validate_datakdl_norm,validate_dataly_norm], axis=0)
    validate_label_raw = np.concatenate([validate_labelzzm_raw, validate_labelwdh_raw,validate_labellyl_raw,validate_labelszh_raw,validate_labelkdl_raw,validate_labelly_raw], axis=0)
    test_data_norm = np.concatenate([test_datazzm_norm, test_datawdh_norm,test_datalyl_norm,test_dataszh_norm,test_datakdl_norm,test_dataly_norm], axis=0)
    test_label_raw = np.concatenate([test_labelzzm_raw, test_labelwdh_raw,test_labellyl_raw,test_labelszh_raw,test_labelkdl_raw,test_labelly_raw], axis=0)

    # train_data_norm = np.concatenate([train_datazzm_norm], axis=0)# 合并数据集（沿样本维度拼接）
    # train_label_raw = np.concatenate([train_labelzzm_raw], axis=0)
    # validate_data_norm = np.concatenate([validate_datazzm_norm], axis=0)
    # validate_label_raw = np.concatenate([validate_labelzzm_raw], axis=0)
    # test_data_norm = np.concatenate([test_datazzm_norm], axis=0)
    # test_label_raw = np.concatenate([test_labelzzm_raw], axis=0)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) #丢弃最后一个，保证所有batch_size相同
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # imu_data = np.random.rand(batch_size, window_size, 5, 5)  # [样本数, 5个IMU, 6个特征]
    # labels = np.random.randint(0, 6, size=batch_size)  # 分类标签
    # 5个节点，4个边
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
    edges1_index = torch.tensor(edges1, dtype=torch.long).t().contiguous()
    print(f"edge_index:{edges1_index.shape}") #edge_index:torch.Size([2, 445])
    print(edges1_index) # 0,...,249
    # adj = torch.tensor(build_symmetric_adj(edges1_index,5*window_size), dtype=torch.float32, device=device)# 构造临接矩阵

    # 初始化模型、损失函数和优化器
    feature_number = train_X.shape[2]
    # window_size = train_X.shape[1]
    cfg = HFeatConfig()
    anth = HumanAnthro()
    num_lstm_hidden_units = 64  # 假设双向LSTM隐藏单元数量为64
    gnn_dim_h = 16
    num_classes = 6  # 6个类别
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = GCNBiLSTM(cfg,anth,batch_size, feature_number, window_size, num_lstm_hidden_units, gnn_dim_h, num_classes).to(device)
    if args.train_newmodel == False:
        try:
            print("load existing hgcn_bilstm_bnn_model")
            model.load_state_dict(torch.load('hgcn_bilstm_bnn_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
        except FileNotFoundError as e:
            print(f"未找到hgcn_bilstm_bnn model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)

    print(f"模型参数量: {count_parameters(model)}")

    writer = SummaryWriter('./logs/gcn_bilstm1_bnn_train') # 日志保存目录
    #训练完成后，在命令行运行 tensorboard --logdir=logs --port=6006
    # 然后在浏览器访问 http://localhost:6006

    best_val_loss = float('inf')
    patience = 5 # 越小越好
    no_improve = 0
    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        all_preds_train = torch.tensor([],device=device)
        all_labels_train = torch.tensor([],device=device)
        for iter, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
            # batch_x形状: [batch_size, window_size, feature_number]32,20,25
            # batch_y形状: [batch_size]
            # gnn_imu_data = Data(batch_x,edges1_index,batch_y) #torch.Size([32, 20, 25]) batch_size, window_size, feature_number
            # batch_adj = adj.repeat(batch_size, 1, 1)# 直接在第0维重复batch_size次 [batch_size, M, N]  注意adj直接创建在gpu中的向量
            # data_list = []
            # for i in range(batch_x.size(0)):
            #     # 使用当前样本的所有时间步的特征作为节点特征
            #     # 假设每个时间步是一个节点，所以节点数量=window_size
            #     graph_data = Data(
            #         x=batch_x[i].view(-1,5).to(device), # 形状: [window_size * num_node, feature_number]  
            #         edge_index=edges1_index,  # 需要预先定义好的边连接关系
            #         y=batch_y[i].unsqueeze(0).to(device)  # 保持标签形状一致
            #     )
            #     data_list.append(graph_data)
            # gcn_batch = Batch.from_data_list(data_list).to(device)

            # 准备CNN输入数据 (时序数据)
            # 添加通道维度 (CNN期望的形状: [batch_size, channels, height, width]), 将window_size视为高度，feature_number视为宽度
            # cnn_input = batch_x.unsqueeze(1)  # 形状变为 [batch_size, 1, window_size, feature_number]
            # batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
            # print(batch_x.shape) #torch.Size([32, 1, 20, 25])
            optimizer.zero_grad()
            outputs,loss = model(batch_x,batch_y,edges1_index,criterion)
            # outputs = model(batch_x,batch_y,batch_adj,criterion)
            # outputs = model(batch_x,batch_y,edges1_index,criterion)
            # loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds_train = torch.cat([all_preds_train, predicted])# 直接在 GPU 上拼接张量
            all_labels_train = torch.cat([all_labels_train, batch_y])
            if i % 10 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(), epoch*len(train_loader)+i)# 每10个batch记录一次 参数：[指标名称和分类, 要记录的标量值, 全局步数]
        train_loss /= len(train_loader)
        train_accuracy = (all_preds_train == all_labels_train).float().mean().item()#GPU操作

        # ========== 新增TensorBoard记录 ==========
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Time/epoch', time.time()-epoch_start_time, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)


        model.eval()
        validate_loss = 0
        all_preds_validate = torch.tensor([],device=device)
        all_labels_validate = torch.tensor([],device=device)
        with torch.no_grad():
            for batch_x,batch_y in validate_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
                # batch_adj = adj.repeat(batch_size, 1, 1)
                outputs,loss = model(batch_x,batch_y,edges1_index,criterion)
                # outputs = model(batch_x,batch_y,batch_adj,criterion)
                # outputs = model(batch_x,batch_y,edges1_index,criterion)
                # loss = criterion(outputs, batch_y)
                validate_loss+=loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds_validate = torch.cat([all_preds_validate, predicted])# 直接在 GPU 上拼接张量
                all_labels_validate = torch.cat([all_labels_validate, batch_y])
        validate_loss /= len(validate_loader)
        validate_accuracy = (all_preds_validate == all_labels_validate).float().mean().item()#GPU操作
        scheduler.step(validate_loss)

        # ========== 新增TensorBoard记录 ==========
        writer.add_scalar('Loss/validate', validate_loss, epoch)
        writer.add_scalar('Accuracy/validate', validate_accuracy, epoch)
        # 记录学习率
        # for i, param_group in enumerate(optimizer.param_groups):
        #     writer.add_scalar(f'LR/group_{i}', param_group['lr'], epoch)
        # ========================================

        # 保存模型
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            no_improve = 0
            # 保存模型 # 保存最佳模型而不是最后模型
            torch.save(model, "hgcn_bilstm_bnn_model_all.pth")#保存完整模型
            torch.save(model.state_dict(), "hgcn_bilstm_bnn_model_params.pth")#只保存模型参数(state_dict)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        epoch_time = time.time()-epoch_start_time
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time_perbatch {epoch_time/len(train_loader)*1000:.2f}ms, Train iter {len(train_loader)}')
    
    writer.close()# 关闭TensorBoard writer

    # 测试模型
    model_test = GCNBiLSTM(cfg,anth,batch_size, feature_number, window_size, num_lstm_hidden_units, gnn_dim_h, num_classes).to(device)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('hgcn_bilstm_bnn_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到hgcn_bilstm_bnn model文件: {e}")
    model_test.eval()
    all_preds_test = torch.tensor([],device=device)#GPU
    all_labels_test = torch.tensor([],device=device)
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
            # batch_adj = adj.repeat(batch_size, 1, 1)
            outputs,loss = model_test(batch_x,batch_y,edges1_index,criterion)
            # outputs = model_test(batch_x,batch_y,batch_adj,criterion)
            # outputs = model(batch_x,batch_y,edges1_index,criterion)
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
#                     预测为正例          预测为负例
# 实际为正例    TP (True Positive)   FN (False Negative)
# 实际为负例    FP (False Positive)   TN (True Negative)
    precision = precision_score(all_labels_test_np, all_preds_test_np, average='macro') # 在所有预测为正例的样本中，真正是正例的比例  TP / (TP + FP)
    recall = recall_score(all_labels_test_np, all_preds_test_np, average='macro') #在所有实际为正例的样本中，被正确预测为正例的比例  TP / (TP + FN)
    f1 = f1_score(all_labels_test_np, all_preds_test_np, average='macro') # Precision和Recall的调和平均数 F1 = 2 × (Precision × Recall) / (Precision + Recall)
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

