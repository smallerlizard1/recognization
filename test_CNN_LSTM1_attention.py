import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import joblib

from dataset_split import loaddata,extract_window_data
'''
1 按文件导入数据
2 按文件划分数据集
3 数据整体归一化
4 滑窗
5 整理组合数据集
'''


class SelfAttention(nn.Module):
    """
    自注意力机制实现
    将输入嵌入分割成多个头(heads)，每个头学习不同的注意力模式
    使用线性变换分别计算查询(Query)、键(Key)和值(Value)
    使用矩阵乘法计算查询和键的点积(注意力分数)
    应用softmax函数得到注意力权重
    使用注意力权重对值进行加权求和
    合并所有头的输出
    通过最后的线性层得到最终输出
    支持mask操作(可用于处理变长序列或防止未来信息泄露)

    参数:
        embed_size (int): 输入嵌入维度
        heads (int): 注意力头的数量
    """
    def __init__(self, embed_size, heads=8):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        # 线性变换得到Q,K,V
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)# in_features  out_features   bias
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 输出线性层
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask=None):
        # x.shape: (batch_size, seq_len, embed_size)
        # 获取批量大小
        batch_size = query.shape[0]
        
        # 获取序列长度
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 将嵌入分割成多个头 (heads)
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = query.reshape(batch_size, query_len, self.heads, self.head_dim)
        
        # 线性变换得到Q,K,V
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # 计算注意力分数
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])#形状为 (q, d) 的query矩阵 与 形状为 (k, d) 的key矩阵 的乘积.得到形状为 (q, k) 的注意力分数矩阵
        
        # 如果有mask，应用mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # 计算注意力权重 (softmax over key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # 应用注意力权重到values上
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        
        # 合并所有头的结果
        out = out.reshape(batch_size, query_len, self.heads * self.head_dim)
        
        # 通过线性层
        out = self.fc_out(out)
        
        return out


# 注意力机制测试代码
def attention_test():
    # 参数设置
    embed_size = 256
    heads = 8
    batch_size = 32
    seq_length = 10
    
    attention = SelfAttention(embed_size, heads)# 创建自注意力层
    
    # 创建随机输入 (batch_size, seq_length, embed_size)
    values = torch.randn((batch_size, seq_length, embed_size))
    keys = torch.randn((batch_size, seq_length, embed_size))
    queries = torch.randn((batch_size, seq_length, embed_size))

    out = attention(values, keys, queries)# 前向传播
    print(f"输入形状: {values.shape}") #32 10 256
    print(f"输出形状: {out.shape}") #32 10 256


class CNNBiLSTM(nn.Module):
    def __init__(self,feature_number, window_size, num_hidden_units, num_classes):
        super(CNNBiLSTM,self).__init__()#父类构造函数
        self.feature_number = feature_number
        self.window_size = window_size
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes

        # 1. 自注意力层
        self.attention = SelfAttention(embed_size=feature_number,heads=4)  # 输入特征维度

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

        #  计算CNN输出后的维度
        cnn_output_dim = 64 * (window_size // 4) * (feature_number // 4)
        
        # 自注意力层
        # self.attention = SelfAttention(embed_size=cnn_output_dim, heads=2)
        # 自注意力后添加LayerNorm

        # self.dropout = nn.Dropout(0.1)

        self.bilstm = nn.LSTM(input_size=cnn_output_dim,#CNN池化层下采样,每次//2  s
                            hidden_size=self.num_hidden_units,
                            num_layers=1,
                            batch_first=True,#输入数据的形状为(batch_size, seq_len, input_size)
                            bidirectional=True)#双向LSTM（前向 + 后向，输出维度会翻倍）
        # self.act2 = nn.Tanh()

        # 层归一化
        self.norm = nn.LayerNorm(feature_number)

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
        # 输入形状: [batch, channels, seq_len, features]
        # print(f'x:{x.shape}')
        # 阶段1: 自注意力处理原始信号
        x = self.attention(x[:,0,:,:], x[:,0,:,:], x[:,0,:,:])  # [batch, seq_len, features]

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

        x = self.flatten(x) # [batch, cnn_output_dim]
        # x = self.dropout(x)

        # 重塑为 (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), -1, 64 * (self.window_size // 4) * (self.feature_number // 4))#permute
        # 重塑为序列形式 [batch, seq_len, features]
        # 这里我们把整个CNN输出视为一个序列，seq_len=1
        # x = x.unsqueeze(1)  # [batch, 1, cnn_output_dim]
        
        # 自注意力处理
        # 注意：这里values, keys, queries都是相同的输入
        # x = self.attention(x, x, x)  # [batch, 1, cnn_output_dim]

        lstm_out,(h_n, c_n) = self.bilstm(x)#双向LSTM输出维度翻倍(batch_size, sequence_length, hidden_size*2)
        #h_n (最终隐藏状态):形状: (num_layers * 2, batch_size, hidden_size)双向LSTM
        #c_n (最终细胞状态):形状同 h_n，但存储的是LSTM的细胞状态
        # print(f'lstm_out:{lstm_out.shape}')#[32 1 128]

        out = lstm_out

        # 残差连接和层归一化
        # out = self.norm(out + attention_out)


        out = out[:, -1, :]# 取LSTM最后一个时间步的隐藏状态，输出至全连接层

        out = self.fc(out)
        # out = self.act4(out)
        # out = self.softmax(out) #如果模型训练，该行注销
        return out


def main():

    offline = True


    stand_raw = loaddata(filepath='./data20250601/bt_msg_52.bin',startpoint=0,endpoint=None,data_png_name='52.png')#静止站立  3000
    stand1_raw = loaddata(filepath='./data20250601/bt_msg_53.bin',startpoint=0,endpoint=None,data_png_name='53.png')#静止站立label=0
    # stand2_raw = loaddata(filepath='./data20250601/bt_msg_40.bin',startpoint=2494,endpoint=3362)#静止站立label=0

    level_raw = loaddata(filepath='./data20250601/bt_msg_54.bin',startpoint=250,endpoint=1700,data_png_name='54.png')#平地,label=1
    level1_raw = loaddata(filepath='./data20250601/bt_msg_55.bin',startpoint=500,endpoint=1700,data_png_name='55.png')#平地,label=1

    upstairs_raw = loaddata(filepath='./data20250601/bt_msg_56.bin',startpoint=250,endpoint=6000,data_png_name='56.png')#上楼梯,label=2
    upstairs1_raw = loaddata(filepath='./data20250601/bt_msg_58.bin',startpoint=250,endpoint=6000,data_png_name='58.png')#上楼梯,label=2

    downstairs_raw = loaddata(filepath='./data20250601/bt_msg_57.bin',startpoint=1000,endpoint=2000,data_png_name='57.png')#下楼梯,label=3
    downstairs1_raw = loaddata(filepath='./data20250601/bt_msg_59.bin',startpoint=1500,endpoint=3500,data_png_name='59.png')#下楼梯,label=3
    # downstairs2_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1087,endpoint=1373,data_png_name='32.png')#下楼梯,label=3
    # downstairs3_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1485,endpoint=1706,data_png_name='33.png')#下楼梯,label=3
    # downstairs4_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1790,endpoint=2022,data_png_name='34.png')#下楼梯,label=3
    # downstairs5_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2154,endpoint=2373,data_png_name='35.png')#下楼梯,label=3
    # downstairs6_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2461,endpoint=2678,data_png_name='36.png')#下楼梯,label=3
    # downstairs7_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2815,endpoint=3052,data_png_name='37.png')#下楼梯,label=3

    dataset_all = np.concatenate((stand_raw,stand1_raw,#stand2_raw,
                                level_raw,level1_raw,
                                upstairs_raw,upstairs1_raw,
                                downstairs_raw,downstairs1_raw,#downstairs2_raw,downstairs3_raw,downstairs4_raw,downstairs5_raw,downstairs6_raw,downstairs7_raw
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
    joblib.dump(scaler, "cnn_lstm1_scaler.save")  # 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    # 保存最大值，最小值，缩放比例和偏移量
    # 保存每个特征均值，标准差，方差，已处理的样本总数

    window_size = 20
    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    # train_data_stand2,train_label_stand2,validate_data_stand2,validate_label_stand2,test_data_stand2,test_label_stand2 = extract_window_data(dataset=stand2_raw,window_size=window_size,label=0)

    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)

    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=2)
    train_data_upstairs1,train_label_upstairs1,validate_data_upstairs1,validate_label_upstairs1,test_data_upstairs1,test_label_upstairs1 = extract_window_data(dataset=upstairs1_raw,window_size=window_size,label=2)

    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=3)
    train_data_downstairs1,train_label_downstairs1,validate_data_downstairs1,validate_label_downstairs1,test_data_downstairs1,test_label_downstairs1 = extract_window_data(dataset=downstairs1_raw,window_size=window_size,label=3)
    # train_data_downstairs2,train_label_downstairs2,validate_data_downstairs2,validate_label_downstairs2,test_data_downstairs2,test_label_downstairs2 = extract_window_data(dataset=downstairs2_raw,window_size=window_size,label=3)
    # train_data_downstairs3,train_label_downstairs3,validate_data_downstairs3,validate_label_downstairs3,test_data_downstairs3,test_label_downstairs3 = extract_window_data(dataset=downstairs3_raw,window_size=window_size,label=3)
    # train_data_downstairs4,train_label_downstairs4,validate_data_downstairs4,validate_label_downstairs4,test_data_downstairs4,test_label_downstairs4 = extract_window_data(dataset=downstairs4_raw,window_size=window_size,label=3)
    # train_data_downstairs5,train_label_downstairs5,validate_data_downstairs5,validate_label_downstairs5,test_data_downstairs5,test_label_downstairs5 = extract_window_data(dataset=downstairs5_raw,window_size=window_size,label=3)
    # train_data_downstairs6,train_label_downstairs6,validate_data_downstairs6,validate_label_downstairs6,test_data_downstairs6,test_label_downstairs6 = extract_window_data(dataset=downstairs6_raw,window_size=window_size,label=3)
    # train_data_downstairs7,train_label_downstairs7,validate_data_downstairs7,validate_label_downstairs7,test_data_downstairs7,test_label_downstairs7 = extract_window_data(dataset=downstairs7_raw,window_size=window_size,label=3)
    # train_data_upramp,validate_data_upramp,test_data_upramp = loaddata()#上坡
    # train_data_downramp,validate_data_downramp,test_data_downramp = loaddata()#下坡

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,#train_data_stand2,
                                     train_data_level,train_data_level1,
                                     train_data_upstairs,train_data_upstairs1,
                                     train_data_downstairs,train_data_downstairs1,#train_data_downstairs2,train_data_downstairs3,train_data_downstairs4,train_data_downstairs5,train_data_downstairs6,train_data_downstairs7
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,#train_label_stand2,
                                     train_label_level,train_label_level1,
                                     train_label_upstairs,train_label_upstairs1,
                                     train_label_downstairs,train_label_downstairs1,#train_label_downstairs2,train_label_downstairs3,train_label_downstairs4,train_label_downstairs5,train_label_downstairs6,train_label_downstairs7
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,#validate_data_stand2,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upstairs,validate_data_upstairs1,
                                     validate_data_downstairs,validate_data_downstairs1,#validate_data_downstairs2,validate_data_downstairs3,validate_data_downstairs4,validate_data_downstairs5,validate_data_downstairs6,validate_data_downstairs7
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,#validate_label_stand2,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upstairs,validate_label_upstairs1,
                                     validate_label_downstairs,validate_label_downstairs1,#validate_label_downstairs2,validate_label_downstairs3,validate_label_downstairs4,validate_label_downstairs5,validate_label_downstairs6,validate_label_downstairs7
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,#test_data_stand2,
                                     test_data_level,test_data_level1,
                                     test_data_upstairs,test_data_upstairs1,
                                     test_data_downstairs,test_data_downstairs1,#test_data_downstairs2,test_data_downstairs3,test_data_downstairs4,test_data_downstairs5,test_data_downstairs6,test_data_downstairs7
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,#test_label_stand2,
                                     test_label_level,test_label_level1,
                                     test_label_upstairs,test_label_upstairs1,
                                     test_label_downstairs,test_label_downstairs1,#test_label_downstairs2,test_label_downstairs3,test_label_downstairs4,test_label_downstairs5,test_label_downstairs6,test_label_downstairs7
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
    num_classes = 4  # 4个类别
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,verbose=True)

    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    # 训练模型
    num_epochs = 100
    # for epoch in range(num_epochs):
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
    #     scheduler.step(validate_loss)

    #     # 保存模型
    #     if validate_loss < best_val_loss:
    #         best_val_loss = validate_loss
    #         no_improve = 0
    #         # 保存模型 # 保存最佳模型而不是最后模型
    #         torch.save(model, "cnn_bilstm1_attention_model_all.pth")#保存完整模型
    #         torch.save(model.state_dict(), "cnn_bilstm1_attention_model_params.pth")#只保存模型参数(state_dict)
    #     else:
    #         no_improve += 1
    #         if no_improve >= patience:
    #             print(f"Early stopping at epoch {epoch}")
    #             break

    #     print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}')
    
    # 测试模型
    model_test = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('cnn_bilstm1_attention_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到cnn_lstm model文件: {e}")
    model_test.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)#重塑形状
            outputs = model_test(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # 打印一些预测和真实标签
    print("Predicted labels:", all_preds[:10])
    print("True labels:", all_labels[:10])

    # 计算评估指标
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 计算准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f'Accuracy: {accuracy:.4f}')

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()
    # attention_test()
