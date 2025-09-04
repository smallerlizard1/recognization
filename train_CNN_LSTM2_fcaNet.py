import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import einsum
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

class FcaLayer(nn.Module):
    """
    FcaNet的频域通道注意力层
    参考: "FcaNet: Frequency Channel Attention Networks" (ICCV 2021)
    channels: 输入特征图的通道数
    reduction: 通道压缩比率,默认16
    dct_h/dct_w: DCT变换的频域分块大小,图像任务:通常默认7x7,时序信号(如IMU):可尝试 1x16 或 1x32
    DCT权重:
    通过get_dct_filter()预计算一个形状为[channel, dct_h*dct_w]的权重矩阵，并注册为模型的缓冲区（不参与训练）
    """
    def __init__(self, channels, reduction=16, dct_h=20,dct_w=8):
        super(FcaLayer, self).__init__()
        self.dct_h = dct_h  # DCT变换的高度
        self.dct_w = dct_w  # DCT变换的宽度
        # 预计算DCT权重矩阵 (shape: [channel, dct_h*dct_w]) 矩阵包含预计算的离散余弦变换（DCT）基函数(不同频率成分的编码),用于后续频域变换
        # register_buffer(name, tensor) 将 tensor 注册为模块的缓冲区，命名为 'weight'
        # 注册后的缓冲区会自动成为模型的一部分，但不参与梯度计算（类似模型参数但不可训练）
        # 避免每次前向传播重复计算，提升效率
        # 保证不同输入样本使用相同的频域变换基
        self.register_buffer('weight', self.get_dct_filter(channel=channels,h=dct_h,w=dct_w))
        # self.weight # 直接访问
        # self._buffers['weight'] # 通过_buffers字典访问
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),  # 升维
            nn.Sigmoid()  # 输出[0,1]的注意力权重
        )

    def forward(self, x):
        b, c, h, w = x.size()  # 输入形状 [batch, channel, height, width]
        
        # 使用DCT进行频域变换
        # 1. 将空间维度展平 [b, c, h*w] h*w空间展平后的像素数
        x_pooled = x.view(b, c, h * w)
        # 2.频域变换：矩阵乘法实现DCT [b, c, dct_h*dct_w] 通过 einsum 实现矩阵乘法，本质是对每个通道的像素值进行DCT变换
        # 公式：$X_{freq} = X_{pooled} \cdot W_{DCT}$，输出形状 [b, c, dct_h*dct_w] 7*7=49
        # 出现在输入但未出现在输出的字母 m 表示需要求和的维度（这里是空间维度 h*w）
        # print(f'x_pooled:{x_pooled.shape}, dct:{self.weight.shape}')
        x_freq = einsum('bcm, cn->bmn', x_pooled, self.weight)
        # x_freq 特征图在频域的表示（每个通道的DCT系数）
        x_freq = x_freq.view(b, c, -1)
        
        # 通道注意力生成
        # 3. 计算频域特征的均值 [b, c] 对每个通道的频域特征取平均，得到 [b, c] 的频域描述符
        # 4. 生成通道注意力权重 [b, c] 通过MLP生成各通道的重要性权重（0~1之间）
        y = self.fc(x_freq.mean(dim=2)) # 获得频域统计量，用于计算通道权重
        
        # 缩放特征
        # 5. 对输入特征进行通道加权 [b, c, 1, 1] * [b, c, h, w] 将权重广播到原始特征图尺寸，进行通道级乘法
        return x * y.view(b, c, 1, 1)

    @staticmethod
    def get_dct_filter(channel,h,w,):
        '''
        为每个通道每个特征预计算一组一维DCT基函数,用于将空间特征转换到频域
        对每个频域块 (i,j),生成2D余弦基函数:basis(u,v,i,j)=cos(πu(2i+1) / 2h)cos(πv(2j+1) / 2w)  (i,j为空间坐标,u,v为频率索引)
        k=0 时使用全1基函数(对应直流分量)
        输出：形状为 [channel, h*w] 的 DCT 基函数矩阵
        torch.arange(h).unsqueeze(1) 生成列向量 $[0,1,...,h-1]^T$，对应空间坐标 $i$
        torch.arange(w).unsqueeze(0) 生成行向量 $[0,1,...,w-1]$，对应空间坐标 $j$
        (2*i + 1) 和 (2*j + 1) 是 DCT 的相位调整项
        '''
        dct_filter = torch.zeros(channel, h * w)#DCT基函数
        part_size_h = h // 2
        part_size_w = w // 2
        for i in range(part_size_h):
            for j in range(part_size_w):
                for k in range(channel):
                    if k == 0:
                        basis = torch.ones(h, w)  # 第0通道使用全1基函数（直流分量）
                    else:
                        # 生成2D DCT基函数
                        basis = torch.cos(torch.pi * (2 * i + 1) * torch.arange(h).unsqueeze(1) / (2 * h)) * \
                                torch.cos(torch.pi * (2 * j + 1) * torch.arange(w).unsqueeze(0) / (2 * w))
                    basis = basis.flatten()
                    dct_filter[k] += basis
        print(f'dct_filter:{dct_filter.shape}')
        return dct_filter / (h * w)  #归一化：所有基函数除以 h*w 以保持数值稳定
        # torch.arange(h).unsqueeze(1)  # shape [h,1]  通过广播机制实现高效的矩阵化计算  避免显式的嵌套循环
        # torch.arange(w).unsqueeze(0)  # shape [1,w]


def test_fcaLayer():
    fcaLayer = FcaLayer(channels=1,dct_h=20,dct_w=8)
    # 创建随机输入 (batch_size, seq_length, embed_size)
    x = torch.randn((32,20,8))
    # weight = torch.randn((8, 4))
    # output = torch.matmul(x, weight)
    out = fcaLayer(x)# 前向传播
    print(f"输入形状: {x.shape}") #32 20 8
    print(f"输出形状: {out.shape}") #32 20 8


class FcaNet_CNN_LSTM(nn.Module):
    def __init__(self,feature_number, window_size, num_hidden_units=256,num_classes=4):
        super(FcaNet_CNN_LSTM, self).__init__()
        self.feature_number = feature_number
        self.window_size = window_size
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 输入通道数,输出通道数（卷积核数量）
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fca1 = FcaLayer(channels=32,dct_h=window_size//2,dct_w=feature_number//2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fca2 = FcaLayer(channels=64,dct_h=window_size//4,dct_w=feature_number//4)

        #  计算CNN输出后的维度
        cnn_output_dim = 64 * (window_size // 4) * (feature_number // 4)

        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=num_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(num_hidden_units, num_hidden_units//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hidden_units//2, num_classes)
        )

    def forward(self, x):
        # 输入形状: [batch, channels, seq_len, features]
        # 重塑为四维张量(batch_size, channels, height, width)
        x = x.view(x.size(0),1,self.window_size,self.feature_number)#保持原始数据的batch_size(第一维度不变)
        x = self.conv1(x)
        x = self.fca1(x)
        x = self.conv2(x)
        x = self.fca2(x)

        # 重塑为 (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), -1, 64 * (self.window_size // 4) * (self.feature_number // 4))#permute
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_hidden_size)
        last_output = lstm_out[:, -1, :]# 取最后一个时间步的输出
        output = self.fc(last_output)# 分类
        return output


def main():

    stand_raw = loaddata(filepath='./data20250513/bt_msg_38.bin',startpoint=0,endpoint=2000,data_png_name='00.png')#静止站立  3000
    stand1_raw = loaddata(filepath='./data20250513/bt_msg_40.bin',startpoint=0,endpoint=230,data_png_name='01.png')#静止站立label=0
    # stand2_raw = loaddata(filepath='./data20250513/bt_msg_40.bin',startpoint=2494,endpoint=3362)#静止站立label=0

    level_raw = loaddata(filepath='./data20250513/bt_msg_39.bin',startpoint=167,endpoint=1645,data_png_name='10.png')#平地,label=1
    level1_raw = loaddata(filepath='./data20250513/bt_msg_40.bin',startpoint=230,endpoint=2493,data_png_name='11.png')#平地,label=1

    upstairs_raw = loaddata(filepath='./data20250513/bt_msg_41.bin',startpoint=244,endpoint=3400,data_png_name='20.png')#上楼梯,label=2

    downstairs_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=219,endpoint=582,data_png_name='30.png')#下楼梯,label=3
    downstairs1_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=714,endpoint=987,data_png_name='31.png')#下楼梯,label=3
    downstairs2_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1087,endpoint=1373,data_png_name='32.png')#下楼梯,label=3
    downstairs3_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1485,endpoint=1706,data_png_name='33.png')#下楼梯,label=3
    downstairs4_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=1790,endpoint=2022,data_png_name='34.png')#下楼梯,label=3
    downstairs5_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2154,endpoint=2373,data_png_name='35.png')#下楼梯,label=3
    downstairs6_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2461,endpoint=2678,data_png_name='36.png')#下楼梯,label=3
    downstairs7_raw = loaddata(filepath='./data20250513/bt_msg_42.bin',startpoint=2815,endpoint=3052,data_png_name='37.png')#下楼梯,label=3

    dataset_all = np.concatenate((stand_raw,stand1_raw,#stand2_raw,
                                     level_raw,level1_raw,
                                     upstairs_raw,
                                     downstairs_raw,downstairs1_raw,downstairs2_raw,downstairs3_raw,downstairs4_raw,downstairs5_raw,downstairs6_raw,downstairs7_raw
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

    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=3)
    train_data_downstairs1,train_label_downstairs1,validate_data_downstairs1,validate_label_downstairs1,test_data_downstairs1,test_label_downstairs1 = extract_window_data(dataset=downstairs1_raw,window_size=window_size,label=3)
    train_data_downstairs2,train_label_downstairs2,validate_data_downstairs2,validate_label_downstairs2,test_data_downstairs2,test_label_downstairs2 = extract_window_data(dataset=downstairs2_raw,window_size=window_size,label=3)
    train_data_downstairs3,train_label_downstairs3,validate_data_downstairs3,validate_label_downstairs3,test_data_downstairs3,test_label_downstairs3 = extract_window_data(dataset=downstairs3_raw,window_size=window_size,label=3)
    train_data_downstairs4,train_label_downstairs4,validate_data_downstairs4,validate_label_downstairs4,test_data_downstairs4,test_label_downstairs4 = extract_window_data(dataset=downstairs4_raw,window_size=window_size,label=3)
    train_data_downstairs5,train_label_downstairs5,validate_data_downstairs5,validate_label_downstairs5,test_data_downstairs5,test_label_downstairs5 = extract_window_data(dataset=downstairs5_raw,window_size=window_size,label=3)
    train_data_downstairs6,train_label_downstairs6,validate_data_downstairs6,validate_label_downstairs6,test_data_downstairs6,test_label_downstairs6 = extract_window_data(dataset=downstairs6_raw,window_size=window_size,label=3)
    train_data_downstairs7,train_label_downstairs7,validate_data_downstairs7,validate_label_downstairs7,test_data_downstairs7,test_label_downstairs7 = extract_window_data(dataset=downstairs7_raw,window_size=window_size,label=3)
    # train_data_upramp,validate_data_upramp,test_data_upramp = loaddata()#上坡
    # train_data_downramp,validate_data_downramp,test_data_downramp = loaddata()#下坡

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,#train_data_stand2,
                                     train_data_level,train_data_level1,
                                     train_data_upstairs,
                                     train_data_downstairs,train_data_downstairs1,train_data_downstairs2,train_data_downstairs3,train_data_downstairs4,train_data_downstairs5,train_data_downstairs6,train_data_downstairs7
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,#train_label_stand2,
                                     train_label_level,train_label_level1,
                                     train_label_upstairs,
                                     train_label_downstairs,train_label_downstairs1,train_label_downstairs2,train_label_downstairs3,train_label_downstairs4,train_label_downstairs5,train_label_downstairs6,train_label_downstairs7
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,#validate_data_stand2,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,validate_data_downstairs1,validate_data_downstairs2,validate_data_downstairs3,validate_data_downstairs4,validate_data_downstairs5,validate_data_downstairs6,validate_data_downstairs7
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,#validate_label_stand2,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,validate_label_downstairs1,validate_label_downstairs2,validate_label_downstairs3,validate_label_downstairs4,validate_label_downstairs5,validate_label_downstairs6,validate_label_downstairs7
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,#test_data_stand2,
                                     test_data_level,test_data_level1,
                                     test_data_upstairs,
                                     test_data_downstairs,test_data_downstairs1,test_data_downstairs2,test_data_downstairs3,test_data_downstairs4,test_data_downstairs5,test_data_downstairs6,test_data_downstairs7
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,#test_label_stand2,
                                     test_label_level,test_label_level1,
                                     test_label_upstairs,
                                     test_label_downstairs,test_label_downstairs1,test_label_downstairs2,test_label_downstairs3,test_label_downstairs4,test_label_downstairs5,test_label_downstairs6,test_label_downstairs7
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
    model = FcaNet_CNN_LSTM(feature_number,window_size,num_hidden_units,num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,verbose=True)

    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        validate_loss = 0
        with torch.no_grad():
            for batch_x,batch_y in validate_loader:
                batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                validate_loss+=loss.item()
        validate_loss /= len(validate_loader)
        scheduler.step(validate_loss)

        # 保存模型
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            no_improve = 0
            # 保存模型 # 保存最佳模型而不是最后模型
            torch.save(model, "cnn_bilstm1_model_all.pth")#保存完整模型
            torch.save(model.state_dict(), "fca_cnn_bilstm2_model_params.pth")#只保存模型参数(state_dict)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}')
    
    # 测试模型
    model_test = FcaNet_CNN_LSTM(feature_number,window_size,num_hidden_units,num_classes)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('fca_cnn_bilstm2_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
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
    # main()
    test_fcaLayer()
