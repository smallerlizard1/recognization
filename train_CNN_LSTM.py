import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import joblib

from readbin import read_bin_file

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
        #self.act1 = nn.Sigmoid()#引入非线性，帮助模型捕捉复杂的特征
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)#(batch, 32, window_size//2, feature_number//2) 128,32,10,2
        # self.drop = nn.Dropout(p=0.01)

        # self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1)#输入输入batch_size*32*20*4 输出输入batch_size*64*20*4
        # self.bn2 = nn.BatchNorm2d(64)#归一化
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)#(batch, 64, window_size//4, feature_number//4) 128 64 5,1

        # self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1)
        # self.bn3 = nn.BatchNorm2d(128)#归一化
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.flatten = nn.Flatten()  # 展平层 (128, 64, 5, 1) 128 * 64 * 5 * 1

        self.dropout = nn.Dropout(0.1)

        self.bilstm = nn.LSTM(input_size=32*(window_size//2)*(feature_number//2),#CNN池化层下采样,每次//2  
                            hidden_size=self.num_hidden_units,
                            num_layers=1,
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
        x = self.pool1(x)
        # print(x.shape)
        # x = self.drop(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.pool3(x)

        x = self.flatten(x)
        x = self.dropout(x)

        # 重塑为 (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), -1, 32 * (self.window_size // 2) * (self.feature_number // 2))#permute

        out,(h_n, c_n) = self.bilstm(x)#双向LSTM输出维度翻倍(batch_size, sequence_length, hidden_size*2)
        #h_n (最终隐藏状态):形状: (num_layers * 2, batch_size, hidden_size)双向LSTM
        #c_n (最终细胞状态):形状同 h_n，但存储的是LSTM的细胞状态
        out = out[:, -1, :]# 取LSTM最后一个时间步的隐藏状态，输出至全连接层

        out = self.fc(out)
        # out = self.act4(out)
        # out = self.softmax(out) #如果模型训练，该行注销
        return out

def loaddata(filepath,startpoint,endpoint,):
    # 导入文件
    data_raw = read_bin_file(filepath=filepath,startpoint=startpoint,endpoint=endpoint)
    dimension = data_raw['sensor_IMU_rpy'].shape #(行 列)
    rows = dimension[0] #行数
    cols = dimension[1] #列数
    print(f"data import shape:{rows},{cols}")  # 添加的打印语句
    dimension = data_raw['sensor_IMU_rpy'].ndim # 维度
    print(dimension) #2

    # 数据集
    dataset = np.zeros((rows, 4))# rows行，列
    dataset[:,0] = data_raw['sensor_IMU_rpy'][:,3]  # 左髋x轴角度    #012 345 678 91011 121314
    dataset[:,1] = data_raw['sensor_IMU_rpy'][:,9]  # 右髋x轴角度
    dataset[:,2] = data_raw['sensor_IMU_gyro'][:,3]  # 左髋x轴角速度
    dataset[:,3] = data_raw['sensor_IMU_gyro'][:,9]  # 右髋x轴角速度

    # 划分数据集
    train_data_rows = int(rows*0.7)
    validate_data_rows = int(rows*0.2)
    test_data_rows = int(rows*0.1)
    train_data = dataset[0:train_data_rows-1,:]
    validate_data = dataset[train_data_rows:train_data_rows+validate_data_rows-1,:]
    test_data  = dataset[train_data_rows+validate_data_rows:train_data_rows+validate_data_rows+test_data_rows-1,:]

    return train_data,validate_data,test_data


def extract_window_data_and_labels(data_tuples, window_size=20, step_size=1):
    '''
    描述：
      从多个数据集中提取窗口数据，并为每个窗口分配标签。
    参数:
    - data_tuples (list of tuples): 每个元组包含一个二维NumPy数组和一个标签字符串。
    - window_size (int): 每个窗口包含的时间步数,默认为20。
    - step_size (int): 窗口滑动的步长,默认为1。
    返回:
    - all_windowed_data (torch.Tensor): 一个包含所有窗口数据的张量。
    - all_window_labels (torch.Tensor): 与窗口数据相对应的标签张量。
    '''
    all_windowed_data = []
    all_window_labels = []
    label_to_index = {}

    # 遍历每个数据元组
    for data, label in data_tuples:
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
            continue

        if label not in label_to_index:
            label_to_index[label] = len(label_to_index)

        datarows, channels = data.shape
        num_windows = (datarows - window_size) // step_size + 1
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window = data[start_idx:end_idx, :]
            all_windowed_data.append(window)
            all_window_labels.append(label_to_index[label])

    # 转换为张量
    all_windowed_data = torch.tensor(np.array(all_windowed_data), dtype=torch.float32)
    all_window_labels = torch.tensor(all_window_labels, dtype=torch.long)

    return all_windowed_data, all_window_labels


def main():
    train_data_level,validate_data_level,test_data_level = loaddata(filepath='bt_msg_37.bin',startpoint=129,endpoint=1047)#平地
    train_data_upstairs,validate_data_upstairs,test_data_upstairs = loaddata(filepath='bt_msg_35.bin',startpoint=233,endpoint=1025)#上楼梯
    train_data_downstairs,validate_data_downstairs,test_data_downstairs = loaddata(filepath='bt_msg_36.bin',startpoint=162,endpoint=1000)#下楼梯
    # train_data_upramp,validate_data_upramp,test_data_upramp = loaddata()#上坡
    # train_data_downramp,validate_data_downramp,test_data_downramp = loaddata()#下坡

    #数据标准化，并标准化参数保存至文件
    #scaler = StandardScaler()  # StandardScaler (Z-score 标准化)
    scaler = MinMaxScaler(feature_range=(-1, 1))  # (归一化到 [-1,1])
    scaler.fit_transform(np.concatenate((train_data_level,train_data_upstairs,train_data_downstairs),axis=0))#计算均值、标准差等参数
    # print("均值:", scaler.mean_)
    # print("标准差:", scaler.scale_)  # 对于 StandardScaler
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler
    joblib.dump(scaler, "cnn_bilstm_scaler.save")  # 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    train_data_norm_level = scaler.transform(train_data_level)  # 应用标准化
    validate_data_norm_level = scaler.transform(validate_data_level)  # 应用标准化
    test_data_norm_level = scaler.transform(test_data_level)  # 应用标准化

    train_data_norm_upstairs = scaler.transform(train_data_upstairs)  # 应用标准化
    validate_data_norm_upstairs = scaler.transform(validate_data_upstairs)  # 应用标准化
    test_data_norm_upstairs = scaler.transform(test_data_upstairs)  # 应用标准化

    train_data_norm_downstairs = scaler.transform(train_data_downstairs)  # 应用标准化
    validate_data_norm_downstairs = scaler.transform(validate_data_downstairs)  # 应用标准化
    test_data_norm_downstairs = scaler.transform(test_data_downstairs)  # 应用标准化
    
    # 创建 data_tuples 列表，每个元素是一个 (数据, 标签) 元组
    train_data = [
        (train_data_norm_level, "level"),
        (train_data_norm_upstairs, "upstairs"),
        (train_data_norm_downstairs, "downstairs"),
        # (train_data_upramp, "upramp"),
        # (train_data_downramp, "downramp")
    ]
    validate_data = [
        (validate_data_norm_level, "level"),
        (validate_data_norm_upstairs, "upstairs"),
        (validate_data_norm_downstairs, "downstairs"),
        # (validate_data_upramp, "upramp"),
        # (validate_data_downramp, "downramp")
    ]
    test_data = [
        (test_data_norm_level, "level"),
        (test_data_norm_upstairs, "upstairs"),
        (test_data_norm_downstairs, "downstairs"),
        # (test_data_upramp, "upramp"),
        # (test_data_downramp, "downramp")
    ]

    # 提取窗口数据和标签
    train_X, train_Y = extract_window_data_and_labels(train_data)
    validate_X, validate_Y = extract_window_data_and_labels(validate_data)
    test_X, test_Y = extract_window_data_and_labels(test_data)
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
    num_classes = 3  # 假设有3个类别
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-3)

    # 训练模型
    num_epochs = 10
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

        # 保存模型

        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}')
    
    # 测试模型
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)#重塑形状
            outputs = model(batch_x)
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

    # 保存模型 # 保存最佳模型而不是最后模型
    torch.save(model, "cnn_bilstm_model_all.pth")#保存完整模型
    torch.save(model.state_dict(), "cnn_bilstm_model_params.pth")#只保存模型参数(state_dict)

    # 保存最佳模型而不是最后模型
    # if validate_loss < best_loss:
    #     torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': validate_loss,
    #     }, 'best_model.pth')


if __name__ == "__main__":
    main()
