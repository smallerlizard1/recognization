import numpy as np
import torch
import torch.nn as nn
# from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import joblib
import struct

from uart_python import SerialCommunication

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

def main():

    #打开串口
    serial_comm = SerialCommunication(port_name='/dev/ttyACM0', baud_rate=115200)# 新建串口对象
    ports = serial_comm.list_ports()  # 列出可用串口
    print("可用串口:", ports)
    serial_comm.open_port()

    #加载模型
    feature_number = 4
    window_size = 20
    num_hidden_units = 64  # 假设双向LSTM隐藏单元数量为64
    num_classes = 3  # 假设有3个类别
    print(f"feature_number:{feature_number},window_size:{window_size}")
    model = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes)
    try:
        model.load_state_dict(torch.load('cnn_bilstm_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到cnn_lstm model文件: {e}")
    try:
        loaded_scaler = joblib.load('cnn_lstm_scaler.save') #从文件加载 scaler  # 方法1：使用 joblib
    except FileNotFoundError as e:
        print(f"未找到scaler文件: {e}")
    count = 0
    data_raw_list = []

    try:
        while True:
            for frame in serial_comm.newframes:
                Head_aa = struct.unpack('B', frame[0:1])  # 1 Byte 170
                Head_bb = struct.unpack('B', frame[1:2])  # 1 Byte 187
                Head_d1 = struct.unpack('B', frame[2:3])  # 1 Byte 209
                Head_d2 = struct.unpack('B', frame[3:4])  # 1 Byte 210
                time    = struct.unpack('I', frame[4:12])  # 8 Bytes 65535
                # Head_d3 = struct.unpack('B', frame[8:9])  # 1 Byte  211
                # Head_d4 = struct.unpack('B', frame[9:10])  # 1 Byte 212, 后面被空两个字节（内存优化）
                joint   = struct.unpack('5f', frame[12:32])  # 20 Bytes，uart先发送低字节再发送高字节
                # Head_d5 = struct.unpack('B', frame[1:2])  # 1 Byte
                # Head_d6 = struct.unpack('B', frame[1:2])  # 1 Byte
                # reserve = struct.unpack('8f', frame[2:18])  # 32 Bytes
                Head_cc = struct.unpack('B', frame[32:33])  # 1 Byte 204
                Head_dd = struct.unpack('B', frame[33:34])  # 1 Byte 221
                # print([Head_aa,Head_bb,Head_d1,Head_d2,time,Head_d3,Head_d4,joint,Head_cc,Head_dd])
                if Head_aa[0]==0xaa and Head_bb[0]==0xbb and Head_cc[0]==0xcc and Head_dd[0]==0xdd:
                    data_raw_list.append(joint[1:5])
                    # print(data_raw_list[-1])
                    count+=1
            serial_comm.newframes.clear()

            if len(data_raw_list)>=20:
                data_raw_np = np.array(data_raw_list[:20])
                data_raw_list = data_raw_list[1:] # 丢弃第一个
                #print(f'data_raw_np shape:{data_raw_np.shape}')#行window_size,列feature
                scaled_frames = loaded_scaler.transform(data_raw_np)
                # print(scaled_frames)
                X = torch.tensor(scaled_frames.reshape(1, window_size, feature_number), dtype=torch.float32) #转化为tensor,batch_size=1,window_size=20,feature_number=4
                model.eval()            
                with torch.no_grad():                    
                    outputs = model(X)
                    print(outputs)
                    # _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                # y_pred_inv = loaded_scaler.inverse_transform(inverse_transform_helper)[:, 3]# 逆标准化
                
                # 数据及识别结果保存至文件
                # np.savetxt('results.txt', output_np)#保存为文本格式
                # loaded_txt = np.loadtxt('results.txt')读取

                # if output_np is not None:
                #     serial_comm.send_data(output_np.tobytes())  # 发送识别结果至下位机
    finally:
        serial_comm.close_port()


if __name__ == "__main__":
    main()
