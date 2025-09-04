import time
import argparse
import numpy as np
import torch
import torch.nn as nn
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
from load_dataset_nn import  load_dataset_yhl_train_valid_test,load_dataset_lys_train_valid_test

# 相比lstm，多了一个卷积层
# cpu版本

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

        out = self.fc(out)
        # out = self.softmax(out) #如果模型训练，该行注销
        return out


def main():

    parser = argparse.ArgumentParser(description='CNN_LSTM1 motion recognization')
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
    train_datayhl_norm,train_labelyhl_raw,validate_datayhl_norm,validate_labelyhl_raw,test_datayhl_norm,test_labelyhl_raw = load_dataset_yhl_train_valid_test(batch_size,window_size)
    train_datalys_norm,train_labellys_raw,validate_datalys_norm,validate_labellys_raw,test_datalys_norm,test_labellys_raw = load_dataset_lys_train_valid_test(batch_size,window_size)

    train_data_norm = np.concatenate([train_datazzm_norm,train_datawdh_norm,train_datalyl_norm,train_dataszh_norm,train_datakdl_norm,train_dataly_norm], axis=0)# 合并数据集（沿样本维度拼接）
    train_label_raw = np.concatenate([train_labelzzm_raw, train_labelwdh_raw,train_labellyl_raw,train_labelszh_raw,train_labelkdl_raw,train_labelly_raw], axis=0)
    validate_data_norm = np.concatenate([validate_datazzm_norm, validate_datawdh_norm,validate_datalyl_norm,validate_dataszh_norm,validate_datakdl_norm,validate_dataly_norm], axis=0)
    validate_label_raw = np.concatenate([validate_labelzzm_raw, validate_labelwdh_raw,validate_labellyl_raw,validate_labelszh_raw,validate_labelkdl_raw,validate_labelly_raw], axis=0)
    test_data_norm = np.concatenate([test_datazzm_norm, test_datawdh_norm,test_datalyl_norm,test_dataszh_norm,test_datakdl_norm,test_dataly_norm], axis=0)
    test_label_raw = np.concatenate([test_labelzzm_raw, test_labelwdh_raw,test_labellyl_raw,test_labelszh_raw,test_labelkdl_raw,test_labelly_raw], axis=0)

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
    model = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes)
    if args.train_newmodel == False:
        try:
            print("load existing cnn_bilstm1_model")
            model.load_state_dict(torch.load('cnn_bilstm1_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
        except FileNotFoundError as e:
            print(f"未找到cnn_bilstm1 model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)#L2正则话，但应设置小一些，避免与batchNorm冲突
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)# 使用学习率调度器

    print(f"模型参数量: {count_parameters(model)}")

    writer = SummaryWriter('./logs/cnn_bilstm1_train') # 日志保存目录
    #训练完成后，在命令行运行 tensorboard --logdir=logs --port=6006
    # 然后在浏览器访问 http://localhost:6006

    best_val_loss = float('inf')
    no_improve = 0
    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        all_preds_train = []
        all_labels_train = []
        # for batch_x, batch_y in train_loader:
        for i, (batch_x, batch_y) in enumerate(train_loader): #为迭代对象添加计数
            batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds_train.extend(predicted.cpu().numpy())
            all_labels_train.extend(batch_y.cpu().numpy())
            if i % 10 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(), epoch*len(train_loader)+i)# 每10个batch记录一次 参数：[指标名称和分类, 要记录的标量值, 全局步数]
        train_loss /= len(train_loader)
        train_accuracy = np.sum(np.array(all_preds_train) == np.array(all_labels_train)) / len(all_labels_train) #训练集准确率

        # ========== 新增TensorBoard记录 ==========
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Time/epoch', time.time()-epoch_start_time, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        model.eval()
        validate_loss = 0
        all_preds_validate = []
        all_labels_validate = []
        with torch.no_grad():
            for batch_x,batch_y in validate_loader:
                batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)  # 重塑为正确的形状
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                validate_loss+=loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds_validate.extend(predicted.cpu().numpy())
                all_labels_validate.extend(batch_y.cpu().numpy())
        validate_loss /= len(validate_loader)
        validate_accuracy = np.sum(np.array(all_preds_validate) == np.array(all_labels_validate)) / len(all_labels_validate) #验证集准确率
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
            torch.save(model, "cnn_bilstm1_model_all.pth")#保存完整模型
            torch.save(model.state_dict(), "cnn_bilstm1_model_params.pth")#只保存模型参数(state_dict)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        epoch_time = time.time()-epoch_start_time
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time_perbatch {epoch_time/len(train_loader)*1000:.2f}ms, Train iter {len(train_loader)}')
    
    writer.close()# 关闭TensorBoard writer

    # 测试模型
    model_test = CNNBiLSTM(feature_number,window_size,num_hidden_units,num_classes)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('cnn_bilstm1_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到cnn_lstm model文件: {e}")
    model_test.eval()
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x = batch_x.view(batch_x.size(0), 1, window_size, feature_number)#重塑形状
            outputs = model_test(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_test.extend(predicted.cpu().numpy())
            all_labels_test.extend(batch_y.cpu().numpy())
    
    # 打印一些预测和真实标签
    print("Predicted labels:", all_preds_test[:10])
    print("True labels:", all_labels_test[:10])

    # 计算评估指标
    conf_matrix = confusion_matrix(all_labels_test, all_preds_test)
    precision = precision_score(all_labels_test, all_preds_test, average='macro')
    recall = recall_score(all_labels_test, all_preds_test, average='macro')
    f1 = f1_score(all_labels_test, all_preds_test, average='macro')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 计算准确率
    accuracy = np.sum(np.array(all_preds_test) == np.array(all_labels_test)) / len(all_labels_test)
    print(f'Accuracy: {accuracy:.4f}')

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()

