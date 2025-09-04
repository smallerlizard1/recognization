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

class HamiltonianLayer(nn.Module):
    def __init__(self, half_d):
        super(HamiltonianLayer, self).__init__()
        self.half_d = half_d
        # 势能函数V(x)的参数化
        self.V = nn.Sequential(
            nn.Linear(half_d, half_d),
            nn.ReLU(),
            nn.Linear(half_d, half_d)
        )
        self.dt = 0.1  # 时间步长
        
    def forward(self, x, y): # x是q，y是p
        batch_size, seq_len, _ = x.shape
        
        # 展平输入为(batch_size*seq_len, half_d)
        x_flat = x.reshape(-1, self.half_d)
        
        # 计算势能函数的梯度
        x_flat.requires_grad_(True)
        V = self.V(x_flat)
        grad_V = torch.autograd.grad(V.sum(), x_flat, create_graph=True)[0]
        grad_V = grad_V.view(batch_size, seq_len, self.half_d)# 重塑梯度回原始形状
        
        # 哈密顿系统的更新方程
        x_new = x + self.dt * y
        y_new = y - self.dt * grad_V
        
        return x_new, y_new


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
        self.half_d = d_model // 2
        self.input_proj = nn.Linear(input_dim, d_model)# 将输入特征维度（51）投影到模型维度（128）
        self.hamilton_layers = nn.ModuleList([HamiltonianLayer(self.half_d) for _ in range(4)])
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
        # 将输入分为两部分，类比为坐标和动量
        q = x[:, :, :self.half_d]  # 坐标部分
        p = x[:, :, self.half_d:]  # 动量部分
        for layer in self.hamilton_layers:# 依次通过各层哈密顿变换
            q, p = layer(q, p)
        out = torch.cat([q, p], dim=-1)# 合并坐标和动量部分
        out = self.pos_encoder(out)
        out = self.transformer(out)
        out = out.mean(dim=1)  # 全局平均池化 # 沿序列维度平均池化 -> (batch, d_model)
        return self.classifier(out) # -> (batch, num_classes)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备设置
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='hamilton_transformer motion recognization')
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
            print("load existing hamilton_transformer_model")
            model.load_state_dict(torch.load('hamilton_transformer_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
        except FileNotFoundError as e:
            print(f"未找到hamilton_transformer model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)

    print(f"模型参数量: {count_parameters(model)}")
    
    writer = SummaryWriter('./logs/hamilton_transformer_train') # 日志保存目录
    #训练完成后，在命令行运行 tensorboard --logdir=logs --port=6006
    # 然后在浏览器访问 http://localhost:6006

    model = model.to(device)# 模型在正确设备上
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        all_preds_train = torch.tensor([],device=device)
        all_labels_train = torch.tensor([],device=device)
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
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
        for batch_x,batch_y in validate_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 确保数据在正确设备上
            with torch.enable_grad():#启用梯度计算Hamiltonian动力学
                outputs = model(batch_x)
            with torch.no_grad():#禁用梯度计算评估指标
                loss = criterion(outputs, batch_y)
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
            torch.save(model, "hamilton_transformer_model_all.pth")#保存完整模型
            torch.save(model.state_dict(), "hamilton_transformer_model_params.pth")#只保存模型参数(state_dict)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        epoch_time = time.time()-epoch_start_time
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time_perbatch {epoch_time/len(train_loader)*1000:.2f}ms, Train iter {len(train_loader)}')
    
    writer.close()# 关闭TensorBoard writer
    
    # 测试模型
    model_test = StandardTransformer(input_dim=feature_number,num_classes=num_classes).to(device)#加载最优模型
    try:
        model_test.load_state_dict(torch.load('hamilton_transformer_model_params.pth',weights_only=True))#从文件加载 cnn_lstm model
    except FileNotFoundError as e:
        print(f"未找到hamilton_transformer model文件: {e}")
    model_test.eval()
    all_preds_test = torch.tensor([],device=device)#GPU
    all_labels_test = torch.tensor([],device=device)
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
