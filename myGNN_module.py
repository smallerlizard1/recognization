import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 模型定义
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0).unsqueeze(0)  # 固定输出为[1, hidden_dim]
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 数据生成
def load_sample_data():
    num_nodes = 15
    num_classes = 5
    x = torch.randn(10, num_nodes, 3)
    edge_index = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14]], dtype=torch.long)
    y = torch.randint(0, num_classes, (10,))
    data_list = []
    for i in range(x.shape[0]):
        data = Data(x=x[i], edge_index=edge_index, y=y[i].item())  # y为标量
        data_list.append(data)
    return data_list

# 数据增强（对节点特征添加噪声）
def add_noise(data, noise_level=0.01):
    data.x = data.x + torch.randn_like(data.x) * noise_level
    return data

# 训练和测试
def train(model, optimizer, data_list):
    model.train()
    total_loss = 0
    for data in data_list:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, torch.tensor([data.y]))  # 目标形状[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_list)

def test(model, data_list):
    model.eval()
    correct = 0
    for data in data_list:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(data_list)


def visualize_embeddings(model, data_list, title="t-SNE Visualization"):
    model.eval()
    
    # 获取所有图的节点嵌入（取最后一层GCN的输出）
    embeddings = []
    labels = []
    for data in data_list:
        with torch.no_grad():
            # 提取节点嵌入（忽略分类头）
            x = model.conv1(data.x, data.edge_index)
            x = model.conv2(x, data.edge_index)  # shape: [num_nodes, hidden_dim]
            embeddings.append(x.cpu().numpy())
            labels.extend([data.y] * x.shape[0])  # 每个节点继承图的标签
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap='viridis',
        alpha=0.6
    )
    plt.colorbar(scatter, label='Class')
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()


if __name__ == '__main__':
    # 主流程
    data_list = load_sample_data()
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    # 检查数据划分
    print("训练集类别分布:", torch.bincount(torch.tensor([d.y for d in train_data])))
    print("测试集类别分布:", torch.bincount(torch.tensor([d.y for d in test_data])))
    train_data = [add_noise(d) for d in train_data]  # 应用噪声增强
    model = GCN(input_dim=3, hidden_dim=32, output_dim=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)# 优化器添加L2正则化
    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    for epoch in range(50):
        loss = train(model, optimizer, train_data)
        visualize_embeddings(model, train_data, "Train Set Embeddings")
        acc = test(model, test_data)
        visualize_embeddings(model, test_data, "Test Set Embeddings")
        scheduler.step(acc)  # 根据测试准确率调整学习率
        print(f'Epoch {epoch+1:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
