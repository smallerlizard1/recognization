import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from sklearn.preprocessing import normalize

class GraphConvolution(nn.Module):
    def __init__(self,feature_num,hidden_size):
        super(GraphConvolution,self).__init__()
        self.w=Parameter(torch.FloatTensor(feature_num,hidden_size))#定义线性层权重和偏置
        self.b=Parameter(torch.FloatTensor(hidden_size))
        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv,stdv)#初始化
        self.b.data.uniform_(-stdv,stdv)
    def forward(self,x,adj):
        # x:特征向量，adj临接矩阵
        x = torch.mm(x,self.w)
        output=torch.spmm(adj,x)
        return output + self.b

class GCN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(GCN1,self).__init__()
        self.gc1 = GraphConvolution(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.gc2 = GraphConvolution(hidden_size,output_size)
    def forward(self,x,adj):
        x = self.gc1(x,adj)
        x = self.relu(x)
        x = self.drop(x)
        x = self.gc2(x,adj)
        return x

def encode_labels(labels):
    classes = sorted(set(labels))
    label2index = {label:idx for idx,label in enumerate(classes)}
    indices = [label2index[label] for label in labels]
    indices = np.array(indices,dtype=np.int32)
    return indices

def build_symmetric_adj(edges,node_num):
    adj = np.zeros((node_num,node_num),dtype=np.float32)
    for i,j in zip(edges[:,0],edges[:,1]):
        adj[i,j]=1
        adj[j,i]=1
    for i in range(node_num):
        adj[i,i]=1
    return adj

def load_cora_data(data_path):
    print("load_cora_data ...")
    content = np.genfromtxt(data_path + '/cora.content',dtype=np.dtype(str))
    idx = content[:,0].astype(np.int32)# 节点id
    features = content[:,1:-1].astype(np.float32)#节点特征向量
    labels = encode_labels(content[:,-1])#节点标签
    node_num = len(idx)
    print(f"node_num: {node_num}")#2708
    cites = np.genfromtxt(data_path + '/cora.cites',dtype=np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    print(f"idx_map: {idx_map}")
    edges = [(idx_map[i],idx_map[j]) for i,j in cites]
    edges = np.array(edges,dtype=np.int32)
    print(f"edge_num: {len(edges)}")
    adj = build_symmetric_adj(edges,node_num)
    print(f"adj: {adj.shape}")
    features = normalize(features)
    adj = normalize(adj)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = torch.tensor(adj).to_sparse()
    return features,labels,adj


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    features, labels, adj = load_cora_data('./cora/')
    print(f"features:{features.shape}") #features:torch.Size([2708, 1433])
    print(f"labels:{labels.shape}") #labels:torch.Size([2708])
    print(f"adj:{adj.shape}") #adj:torch.Size([2708, 2708])

    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)
    assert len(features) == len(labels)
    assert len(features) == len(adj)

    sample_num = features.shape[0]
    train_num = int(sample_num*0.15)#训练数据 406
    test_num = sample_num - train_num #测试数据 2302
    print(f"train_num:{train_num}")
    print(f"test_num:{test_num}")

    feature_num = features.shape[1]
    hidden_size = 16
    class_num = labels.max().item()+1
    dropout = 0.5
    print(f"feature_num:{feature_num}") #1433
    print(f"hidden_size:{hidden_size}") #16
    print(f"class_num:{class_num}") #7

    model = GCN1(feature_num,hidden_size,class_num,dropout).to(device)
    model.train() # 调整为训练模式
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    n_epoch = 3000
    for epoch in range(1,n_epoch+1):
        optimizer.zero_grad()
        outputs = model(features,adj)
        loss = criterion(outputs[:train_num],labels[:train_num])
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{n_epoch}, Loss: {loss.item():.3f}')
    model.eval()
    outputs = model(features,adj)
    predicted = torch.argmax(outputs[train_num:],dim=1)
    correct = (predicted == labels[train_num:]).sum().item()
    accuracy = 100*correct/test_num
    print(f'Accuracy: {correct}/{test_num}={accuracy:.1f}%')#Accuracy: 1817/2302=78.9%