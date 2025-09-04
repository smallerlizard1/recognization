import numpy as np
import math
import torch
from torch.nn import Linear, Parameter, Module, init
import torch.nn.functional as F 
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


def build_symmetric_adj(edges,node_num):
    adj = np.zeros((node_num,node_num),dtype=np.float32)
    for i,j in zip(edges[:,0],edges[:,1]):
        adj[i,j]=1
        adj[j,i]=1
    for i in range(node_num):
        adj[i,i]=1
    return adj

# 强制权重矩阵是对称的：W_new = W + W^T
# y = (W + W^T)x + b
# 对称矩阵征值都是实数
# 特征向量相互正交
# 数值稳定性
# 能量基模型（Energy-based models）
# 对称系统建模（如物理系统、图拉普拉斯矩阵）
# 特殊架构的神经网络（如哈密顿网络）
# 需要保证特定数学性质的模型
class SymmetricLinear(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ##### Change: Assert that in_features == out_features ##### 对称矩阵必须是方阵
        assert in_features==out_features, f"Expects in_features = out_features, got {in_features} and {out_features}"
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # 使用 Kaiming Uniform 初始化权重（适合ReLU类激活函数）
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        ##### Change: Wnew = W + W^T #####
        return F.linear(input, self.weight + self.weight.t(), self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class SymplecticMessagePassing(torch.nn.Module):
    def __init__(self,feature_num,hidden_size):
        super(SymplecticMessagePassing,self).__init__()
        self.feature_num = feature_num # 行，对接每个节点的特征
        self.hidden_size = hidden_size # 列，对接线性变换后的特征
        self.w=torch.nn.Parameter(torch.FloatTensor(feature_num,hidden_size))#定义线性层权重
        self.b=torch.nn.Parameter(torch.FloatTensor(hidden_size)) # 偏置
        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv,stdv)#初始化
        self.b.data.uniform_(-stdv,stdv)
        # nn.init.kaiming_normal_(self.w,mode="fan_in",nonlinearity="relu")
        # nn.init.zeros_(self.b)
    def forward(self,x,edge_index):
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
            adj = torch.tensor(build_symmetric_adj(edge_index,num_nodes), dtype=torch.float32, device='cuda')# 构造临接矩阵
            batch_adj = adj.repeat(batch_size, 1, 1)# 直接在第0维重复batch_size次 [batch_size, M, N]  注意adj直接创建在gpu中的向量
            # 批量邻接矩阵: [batch_size, num_nodes, num_nodes]
            output = torch.bmm(batch_adj, x)  # 批量矩阵乘法
            # 添加偏置并恢复形状
            output = output + self.b
            return output
        # 单个图: [num_nodes, feature_num]
        elif input_ndim == 2:
            x = torch.mm(x,self.w)
            # print(f"gcn_x{x.shape}")
            # 使用稀疏矩阵乘法（如果adj是稀疏的）或普通矩阵乘法
            output=torch.spmm(adj,x) if adj.is_sparse else torch.mm(adj,x)
            return output + self.b
        else:
            raise ValueError(f"输入x的维度必须是2或3,但得到的是 {input_ndim}")



# 结合了对称线性变换和图消息传递，实现了具有特殊数学性质（辛性质）的图神经网络层
# 这个 SymplecticMessagePassing 层是一个专门为保持物理约束设计的图神经网络层，它结合了对称线性变换和消息传递，特别适合需要保持辛性质的应用场景。
# 核心价值：在保持图神经网络表达能力的同时，引入了物理意义的数学约束，使得模型更适合物理系统的建模和学习。
# 输入输出维度: 由于使用 SymmetricLinear，in_channels 必须等于 out_channels
# 计算复杂度: 对称线性变换增加了一些计算开销
# 理论保证: 辛性质在某些应用中提供更好的理论保证
# 特性	    标准GCN	   SymplecticMessagePassing
# 线性变换	普通线性层	        对称线性层
# 数学性质	无特殊约束	        保持辛结构
# 参数数量	较多	         较少（对称约束）
# 适用场景	通用图学习	      物理约束的系统
# 这种层特别适用于：
# 物理系统建模：辛性质对应物理守恒律
# 分子图神经网络：保持物理约束
# 动力系统学习：学习哈密顿或拉格朗日动力学
# 需要保持特定数学性质的图任务
# 消息传递公式：
#     对节点i的更新：  h(l+1)(i) = SymmetricLinear h(l)(i) + sum(j in N(i))( 1/sqrt(di*dj)*h(l)(j) ) + b
# 其中，SymmetricLinear是对称线性变换
# 1/sqrt(di*dj)是对称归一化系数， GCN风格的对称归一化，保持数值稳定，避免度数大的节点主导学习过程
# b是偏置项
# 辛性质 (Symplectic)
# 通过使用 SymmetricLinear 实现：
# 对称权重矩阵：W=W+WT
# 这在物理系统中对应辛变换，保持系统的某些不变性
# class SymplecticMessagePassing(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr = 'add') # 使用加法聚合
#         self.lin = SymmetricLinear(in_channels, out_channels, bias = False)
#         self.bias= Parameter(torch.empty(out_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         self.bias.data.zero_()

#     def forward(self, x, edge_index):
#         # 计算流程：
#         # 1 节点特征变换: X′=(W+WT)X
#         # 2 度计算: 计算每个节点的度并归一化
#         # 3 消息传递: 对每条边计算归一化消息
#         # 4 聚合: 对每个节点聚合邻居消息
#         # 5 偏置添加: 加上可学习的偏置

#         #x: [N, in_channels]
#         #edge_index: [2,E]

#         #edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

#         x = self.lin(x) # 1. 对称线性变换
#         # print(x.shape)#torch.Size([32, 32])

#         # 2. 计算归一化系数
#         row, col = edge_index ## col 包含所有目标节点的索引
#         # print(f"row {row.shape}, col {col.shape}")#128 128
#         deg = degree(col, x.size(0), dtype=x.dtype)# 计算每个节点的度数，每个节点与旁边节点连接个数,  # x.size(0)就是图中节点的总数
#         # print(deg)
#         # deg = torch.clamp(deg, min=0)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # 3. 消息传递
#         # 自动处理消息的生成、聚合和更新
#         # 使用 aggr='add' 进行邻居信息求和
#         out = self.propagate(edge_index, x=x, norm=norm)#edge_index:边索引张量，形状为[2,num_edges].  x消息[num_edges,out_channels]

#         # 4. 添加偏置
#         out = out + self.bias

#         return out

#     def message(self, x_j, norm):
#         #x_j: [E, out_channels] 源节点特征
#         #norm: 归一化系数 [E]
#         return norm.view(-1,1)*x_j #对每个边的消息进行归一化



# 拉格朗日-欧拉消息传递 (LA Message Passing) 的上下行传播机制。
# 实现了一种对称的消息传递模式，分别处理"上行"和"下行"传播，保持某种物理系统的对称性。
# 对称更新模式：
# 上行传播：p(t+1) = p(t) + f[q(t)]  用q更新p
# 下行传播：q(t+1) = q(t) + g[p(t)]  用p更新q
# 其中 f 和 g 都是 SymplecticMessagePassing 函数
# 设计意图：
# 1 保持能量守恒，通过对称的更新方式，可能保持系统的某种"能量"不变。
# 2 数值稳定性，交替更新 p 和 q 可以提供更好的数值稳定性。
# 3 物理意义，更新模式对应物理系统的辛积分器
# 可能的应用
# 分子动力学模拟: p=动量，q=位置
# 哈密顿神经网络: 学习物理系统的动力学
# 对称图神经网络: 需要保持特定不变性的任务
# 数值积分器: 离散化的物理系统模拟
# 注意事项
# 维度要求: 由于使用 SymplecticMessagePassing，p 和 q 必须有相同的维度
# 收敛性: 交替更新可能需要仔细设计以确保收敛
# 初始化: p和q的初始化可能很重要
# 这两个类实现了一种具有物理意义的对称消息传递机制：
# LAMessagePassingUp: 用 q 的信息更新 p，保持 q 不变
# LAMessagePassingDown: 用 p 的信息更新 q，保持 p 不变
# 这种设计保持了系统的辛结构，非常适合需要模拟物理系统或保持特定数学性质的图学习任务。通过交替使用这两种层，可以构建出具有良好理论性质的深度网络架构。
class LAMessagePassingUp(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingUp, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n) # 辛消息传递层
        

    def forward(self,p,q,edge_index):
        p = p + self.message_passing(q, edge_index) # 用q更新p
        q = q # q保持不变
        return p,q


class LAMessagePassingDown(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingDown, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n) # 辛消息传递层

    def forward(self,p,q,edge_index):
        p = p # p保持不变
        q = q + self.message_passing(p, edge_index) # 用p更新q
        return p,q




# 不同变体，都是基于交替更新 p 和 q 的对称模式，但采用了不同的更新策略
# 纯线性变换： q = q + W*p
# 移除了图结构信息（无 edge_index 使用）
# 相当于全连接层的对称版本
# 计算效率更高，但失去了图拓扑信息

# 使用场景建议
# 1. NoInteraction
# 当图结构不重要或过于嘈杂时,需要高效计算时,作为基线模型
# 2. Activation
# 需要建模非线性动力学时,图结构简单或已通过其他方式处理
# 3. Combined
# 复杂的图结构学习任务,需要最大表达能力的场景,物理系统的精细建模

class LAMessagePassingDown_NoInteraction(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingDown_NoInteraction, self).__init__()
        self.linear = SymmetricLinear(n,n, bias = False)

    def forward(self,p,q,edge_index):
        p = p
        q = q + self.linear(p) # 仅使用线性变换，无消息传递，特点：去除了图结构信息，只进行节点级别的线性变换
        return p,q
        
# 非线性激活: p = p + a*sigma(q)
# 可学习的逐维度缩放参数 a
# 非线性激活函数（Tanh）
# 为系统引入非线性动力学的能力
class ActivationUpModule(torch.nn.Module):
    def __init__(self,n):
        super(ActivationUpModule,self).__init__()
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self,p,q):
        sigma_q = self.activation(q)
        p = p + self.a*sigma_q # 使用激活函数和可学习参数，特点：引入非线性激活函数和可学习的缩放参数
        q = q
        return p,q
        
# 非线性激活: q = q + a*sigma(p)
class ActivationDownModule(torch.nn.Module):
    def __init__(self,n):
        super(ActivationDownModule,self).__init__()
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self,p,q):
        sigma_p = self.activation(p) # 使用激活函数和可学习参数，特点：引入非线性激活函数和可学习的缩放参数
        p = p
        q = self.a*sigma_p + q
        return p,q

# combine: p = p + sigma(W*q + a*q) 消息传递+残差+非线性
# 这是最复杂的变体，包含：
# 图消息传递: 聚合邻居信息
# 残差连接: 保持原始信息流
# 非线性变换: 增强表达能力
class LACombinedUp(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedUp, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index):
        q_prime = self.message_passing(q, edge_index) # 图消息传递
        q_prime = q_prime + self.a*q                  # 残差连接
        p = p + self.activation(q_prime)              # 非线性变换，特点：结合了图消息传递、残差连接和非线性激活
        q = q
        return p,q

class LACombinedDown(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedDown, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index):
        p_prime = self.message_passing(p, edge_index)   #[n,d]
        p_prime = p_prime + self.a*p                    #[n,d] + [1,d]*[n,d]
        p = p
        q = q + self.activation(p_prime)
        return p,q

############# Attention##########
# 带有注意力机制的辛消息传递, 在基础的辛消息传递基础上，引入了注意力机制来动态调整边的重要性权重
# 梯度稳定性: 注意力分数可能带来梯度问题，考虑使用softmax
# 计算复杂度: 注意力计算增加 O(E) 复杂度
# 过拟合风险: 更多参数需要适当正则化
# 这个带注意力的辛消息传递架构实现了：
# 结构感知: 通过度归一化保持图结构信息
# 内容感知: 通过注意力机制考虑节点特征相似性
# 物理约束: 通过对称线性变换保持辛结构
# 动态调整: 注意力权重允许模型自适应学习边重要性
# 这种设计非常适合需要同时考虑图结构和节点内容的复杂任务，特别是在物理系统建模或需要可解释性的应用中。注意力机制为传统的消息传递注入了新的灵活性。
class SymplecticMessagePassingAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = 'add')
        self.lin = SymmetricLinear(in_channels, out_channels, bias = False)
        self.bias= Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, att_scores):

        #x: [N, in_channels]
        #edge_index: [2,E]

        #edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]*att_scores # 加入注意力权重, 动态调整每条边的重要性
        #print(norm)
        out = self.propagate(edge_index, x=x, norm=norm)

        out = out + self.bias

        return out

    def message(self, x_j, norm):
        #x_j: [E, out_channels]
        return norm.view(-1,1)*x_j

# 在原有的LA架构基础上，传入注意力分数
# 对节点i的更新：  h(l+1)(i) = SymmetricLinear h(l)(i) + sum(j in N(i))( attention(ij)/sqrt(di*dj)*h(l)(j) ) + b
class LAMessagePassingUpAttention(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingUpAttention, self).__init__()
        self.message_passing = SymplecticMessagePassingAttention(n,n)
        

    def forward(self,p,q,edge_index, att_scores):
        p = p + self.message_passing(q, edge_index, att_scores)
        q = q
        return p,q

class LAMessagePassingDownAttention(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingDownAttention, self).__init__()
        self.message_passing = SymplecticMessagePassingAttention(n,n)

    def forward(self,p,q,edge_index, att_scores):
        p = p
        q = q + self.message_passing(p, edge_index, att_scores)
        return p,q


# 使用共享的线性层生成Q和K
# 计算边上的点积注意力分数
# 输出每个边的注意力权重 [E]
# 注意力计算：attention(ij) = Q(i) * K(j) = Linear(xi) * Linear(xj)
class Attention(torch.nn.Module):
    def __init__(self,n):
        super(Attention, self).__init__()
        self.QK = torch.nn.Linear(n,n, bias = False) # 共享的QK变换

    def forward(self, x, edge_index):
        Q = self.QK(x) # [N, n]
        K = self.QK(x) # [N, n]
        attention_scores = torch.sum(Q[edge_index[0]] * K[edge_index[1]], dim = 1) # [E]
        return attention_scores


# class DropoutUpModule(torch.nn.Module):
#     def __init__self(,n

# 基础的Combined架构基础上，引入了外部源项 (source)，这使得模型能够接收外部输入或偏置信号
# 更新规则：
# q` = messagepassing(q) + a*q + source
# p = p + sigma(q`)
# q = q
# 源项 (Source) 的作用
# 1. 外部驱动 force
# source 充当外部驱动 force
# 可以表示系统的输入信号或偏置

# 2. 偏置项的扩展
# 比固定的偏置参数更灵活
# 可以随时间或位置变化

# 3. 多模态融合
# 可以注入其他模态的信息
# 实现不同数据源的融合

# 设计优势
# 1. 增强表达能力 q_prime = q_prime + self.a * q + source   源项提供了额外的自由度来调整系统的动力学

# 2. 物理意义明确
# 在物理系统中，源项通常表示：
# 外力（力学系统）
# 源汇项（流体系统）
# 输入信号（控制系统）

# 3. 灵活的信息注入
# 源项可以来自：  其他神经网络模块的输出; 外部传感器的数据;  时间序列的编码

class LACombinedUpSource(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedUpSource, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index, source):
        q_prime = self.message_passing(q, edge_index)
        q_prime = q_prime + self.a*q + source # 加入源项
        p = p + self.activation(q_prime)
        q = q
        return p,q

class LACombinedDownSource(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedDownSource, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index, source):
        p_prime = self.message_passing(p, edge_index)   #[n,d]
        p_prime = p_prime + self.a*p + source                   #[n,d] + [1,d]*[n,d]
        p = p
        q = q + self.activation(p_prime)
        return p,q
        