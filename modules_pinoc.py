from __future__ import annotations
import time,math
import argparse
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import joblib

try:
    import pinocchio as pin
except Exception as e:
    raise ImportError("Pinocchio not available. Install with `pip install pin` or `conda install -c conda-forge pinocchio`.") from e

from dataset_split import loaddata,loaddata_preproc,loaddata_preproc_v2,extract_window_data
from user_utilities import count_parameters
from load_dataset_nn import load_dataset_zzm_train_valid_test,load_dataset_wdh_train_valid_test,load_dataset_lyl_train_valid_test
from load_dataset_nn import load_dataset_szh_train_valid_test,load_dataset_kdl_train_valid_test,load_dataset_ly_train_valid_test

# 优化 train_hamilton_h_feat.py


# =============================
# 1) 配置参数类（人体、环境、特征模型）
# =============================
@dataclass
class HumanAnthro:
    """人体测量学参数 (Anthropometry)包含人体整体质量、身高，以及每个身体段（躯干、大腿、小腿）的质量比例、长度比例、质心比例"""
    mass: float = 70.0 # 总质量，kg
    height: float = 1.75 # 总身高，m
    seg_mass_frac: torch.Tensor = torch.tensor([0.50, 0.10, 0.07, 0.10, 0.07]) # 比例
    seg_len_frac: torch.Tensor = torch.tensor([0.35, 0.245, 0.246, 0.245, 0.246])
    com_frac: torch.Tensor = torch.tensor([0.50, 0.433, 0.433, 0.433, 0.433])

@dataclass
class EnvContext:
    """环境上下文信息（外部条件） 包括坡度、楼梯台阶高度、运动模式 ID。"""
    slope_deg: float = 0.0
    stair_rise: float = 0.0
    mode_id: int = 0

@dataclass
class HFeatConfig:
    """H-Feat 特征层配置参数"""
    dof: int = 5   # 自由度数（5连杆模型）
    low_rank_r: int = 2  # 低秩近似阶数
    contact_potential_weight: float = 50.0  # 接触势能权重
    gravity: float = 9.80665  # 重力加速度
    symplectic_reg: float = 1e-4  # 辅助正则项（辛结构约束）
    energy_drift_reg: float = 1e-3  # 能量漂移正则
    hamilton_residual_reg: float = 1e-3  # 哈密顿残差正则
    mmd_weight: float = 1e-3  # MMD 损失权重
    phase_feat: bool = True  # 是否使用相位特征
    impulse_feat: bool = True  # 是否使用冲量特征
    dim_head: int = 128  # 分类头维度
    num_classes: int = 6  # 类别数
    lora_r: int = 4  # LoRA 低秩维度
    lora_alpha: float = 16.0  # LoRA 缩放因子

# =============================
# 2) Pinocchio连杆模型 5-link with IMU (fixed)
# =============================
class Pin5LinkWithIMU:
    """基于 Pinocchio 的 5 连杆平面模型,每个连杆带一个 IMU,关节为绕x轴旋转(对应人体矢状面运动)
    - 提供质量矩阵、势能、脚部高度、IMU 位姿等计算
    imu:安装的时候,x轴朝左 ,y轴朝上,z轴朝前
    pinocchio:x朝右,y轴朝前,z轴朝上
    这里所有都按照pinocchio默认坐标系来
    pinocchio使用旋量理论表示运动学,使用 李群SE(3)与李代数se(3)表示刚体运动
    哈密顿系统，假设骨盆中心为势能零位
    """
    def __init__(self, anth:HumanAnthro, dof:int=5, gravity:float=9.80665):
        assert dof == 5, "This helper is written for 5 DoF."
        self.anth = anth
        self.dof = dof
        self.g = float(gravity)

        # convert anthropometric params to numpy floats# 将人体测量学参数转换为 numpy 浮点数
        seg_mass = (anth.mass * anth.seg_mass_frac[:dof]).detach().cpu().numpy().astype(float)
        seg_len = (anth.height * anth.seg_len_frac[:dof]).detach().cpu().numpy().astype(float)
        com_frac = anth.com_frac[:dof].detach().cpu().numpy().astype(float)
        self.seg_mass = seg_mass
        self.seg_len = seg_len
        self.seg_com = seg_len * com_frac  # 质心位置 = 段长 * 比例

        # 创建 pinocchio 模型
        self.model = pin.Model()  
        # 设置重力（Pinocchio用Motion表示，前3为线速度，后3为角速度）,这里加载在z轴，pinocchio默认坐标系z轴朝上
        try:
            self.model.gravity = pin.Motion(np.array([0., 0., -self.g, 0., 0., 0.]))
        except Exception:
            # fallback: some pinocchio builds accept a numpy array for gravity
            self.model.gravity = np.array([0., 0., -self.g, 0., 0., 0.])

        parent_id = 0  # 父关节 ID，0 表示 world
        self.joint_ids = []
        self.body_frame_ids = []

        # 3. 添加躯干
        # 躯干固定在世界，可绕X旋转（前屈/后仰）
        joint_placement = pin.SE3.Identity() #初始位姿,世界坐标系原点在骨盆中心位置
        jmodel = pin.JointModelRX()
        jid = self.model.addJoint(parent_id, jmodel, joint_placement, 'torso_joint')
        self.joint_ids.append(jid)
        mass = self.seg_mass[0]
        L = self.seg_len[0]
        Ix = (mass * L**2)/12
        inertia = np.diag([Ix, 1e-6, 1e-6])
        com_local = np.array([0,0,self.seg_com[0]])
        self.model.appendBodyToJoint(jid, pin.Inertia(mass, com_local, inertia), pin.SE3.Identity())

        # 4. 左腿：大腿+小腿
        parent_id = jid
        # 左大腿髋关节
        joint_placement = pin.SE3(np.eye(3),np.array([-0.1, 0.0, 0.0])) #左大腿局部坐标系，没有旋转，x轴偏移
        jmodel = pin.JointModelRX()
        jid = self.model.addJoint(parent_id, jmodel, joint_placement, 'l_thigh_joint')
        self.joint_ids.append(jid)
        mass = self.seg_mass[1]
        L = self.seg_len[1]
        Ix = (mass * L**2)/12
        inertia = np.diag([Ix, 1e-6, 1e-6])
        com_local = np.array([0,0,-self.seg_com[1]])
        self.model.appendBodyToJoint(jid, pin.Inertia(mass, com_local, inertia), pin.SE3.Identity())

        # 左小腿膝关节
        parent_id = jid
        joint_placement = pin.SE3(np.eye(3), np.array([0,0,-L]))#左小腿局部坐标系，没有旋转，z轴偏移
        jmodel = pin.JointModelRX()
        jid = self.model.addJoint(parent_id, jmodel, joint_placement, 'l_shank_joint')
        self.joint_ids.append(jid)
        mass = self.seg_mass[2]
        L = self.seg_len[2]
        Ix = (mass * L**2)/12
        inertia = np.diag([Ix, 1e-6, 1e-6])
        com_local = np.array([0,0,-self.seg_com[2]])
        self.model.appendBodyToJoint(jid, pin.Inertia(mass, com_local, inertia), pin.SE3.Identity())

        # 5. 右腿：大腿+小腿
        parent_id = jid = 1  # 躯干关节
        # 右大腿髋关节
        joint_placement = pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.0]))  #右大腿局部坐标系，没有旋转，x轴偏移
        jmodel = pin.JointModelRX()
        jid = self.model.addJoint(parent_id, jmodel, joint_placement, 'r_thigh_joint')
        self.joint_ids.append(jid)
        mass = self.seg_mass[3]
        L = self.seg_len[3]
        Ix = (mass * L**2)/12
        inertia = np.diag([Ix, 1e-6, 1e-6])
        com_local = np.array([0,0,-self.seg_com[3]])
        self.model.appendBodyToJoint(jid, pin.Inertia(mass, com_local, inertia), pin.SE3.Identity())

        # 右小腿膝关节
        parent_id = jid
        joint_placement = pin.SE3(np.eye(3), np.array([0,0,-L]))#右小腿局部坐标系，没有旋转，z轴偏移
        jmodel = pin.JointModelRX()
        jid = self.model.addJoint(parent_id, jmodel, joint_placement, 'r_shank_joint')
        self.joint_ids.append(jid)
        mass = self.seg_mass[4]
        L = self.seg_len[4]
        Ix = (mass * L**2)/12
        inertia = np.diag([Ix, 1e-6, 1e-6])
        com_local = np.array([0,0,-self.seg_com[4]])
        self.model.appendBodyToJoint(jid, pin.Inertia(mass, com_local, inertia), pin.SE3.Identity())

        # end-effector local offset relative to last joint (foot at end of last link)
        # 足端相对于最后一个关节的局部偏移
        self.eel_local = np.array([0.0,0.0,-float(self.seg_len[2])], dtype=float) #[ 0.      0.     -0.4305]
        self.eer_local = np.array([0.0,0.0,-float(self.seg_len[-1])], dtype=float) #[ 0.      0.     -0.4305]

        # IMU mounting rotation: body -> imu
        # Map from body axes -> imu axes following user spec
        # We choose R_body_to_imu such that imu.x is left, imu.y up, imu.z front.
        # After experimenting with axis permutations, this matrix fits the described mapping:
        # (this matrix was derived earlier; keep consistent)
        # 定义 IMU 安装方向变换矩阵（body -> imu）
        self.R_body_to_imu = np.array([[0.0, 0.0, -1.0],
                                       [0.0, 1.0,  0.0],
                                       [1.0, 0.0,  0.0]], dtype=float)
        # create data # 创建数据缓存
        self.data = self.model.createData()

    # helper to convert torch/numpy to 1D numpy array of length D # 辅助函数：确保输入为 numpy 1D 向量
    @staticmethod
    def _ensure_numpy_1d(q_like, D:int):
        if isinstance(q_like, torch.Tensor):
            q_np = q_like.detach().cpu().reshape(-1).numpy().astype(float)
        else:
            q_np = np.array(q_like, dtype=float).reshape(-1)
        assert q_np.shape[0] == D, f"Expected length {D}, got {q_np.shape[0]}"
        return q_np

    # ---------- single-sample computations 单样本计算----------
    def mass_matrix_single(self, q_1d:np.ndarray) -> np.ndarray:
        """计算单个姿态下的质量矩阵 M(q)"""
        # q = self._ensure_numpy_1d(q_1d, self.dof)
        M = pin.crba(self.model, self.data, q_1d) # Composite Rigid Body Algorithm
        M = np.array(M, dtype=float)
        # tiny jitter to ensure numerical stability
        M[np.diag_indices_from(M)] += 1e-8  # 防止奇异
        return M

    def potential_energy_single(self, q_1d:np.ndarray) -> float:
        """计算势能:sum(m g z)"""
        # q = self._ensure_numpy_1d(q_1d, self.dof)
        # forward kinematics and place frames
        pin.forwardKinematics(self.model, self.data, q_1d) # 结果存储在self.data中，特别是data.oMi数组包含了每个关节在世界坐标系中的变换矩阵
        # pin.updateFramePlacements(self.model, self.data) # 更新坐标系的位置：如果模型中有定义额外的（frames），这个函数会更新它们的位置
        V = 0.0
        # sum m * g * z for each body's center of mass
        # model.inertias index corresponds to joints; inertias[0] is universe and usually empty,
        # so iterate joints 1..njoints-1 as earlier convention
        for jid in range(1, self.model.njoints): # 遍历所有关节（从1开始，因为0是世界坐标系/根节点）
            # lever (local com) stored in model.inertias[jid].lever
            # self.model.inertias[jid] 包含第jid个连杆的惯性参数
            # .lever 属性是质心在连杆本地坐标系中的3D位置向量
            com_local = self.model.inertias[jid].lever  # numpy array # 获取在连杆本地坐标系中的质心位置
            # self.data.oMi[jid] 是从世界坐标系到第jid个关节坐标系的变换矩阵
            # .act() 方法执行坐标变换：将点从本地坐标系转换到世界坐标系
            # 结果 com_world 是质心在世界坐标系中的3D位置
            com_world = self.data.oMi[jid].act(com_local)  # world position of com (3,) 将本地坐标系中的点坐标转换到世界坐标系  的质心
            # data.oMi[jid] 是一个 SE(3)变换矩阵,从世界坐标系到第jid个关节坐标系的变换
            # data.oMi[jid].rotation（旋转矩阵）
            # data.oMi[jid].translation（平移向量）
            # 这两行是注释，解释了data.oMi[jid]的内部结构：
            # - rotation: 3x3旋转矩阵，描述方向
            # - translation: 3D平移向量，描述位置
            z = float(com_world[2])# z方向
            # 提取质心在世界坐标系中的z坐标（高度）
            # 假设世界坐标系的z轴是垂直方向，向上为正
            mass = float(self.model.inertias[jid].mass)
            # 获取连杆的质量
            # self.model.inertias[jid].mass 是第jid个连杆的质量
            V += mass * self.g * z
            # 计算该连杆的势能并累加到总势能V中
            # 势能公式: E_potential = m * g * h
            # 其中 h 是高度（z坐标），self.g 是重力加速度常数（通常是9.81 m/s²）
        return float(V)

    def foot_height_single(self, q_1d:np.ndarray):
        """计算足端在世界坐标系下的高度 z"""
        # q = self._ensure_numpy_1d(q_1d, self.dof)
        pin.forwardKinematics(self.model, self.data, q_1d)
        pin.updateFramePlacements(self.model, self.data)
        # # last joint index is model.njoints-1
        # last_joint_idx = self.model.njoints - 1# 足端关节id
        # oMi_last = self.data.oMi[last_joint_idx]
        foot_world = [self.data.oMi[2].act(self.eer_local),self.data.oMi[-1].act(self.eer_local)]
        # foot_world = [oMi_last.act(self.eel_local),oMi_last.act(self.eer_local)]
        return foot_world

    def imu_pose_single(self, q_1d:np.ndarray):
        """
        Return per-link IMU poses as tuples (R, t) where R is 3x3 and t is 3-vector in world frame.
        Order: joints 1..dof (matches model.inertias indices).
        计算每个连杆上 IMU 的姿态（旋转 R,平移 t)
        """
        # q = self._ensure_numpy_1d(q_1d, self.dof)
        pin.forwardKinematics(self.model, self.data, q_1d)
        pin.updateFramePlacements(self.model, self.data)
        Rs = []
        Ts = []
        for jid in range(1, self.model.njoints):
            R_world_body = np.array(self.data.oMi[jid].rotation, dtype=float)
            t_world_body = np.array(self.data.oMi[jid].translation, dtype=float)
            # compute world rotation of imu frame: R_world_imu = R_world_body * R_body_to_imu
            # 世界坐标下的 IMU 姿态
            R_world_imu = R_world_body @ self.R_body_to_imu
            Rs.append(R_world_imu)
            Ts.append(t_world_body)
        # returns list length = dof, each R (3x3), t (3,)
        return Rs, Ts

    # ---------- batched wrappers 批处理封装----------
    def mass_matrix(self, q:torch.Tensor) -> torch.Tensor:
        """批量计算质量矩阵"""
        D = q.shape[-1]
        assert D == self.dof
        q_flat = q.reshape(-1, D).detach().cpu().numpy()
        Ms = [self.mass_matrix_single(qi) for qi in q_flat]
        Ms = np.stack(Ms, axis=0)
        return torch.from_numpy(Ms).to(q.device, dtype=q.dtype).reshape(q.shape[:-1] + (D, D))

    def potential_energy(self, q:torch.Tensor) -> torch.Tensor:
        """批量计算势能"""
        D = q.shape[-1]
        q_flat = q.reshape(-1, D).detach().cpu().numpy()
        Vs = [self.potential_energy_single(qi) for qi in q_flat]
        Vs = np.array(Vs, dtype=float)
        return torch.from_numpy(Vs).to(q.device, dtype=q.dtype).reshape(q.shape[:-1])

    def foot_height(self, q:torch.Tensor) -> torch.Tensor:
        """批量计算足端高度"""
        D = q.shape[-1]# 最后一个自由度，5
        q_flat = q.reshape(-1, D).detach().cpu().numpy()
        hs = [self.foot_height_single(qi) for qi in q_flat]
        hs = np.array(hs, dtype=float)
        return torch.from_numpy(hs).to(q.device, dtype=q.dtype).reshape(q.shape[:-1])

    def imu_pose(self, q:torch.Tensor):
        """批量计算 IMU 姿态
        Return R_imu (B, dof, 3, 3) and T_imu (B, dof, 3)
        """
        D = q.shape[-1]
        q_flat = q.reshape(-1, D).detach().cpu().numpy()
        Rs_all = []
        Ts_all = []
        for qi in q_flat:
            Rs, Ts = self.imu_pose_single(qi)
            Rs_all.append(np.stack(Rs, axis=0))   # (dof,3,3)
            Ts_all.append(np.stack(Ts, axis=0))   # (dof,3)
        Rs_all = np.stack(Rs_all, axis=0)  # (B_flat, dof,3,3)
        Ts_all = np.stack(Ts_all, axis=0)  # (B_flat, dof,3)
        return (torch.from_numpy(Rs_all).to(q.device, dtype=q.dtype).reshape(q.shape[:-1] + (self.dof, 3, 3)),
                torch.from_numpy(Ts_all).to(q.device, dtype=q.dtype).reshape(q.shape[:-1] + (self.dof, 3)))


# =============================
# Low-rank residual, ContactPotential (use exact foot height)
# =============================
class LowRankMassResidual(nn.Module):
    def __init__(self, dof:int, r:int):
        super().__init__()
        # 低秩矩阵 U (D x r)
        self.U = nn.Parameter(torch.randn(dof, r) * 0.01)
        # 对角向量 s (r)
        self.s = nn.Parameter(torch.randn(r) * 0.01)

    def forward(self) -> torch.Tensor:
        # 低秩修正质量矩阵 R = U * diag(s) * U^T
        return (self.U * self.s).matmul(self.U.T) #等价于U @ torch.diag(s) 避免了构建一个 r x r 的对角矩阵，计算效率更高，内存开销更小

class ContactPotential(nn.Module):
    def __init__(self, cfg:HFeatConfig, foot_height_fn):
        super().__init__()
        # 可学习的接触刚度权重
        self.w = nn.Parameter(torch.tensor(cfg.contact_potential_weight))
        # 精确足部高度函数（来自 pinocchio）
        self.foot_height_fn = foot_height_fn

    def forward(self, q:torch.Tensor, ctx:EnvContext) -> torch.Tensor:
        # use exact foot height computed by pinocchio
        # 计算足部高度 h
        h = self.foot_height_fn(q)  # (...,) tensor
        # 根据地形调整接触强度（坡度/台阶）
        gamma = 1.0 + 0.2 * abs(ctx.slope_deg)/10.0 + 0.5 * (ctx.stair_rise > 0.0)
        # 势能 Vc = 0.5 * w * gamma * relu(-h)^2 （仅在穿透时生效）
        Vc = 0.5 * (self.w * gamma) * (torch.relu(-h))**2
        return Vc

# =============================
# 5) H-Feat (use RBDExactWithPinIMU) 特征提取
# =============================
class HFeat(nn.Module):
    def __init__(self, cfg:HFeatConfig, anth:HumanAnthro):
        super().__init__()
        self.cfg = cfg
        self.anth = anth
        # 使用 pinocchio 精确 RBD 计算
        self.rbd_exact = Pin5LinkWithIMU(anth, cfg.dof, cfg.gravity)
        # 接触势能（基于精确足高）
        self.contact = ContactPotential(cfg, foot_height_fn=self.rbd_exact.foot_height)
        # 低秩质量矩阵修正
        self.lowrankM = LowRankMassResidual(cfg.dof, cfg.low_rank_r)

    def _mass(self, q:torch.Tensor) -> torch.Tensor:
        # 基础质量矩阵 (B,D,D)
        M = self.rbd_exact.mass_matrix(q)  # (...,D,D)
        # 添加低秩修正
        R = self.lowrankM()          # (D,D)
        M = M + R                    # broadcast
        # 防止奇异，添加微小扰动
        M = M + 1e-6 * torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
        return M
    
    def forward(self,x) -> torch.Tensor:
        input_ndim = x.dim()
        # 处理批量图数据: [batch_size, num_nodes, feature_num]
        if input_ndim == 3:
            batch_size, num_nodes, features_pernode = x.shape
            q = x[:,:,[2,7,12,17,22]]
            qd = x[:,:,[3,8,13,18,23]]
            # q:torch.Tensor, qd:torch.Tensor, ctx:EnvContext
            # q: (B,T,D) or (B,D) or (D,)# 质量矩阵
            M = self._mass(q)
            # print(f"M:{M}")  5*5
            # 动量 p = M * qd
            p = torch.einsum('...ij,...j->...i', M, qd)  #矩阵-向量 相乘
            # print(f"p:{p.shape}") 5维度的向量
            # 动能 T = 0.5 * qd^T * M * qd
            Tkin = 0.5 * torch.einsum('...i,...i->...', qd, p)
            # 重力势能
            Vg = self.rbd_exact.potential_energy(q)
            # 接触势能
            # Vc = self.contact(q, ctx)
            Vc = 0
            # 哈密顿量 H = T + V
            H = Tkin + Vg + Vc
            feats = {'M': M, 'p': p, 'T': Tkin, 'Vg': Vg, 'Vc': Vc, 'H': H}
            # 相位特征（如果启用）
            if self.cfg.phase_feat:
                x = q[..., -1]
                y = qd[..., -1]
                feats['phi'] = torch.atan2(y, 1e-6 + x)
            # IMU poses from pinocchio (R: (..., dof,3,3), T: (..., dof,3))
            # 计算 IMU 位姿 (旋转+平移)
            R_imu, T_imu = self.rbd_exact.imu_pose(q)
            feats['R_imu'] = R_imu
            feats['T_imu'] = T_imu
            return M,Tkin,Vg,H
        # 单个图: [num_nodes, feature_num]
        elif input_ndim == 2:
            pass

        return feats

    # def forward(self, q:torch.Tensor, qd:torch.Tensor, ctx:EnvContext) -> Dict[str, torch.Tensor]:
    #     # q: (B,T,D) or (B,D) or (D,)# 质量矩阵
    #     M = self._mass(q)
    #     # 动量 p = M * qd
    #     p = torch.einsum('...ij,...j->...i', M, qd)
    #     # 动能 T = 0.5 * qd^T * M * qd
    #     Tkin = 0.5 * torch.einsum('...i,...i->...', qd, p)
    #     # 重力势能
    #     Vg = self.rbd_exact.potential_energy(q)
    #     # 接触势能
    #     # Vc = self.contact(q, ctx)
    #     Vc = 0
    #     # 哈密顿量 H = T + V
    #     H = Tkin + Vg + Vc
    #     feats = {'M': M, 'p': p, 'T': Tkin, 'Vg': Vg, 'Vc': Vc, 'H': H}
    #     # 相位特征（如果启用）
    #     if self.cfg.phase_feat:
    #         x = q[..., -1]
    #         y = qd[..., -1]
    #         feats['phi'] = torch.atan2(y, 1e-6 + x)
    #     # IMU poses from pinocchio (R: (..., dof,3,3), T: (..., dof,3))
    #     # 计算 IMU 位姿 (旋转+平移)
    #     R_imu, T_imu = self.rbd_exact.imu_pose(q)
    #     feats['R_imu'] = R_imu
    #     feats['T_imu'] = T_imu
    #     return feats


# =============================
# Full Model Wrapper 整体模型封装
# =============================
class HFeatIntentModel(nn.Module):
    def __init__(self, cfg:HFeatConfig, anth:HumanAnthro):
        super().__init__()
        self.cfg = cfg
        self.hfeat = HFeat(cfg, anth) # H-Feat 特征提取
        base_feats = 3 + cfg.dof * 3 # T,Vg,H + p,q,qd
        if cfg.phase_feat:
            base_feats += 1
        # 双向 LSTM 提取时序特征
        self.bilstm = nn.LSTM(input_size=base_feats, hidden_size=cfg.dim_head//2, num_layers=1,
                            batch_first=True, bidirectional=True)

        self.fc = nn.Linear(in_features=cfg.dim_head,out_features=cfg.num_classes)

    def forward(self, q:torch.Tensor, qd:torch.Tensor, dt:float, ctx:EnvContext):
        feats = self.hfeat(q, qd, ctx)  # 物理特征提取
        # 拼接输入特征
        parts=[feats['T'].unsqueeze(-1),feats['Vg'].unsqueeze(-1),feats['H'].unsqueeze(-1),feats['p'],q,qd]
        if 'phi' in feats:
            parts.append(feats['phi'].unsqueeze(-1))
        X = torch.cat(parts,dim=-1)  # (B,T,F)

        # print(X.shape) #torch.Size([32, 50, 19])
        out,(h_n, c_n) = self.bilstm(X)#双向LSTM输出维度翻倍(batch_size, sequence_length, hidden_size*2)
        #h_n (最终隐藏状态):形状: (num_layers * 2, batch_size, hidden_size)双向LSTM
        #c_n (最终细胞状态):形状同 h_n，但存储的是LSTM的细胞状态
        out = out[:, -1, :]# 取LSTM最后一个时间步的隐藏状态，输出至全连接层
        logits = self.fc(out)
        # return logits
        # logits = self.head(X)  # 分类输出
        return logits,feats,(q,qd)


class PhysicsLosses(nn.Module):
    def __init__(self, cfg:HFeatConfig):
        super().__init__()
        self.cfg = cfg

    def energy_drift(self, H:torch.Tensor):
        # 能量漂移正则化（相邻时间步的能量差平方和）
        if H.shape[1] < 2:
            return torch.tensor(0., device=H.device)
        dH = H[:,1:] - H[:,:-1]
        return dH.pow(2).mean()

    def forward(self, q, qd, feats:Dict[str,torch.Tensor]):
        loss = 0.0
        if self.cfg.energy_drift_reg > 0:
            loss += self.cfg.energy_drift_reg * self.energy_drift(feats['H'])
        return loss





def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备设置
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='hamilton h_feat motion recognization')
    # parser.add_argument('--train_newmodel', type=bool, required=True, help='flag train new model') #required=True, 参数用户必须提供
    parser.add_argument('--train_newmodel', 
                   action='store_true',  # 用户提供时设为 True，否则为 False
                   help='flag to train a new model')
    # python train_CNN_LSTM1.py                  # args.train_newmodel == False
    # python train_CNN_LSTM1.py --train_newmodel # args.train_newmodel == True
    parser.add_argument('--test',action='store_true', help='test model')
    parser.add_argument('--dataset', type=str, default='wdh', help='dataset')
    args = parser.parse_args()

    batch_size = 32
    window_size = 50
    train_datazzm_norm,train_labelzzm_raw,validate_datazzm_norm,validate_labelzzm_raw,test_datazzm_norm,test_labelzzm_raw = load_dataset_zzm_train_valid_test(batch_size,window_size,"physics")
    # train_datawdh_norm,train_labelwdh_raw,validate_datawdh_norm,validate_labelwdh_raw,test_datawdh_norm,test_labelwdh_raw = load_dataset_wdh_train_valid_test(batch_size,window_size)
    # train_datalyl_norm,train_labellyl_raw,validate_datalyl_norm,validate_labellyl_raw,test_datalyl_norm,test_labellyl_raw = load_dataset_lyl_train_valid_test(batch_size,window_size)
    # train_dataszh_norm,train_labelszh_raw,validate_dataszh_norm,validate_labelszh_raw,test_dataszh_norm,test_labelszh_raw = load_dataset_szh_train_valid_test(batch_size,window_size)
    # train_datakdl_norm,train_labelkdl_raw,validate_datakdl_norm,validate_labelkdl_raw,test_datakdl_norm,test_labelkdl_raw = load_dataset_kdl_train_valid_test(batch_size,window_size)
    # train_dataly_norm, train_labelly_raw, validate_dataly_norm, validate_labelly_raw, test_dataly_norm, test_labelly_raw  = load_dataset_ly_train_valid_test(batch_size,window_size)

    # train_data_norm = np.concatenate([train_datazzm_norm,train_datawdh_norm,train_datalyl_norm,train_dataszh_norm,train_datakdl_norm,train_dataly_norm], axis=0)# 合并数据集（沿样本维度拼接）
    # train_label_raw = np.concatenate([train_labelzzm_raw, train_labelwdh_raw,train_labellyl_raw,train_labelszh_raw,train_labelkdl_raw,train_labelly_raw], axis=0)
    # validate_data_norm = np.concatenate([validate_datazzm_norm, validate_datawdh_norm,validate_datalyl_norm,validate_dataszh_norm,validate_datakdl_norm,validate_dataly_norm], axis=0)
    # validate_label_raw = np.concatenate([validate_labelzzm_raw, validate_labelwdh_raw,validate_labellyl_raw,validate_labelszh_raw,validate_labelkdl_raw,validate_labelly_raw], axis=0)
    # test_data_norm = np.concatenate([test_datazzm_norm, test_datawdh_norm,test_datalyl_norm,test_dataszh_norm,test_datakdl_norm,test_dataly_norm], axis=0)
    # test_label_raw = np.concatenate([test_labelzzm_raw, test_labelwdh_raw,test_labellyl_raw,test_labelszh_raw,test_labelkdl_raw,test_labelly_raw], axis=0)

    train_data_norm = np.concatenate([train_datazzm_norm,], axis=0)# 合并数据集（沿样本维度拼接）
    train_label_raw = np.concatenate([train_labelzzm_raw, ], axis=0)
    validate_data_norm = np.concatenate([validate_datazzm_norm, ], axis=0)
    validate_label_raw = np.concatenate([validate_labelzzm_raw, ], axis=0)
    test_data_norm = np.concatenate([test_datazzm_norm, ], axis=0)
    test_label_raw = np.concatenate([test_labelzzm_raw, ], axis=0)

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

    # # 初始化模型、损失函数和优化器
    feature_number = train_X.shape[2]
    window_size = train_X.shape[1]
    num_hidden_units = 64  # 假设双向LSTM隐藏单元数量为64
    num_classes = 6  # 6个类别
    patience = 5 # 越小越好
    ctx = EnvContext()
    cfg = HFeatConfig()
    anth = HumanAnthro()
    phys = PhysicsLosses(cfg)
    dt = 1/100.0
    # print(f"feature_number:{feature_number},window_size:{window_size}")
    model = HFeatIntentModel(cfg,anth).to(device)
    if args.train_newmodel == False:
        try:
            print("load existing hamilton_hfeat_cnn_bilstm1_model")
            model.load_state_dict(torch.load('hamilton_hfeat_cnn_bilstm1_model_params.pth',weights_only=True))#从文件加载 model
        except FileNotFoundError as e:
            print(f"未找到hamilton_hfeat_cnn_bilstm1 model文件: {e}")
            args.train_newmodel = True  # 自动切换到使用新模型
        except Exception as e:
            print(f"错误: 加载模型失败（文件可能损坏）。{e}")
            raise
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)#L2正则话，但应设置小一些，避免与batchNorm冲突
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)# 使用学习率调度器

    print(f"模型参数量: {count_parameters(model)}")

    writer = SummaryWriter('./logs/hamilton_hfeat_cnn_bilstm1_train') # 日志保存目录
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
        all_preds_train = torch.tensor([],device=device)
        all_labels_train = torch.tensor([],device=device)
        for i, (batch_x, batch_y) in enumerate(train_loader): #为迭代对象添加计数
            # print(batch_x.shape) # torch.Size([32, 50, 25])
            ang = batch_x[:,:,10:15].to(device) #切片是左闭右开
            gyro = batch_x[:,:,15:20].to(device)
            # print(ang.shape) #torch.Size([32, 50, 5])
            batch_y = batch_y.to(device)
            logits, feats, (q, qd) = model(ang, gyro, dt, ctx)# 前向传播
            loss_task = criterion(logits, batch_y)# 分类损失
            q = q.requires_grad_(True)
            loss_phys = phys(q,qd,feats) # 物理约束损失
            loss = loss_task + loss_phys
            optimizer.zero_grad()
            loss.backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            all_preds_train = torch.cat([all_preds_train, predicted])# 直接在 GPU 上拼接张量
            all_labels_train = torch.cat([all_labels_train, batch_y])
            if i % 10 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(), epoch*len(train_loader)+i)# 每10个batch记录一次 参数：[指标名称和分类, 要记录的标量值, 全局步数]
        train_loss /= len(train_loader)
        train_accuracy = (all_preds_train == all_labels_train).float().mean().item() # GPU操作

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
                ang = batch_x[:,:,10:15].to(device) #切片是左闭右开
                gyro = batch_x[:,:,15:20].to(device)
                batch_y = batch_y.to(device)
                logits, feats, (q, qd) = model(ang, gyro, dt, ctx)# 前向传播
                loss_task = criterion(logits, batch_y)# 分类损失
                q = q.requires_grad_(True)
                loss_phys = phys(q,qd,feats) # 物理约束损失
                loss = loss_task + loss_phys
                validate_loss+=loss.item()
                _, predicted = torch.max(logits.data, 1)
                all_preds_validate = torch.cat([all_preds_validate, predicted])# 直接在 GPU 上拼接张量
                all_labels_validate = torch.cat([all_labels_validate, batch_y])
        validate_loss /= len(validate_loader)
        validate_accuracy = (all_preds_validate == all_labels_validate).float().mean().item()#GPU操作
        # validate_accuracy = np.sum(np.array(all_preds_validate) == np.array(all_labels_validate)) / len(all_labels_validate) #验证集准确率
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
            torch.save(model, "hamilton_hfeat_cnn_bilstm1_model_all.pth")#保存完整模型
            torch.save(model.state_dict(), "hamilton_hfeat_cnn_bilstm1_model_params.pth")#只保存模型参数(state_dict)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        epoch_time = time.time()-epoch_start_time
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Validate Loss {validate_loss:.4f}, Time_perbatch {epoch_time/len(train_loader)*1000:.2f}ms, Train iter {len(train_loader)}')
    
    writer.close()# 关闭TensorBoard writer

    # 测试模型
    model_test = HFeatIntentModel(cfg,anth).to(device)
    try:
        model_test.load_state_dict(torch.load('hamilton_hfeat_cnn_bilstm1_model_params.pth',weights_only=True))#从文件加载 model
    except FileNotFoundError as e:
        print(f"未找到 hamilton_hfeat_cnn_bilstm1 model文件: {e}")
    model_test.eval()
    all_preds_test = torch.tensor([],device=device)#GPU
    all_labels_test = torch.tensor([],device=device)
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            ang = batch_x[:,:,10:15].to(device) #切片是左闭右开
            gyro = batch_x[:,:,15:20].to(device)
            batch_y = batch_y.to(device)
            logits, feats, (q, qd) = model(ang, gyro, dt, ctx)# 前向传播
            _, predicted = torch.max(logits.data, 1)
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

def test_imu5():
    print("test imu5")
    ctx = EnvContext()
    cfg = HFeatConfig()
    anth = HumanAnthro()
    phys = PhysicsLosses(cfg)

    print(cfg)

    rbd_exact = Pin5LinkWithIMU(anth, cfg.dof, cfg.gravity)

    nq = rbd_exact.model.nq
    print(f"nq:{nq}") # nq:5 # 配置向量 q 的维度
    nv = rbd_exact.model.nv
    print(f"nv:{nv}") # nv:5

    q = np.zeros((cfg.dof),dtype=np.float32)
    q[1] = 0#3.14/4
    q[2] = -1.57/2
    print(q)

    print(rbd_exact.eel_local)
    print(rbd_exact.eer_local)

    motion_vector = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])  # [vx, vy, vz, ωx, ωy, ωz]
    motion = pin.Motion(motion_vector)
    print("从numpy数组创建:", motion)

    pin.forwardKinematics(rbd_exact.model,rbd_exact.data,q)

    for jid in range(1, rbd_exact.model.njoints): # 遍历所有关节（从1开始，因为0是世界坐标系/根节点）
        joint_type = rbd_exact.model.joints[jid]
        # lever (local com) stored in model.inertias[jid].lever
        # self.model.inertias[jid] 包含第jid个连杆的惯性参数
        # .lever 属性是质心在连杆本地坐标系中的3D位置向量
        com_local = rbd_exact.model.inertias[jid].lever  # numpy array # 获取在连杆本地坐标系中的质心位置
        print(f"com_local:{com_local}")
        # self.data.oMi[jid] 是从世界坐标系到第jid个关节坐标系的变换矩阵
        # .act() 方法执行坐标变换：将点从本地坐标系转换到世界坐标系
        # 结果 com_world 是质心在世界坐标系中的3D位置
        com_world = rbd_exact.data.oMi[jid].act(com_local)  # world position of com (3,) 将本地坐标系中的点坐标转换到世界坐标系  的质心
        print(f"com_world:{com_world}")
        # data.oMi[jid].rotation（旋转矩阵）
        # data.oMi[jid].translation（平移向量）
        # 这两行是注释，解释了data.oMi[jid]的内部结构：
        # - rotation: 3x3旋转矩阵，描述方向
        # - translation: 3D平移向量，描述位置

        print(f"translation:{rbd_exact.data.oMi[jid].translation}")
        print(f"rotaion:{rbd_exact.data.oMi[jid].rotation}")
        print("----------------------------------------")

    # rbd_exact.mass_matrix(q)

    # rbd_exact.potential_energy(q)


if __name__ == '__main__':
    main()
    # test_imu5()
