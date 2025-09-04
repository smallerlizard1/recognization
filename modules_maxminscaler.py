import numpy as np
import joblib


class datasetphysicalSharedMaxMinScalers:
    def __init__(self,feature_range=(-1,1),scalers_seq=[1,2,3,4],loaded_scaler_file=None):
        self.feature_range = feature_range
        self.dataset_scales_seq = scalers_seq
        self.loaded_scaler_file = loaded_scaler_file
        # 统计一共有几个不同的数字,每个数字代表一个归一化器
        self.scales_type = np.unique(self.dataset_scales_seq)
        print(f"一共有{len(self.scales_type)}个不同的数字:", self.scales_type)
        #初始化归一化器
        self.indices = [] # 每一个归一化器的索引
        # 获取每个数字在数列中的索引
        print("每个数字在数列中的索引:")
        for scale_type in self.scales_type:
            self.indices.append(np.where(self.dataset_scales_seq == scale_type)[0])
            print(f"数字 {scale_type}: 索引 {self.indices[-1]}, (出现 {len(self.indices[-1])} 次)")
        self.data_min_ = None
        self.data_max_ = None
        self.is_fit = False
    
    def load_joblib_file(self):
        loaded_scaler_joblib = joblib.load(self.loaded_scaler_file) # 从文件中加载归一化参数，并与当前数据进行合并
        self.data_min_ = loaded_scaler_joblib.data_min_
        self.data_max_ = loaded_scaler_joblib.data_max_
        self.is_fit = True
        print("加载的归一化参数 data_min_:", loaded_scaler_joblib.data_min_)
        print("加载的归一化参数 data_max_:", loaded_scaler_joblib.data_max_)

    def fit(self, X):
        row,col = X.shape
        if col != len(self.dataset_scales_seq):
            raise ValueError(f"数据维度不一致！输入数据有 {col} 个特征，但期望 {len(self.dataset_scales_seq)} 个特征")
        if self.loaded_scaler_file == None:
            self.data_min_ = np.ones((col))
            self.data_max_ = np.ones((col))
            for i,scale_type in enumerate(self.scales_type):
                X_group = X[:,self.indices[i]]
                self.data_min_[self.indices[i]] = np.min(X_group) * np.ones((len(self.indices[i])))
                self.data_max_[self.indices[i]] = np.max(X_group) * np.ones((len(self.indices[i])))# 对应矩阵所有元素的最大值, 若输入轴参数可比较对应行或列信息
        # 从文件中加载归一化参数，并与当前数据进行合并
        else:
            loaded_scaler_joblib = joblib.load(self.loaded_scaler_file) 
            self.data_min_ = loaded_scaler_joblib.data_min_
            self.data_max_ = loaded_scaler_joblib.data_max_
            print("加载的归一化参数 data_min_:", loaded_scaler_joblib.data_min_)
            print("加载的归一化参数 data_max_:", loaded_scaler_joblib.data_max_)
            data_min_group = np.zeros((col))
            data_max_group = np.zeros((col))
            for i,scale_type in enumerate(self.scales_type):
                X_group = X[:,self.indices[i]]
                data_min_group[self.indices[i]] = np.min(X_group) * np.ones((len(self.indices[i])))
                data_max_group[self.indices[i]] = np.max(X_group) * np.ones((len(self.indices[i])))# 对应矩阵所有元素的最大值, 若输入轴参数可比较对应行或列信息
            if self.is_fit == True:
                self.data_min_ = np.minimum(self.data_min_,data_min_group)
                self.data_max_ = np.maximum(self.data_max_,data_max_group)
            else:
                self.data_min_ = data_min_group
                self.data_max_ = data_max_group
        self.is_fit = True
        return self
    def transform(self, X):
        """使用共享参数转换数据"""
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("必须先调用fit方法")
        min_range, max_range = self.feature_range
        # 注意：数据维度不一致，使用numpy的广播机制进行计算
        # NumPy 会自动将一维数组从 (4) 扩展为 (1,4)，然后再扩展为 (4,4)
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_) # 归一化到[0,1],
        # TODO 避免除零错误 如果最大最小值相同，避免除零
        X_scaled = X_std * (max_range - min_range) + min_range # 归一化到[-1,1]
        return X_scaled
    def fit_transform(self, X):
        """拟合并转换"""
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        """反归一化"""
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("必须先调用fit方法")
        min_range, max_range = self.feature_range
        X_std = (X - min_range) / (max_range - min_range)
        X_original = X_std * (self.data_max_ - self.data_min_) + self.data_min_
        return X_original
