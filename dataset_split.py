import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt,savgol_filter
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from readbin import read_bin_file
from readbin_v2 import read_bin_file_v2


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    巴特沃斯带通滤波器
    
    参数:
        data: 输入信号 (n_samples, n_channels)
        lowcut: 低截止频率(Hz)
        highcut: 高截止频率(Hz)
        fs: 采样频率(Hz)
        order: 滤波器阶数
        
    返回:
        滤波后的信号
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)  # 使用filtfilt实现零相位滤波
    return y


def test_butter_bandpass_filter():
    """
    测试 butter_bandpass_filter 函数
    
    生成包含多个频率成分的测试信号，应用带通滤波后
    绘制原始信号、滤波后信号和频谱对比图
    """
    # 参数设置
    fs = 1000.0       # 采样率 (Hz)
    duration = 1.0    # 信号时长 (秒)
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # 生成测试信号：包含低频、带通范围内频率和高频成分
    low_freq = 5.0    # 低频 (Hz)
    pass_freq1 = 20.0 # 带通范围内频率 (Hz)
    pass_freq2 = 50.0 # 带通范围内频率 (Hz)
    high_freq = 100.0 # 高频 (Hz)
    
    signal = (np.sin(2 * np.pi * low_freq * t) + 
             0.5 * np.sin(2 * np.pi * pass_freq1 * t) + 
             0.8 * np.sin(2 * np.pi * pass_freq2 * t) + 
             0.3 * np.sin(2 * np.pi * high_freq * t))
    
    # 设置滤波器参数
    lowcut = 15.0     # 低截止频率 (Hz)
    highcut = 60.0    # 高截止频率 (Hz)
    order = 4         # 滤波器阶数
    
    # 应用滤波器
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order)
    
    # 计算频谱
    def compute_fft(signal, fs):
        n = len(signal)
        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(n, 1/fs)
        return freq[:n//2], np.abs(fft[:n//2])*2/n
    
    freq, orig_fft = compute_fft(signal, fs)
    _, filtered_fft = compute_fft(filtered_signal, fs)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    # 绘制时域信号
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='raw', alpha=0.7)
    plt.plot(t, filtered_signal, label='filter', linewidth=2)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('time area')
    plt.legend()
    plt.grid(True)
    
    # 绘制频域信号
    plt.subplot(2, 1, 2)
    plt.plot(freq, orig_fft, label='raw', alpha=0.7)
    plt.plot(freq, filtered_fft, label='filter', linewidth=2)
    plt.axvline(lowcut, color='red', linestyle='--', label='lowcut off')
    plt.axvline(highcut, color='red', linestyle='--',label='highcut off')
    plt.xlabel('freq (Hz)')
    plt.ylabel('amplitude')
    plt.title('frequence')
    plt.xlim(0, high_freq * 1.2)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()

    # 保存图像到本地文件夹
    os.makedirs("results_temp", exist_ok=True)  # 自动创建文件夹（如果不存在）
    output_path = os.path.join("results_temp", "filter_result.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {os.path.abspath(output_path)}")

    plt.show(block=False)  # 非阻塞显示
    
    # 打印测试结果
    print("滤波器测试完成")
    print(f"滤波器参数: 低截止={lowcut}Hz, 高截止={highcut}Hz, 阶数={order}")
    print("观察图形验证:")
    print("1. 滤波后信号应保留20Hz和50Hz成分")
    print("2. 5Hz和100Hz成分应被显著衰减")
    
    # 返回结果供后续处理
    return {
        'time': t,
        'original_signal': signal,
        'filtered_signal': filtered_signal,
        'frequency': freq,
        'original_fft': orig_fft,
        'filtered_fft': filtered_fft
    }


def loaddata(filepath,startpoint,endpoint,data_png_name):
    '''
    导入数据,划分窗口,划分数据集721
    不打乱,不归一化
    '''
    # 导入文件
    data_raw = read_bin_file(filepath=filepath,startpoint=startpoint,endpoint=endpoint)
    file_directory = os.path.dirname(filepath)
    print(filepath)
    dimension = data_raw['sensor_IMU_rpy'].shape #(行 列)
    datarows = dimension[0] #行数
    # cols = dimension[1] #列数
    # print(f"data import shape:{rows},{cols}")  # 添加的打印语句
    # dimension = data_raw['sensor_IMU_rpy'].ndim # 维度
    # print(dimension) #2
    feature_num = 8
    # 原始数据
    dataset_np = np.zeros((datarows, feature_num))# rows行，列
    dataset_np[:,0] = data_raw['sensor_IMU_acc'][:,3]  # 左髋x轴加速度
    dataset_np[:,1] = data_raw['sensor_IMU_acc'][:,4]  # 左髋y轴加速度
    dataset_np[:,2] = data_raw['sensor_IMU_rpy'][:,3]  # 左髋x轴角度    #012 345 678 91011 121314
    dataset_np[:,3] = data_raw['sensor_IMU_gyro'][:,3]  # 左髋x轴角速度

 
    dataset_np[:,4] = data_raw['sensor_IMU_acc'][:,9]  # 右髋x轴加速度
    dataset_np[:,5] = data_raw['sensor_IMU_acc'][:,10]  # 右髋y轴加速度
    dataset_np[:,6] = data_raw['sensor_IMU_rpy'][:,9]  # 右髋x轴角度   
    dataset_np[:,7] = data_raw['sensor_IMU_gyro'][:,9]  # 右髋x轴角速度

    # 绘制结果,存盘
    fig1 = plt.figure(figsize=(12, 8))
    t = np.linspace(0, datarows, datarows, endpoint=False)# 即 t = [0, 1, 2, ..., datarows-1]
    plt.subplot(4, 1, 1)
    plt.plot(t, dataset_np[:,0], label='x_acc_l', linewidth=2)
    plt.plot(t, dataset_np[:,4], label='x_acc_r', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(t, dataset_np[:,1], label='y_acc_l', linewidth=2)
    plt.plot(t, dataset_np[:,5], label='y_acc_r', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(t, dataset_np[:,2], label='x_ang_l', linewidth=2)
    plt.plot(t, dataset_np[:,6], label='x_ang_r', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(t, dataset_np[:,3], label='x_angvel_l', linewidth=2)
    plt.plot(t, dataset_np[:,7], label='x_angvel_r', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)  # 非阻塞显示
    results_dir = os.path.join(file_directory, "results_temp")
    os.makedirs(results_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    output_path = os.path.join(results_dir, data_png_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {os.path.abspath(output_path)}")
    plt.close(fig1)
    # 对每个通道应用滤波器
    # filtered_data = np.zeros_like(dataset_np)
    # for i in range(dataset_np.shape[1]):  # 对每个特征通道
    #     filtered_data[:, i] = butter_bandpass_filter(dataset_np[:, i],lowcut=0.5,highcut=20.0,fs=50)#fs: 采样频率(Hz) lowcut: 低截止频率(Hz) highcut: 高截止频率(Hz)
    
    # return filtered_data
    return dataset_np


def loaddata_preproc(filepath,startpoint,endpoint,data_png_name):
    '''
    导入原始数据,添加数据的角加速度
    划分窗口,划分数据集721
    不打乱,不归一化
    5连杆生物力学模型: 躯干、左大腿、左小腿、右大腿、右小腿
    '''
    # 导入文件
    data_raw = read_bin_file(filepath=filepath,startpoint=startpoint,endpoint=endpoint)
    file_directory = os.path.dirname(filepath)
    print(filepath)
    dimension = data_raw['sensor_IMU_rpy'].shape #(行 列)
    datarows = dimension[0] #行数
    # cols = dimension[1] #列数
    # print(f"data import shape:{rows},{cols}")  # 添加的打印语句
    # dimension = data_raw['sensor_IMU_rpy'].ndim # 维度
    # print(dimension) #2

    # 使用 Savitzky-Golay 滤波器平滑并求导（适合噪声数据）

    window_length, polyorder = 11, 3  # savgol_filter函数参数。Savitzky-Golay 滤波器平滑并求导（适合噪声数据）
    feature_num = 5*5
    # 原始数据
    dataset_np = np.zeros((datarows, feature_num))# rows行，列
    dataset_np[:,0] = data_raw['sensor_IMU_acc'][:,0]  # 躯干x轴加速度
    dataset_np[:,1] = data_raw['sensor_IMU_acc'][:,1]  # 躯干y轴加速度
    dataset_np[:,2] = data_raw['sensor_IMU_rpy'][:,0]  # 躯干x轴角度    #012 345 678 91011 121314
    dataset_np[:,3] = data_raw['sensor_IMU_gyro'][:,0]  # 躯干x轴角速度
    dataset_np[:,4] = savgol_filter(data_raw['sensor_IMU_gyro'][:,0], window_length, polyorder, deriv=1, delta=0.01)  # 躯干x轴角加速度

    dataset_np[:,5] = data_raw['sensor_IMU_acc'][:,3]  # 左大腿x轴加速度
    dataset_np[:,6] = data_raw['sensor_IMU_acc'][:,4]  # 左大腿y轴加速度
    dataset_np[:,7] = data_raw['sensor_IMU_rpy'][:,3]  # 左大腿x轴角度    #012 345 678 91011 121314
    dataset_np[:,8] = data_raw['sensor_IMU_gyro'][:,3]  # 左大腿x轴角速度
    dataset_np[:,9] = savgol_filter(data_raw['sensor_IMU_gyro'][:,3], window_length, polyorder, deriv=1, delta=0.01)  # 左大腿x轴角加速度

    dataset_np[:,10] = data_raw['sensor_IMU_acc'][:,6]  # 左小腿x轴加速度
    dataset_np[:,11] = data_raw['sensor_IMU_acc'][:,7]  # 左小腿y轴加速度
    dataset_np[:,12] = data_raw['sensor_IMU_rpy'][:,6]  # 左小腿x轴角度    #012 345 678 91011 121314
    dataset_np[:,13] = data_raw['sensor_IMU_gyro'][:,6]  # 左小腿x轴角速度
    dataset_np[:,14] = savgol_filter(data_raw['sensor_IMU_gyro'][:,6], window_length, polyorder, deriv=1, delta=0.01)  # 左小腿x轴角加速度

    dataset_np[:,15] = data_raw['sensor_IMU_acc'][:,9]  # 右大腿x轴加速度
    dataset_np[:,16] = data_raw['sensor_IMU_acc'][:,10]  # 右大腿y轴加速度
    dataset_np[:,17] = data_raw['sensor_IMU_rpy'][:,9]  # 右大腿x轴角度   
    dataset_np[:,18] = data_raw['sensor_IMU_gyro'][:,9]  # 右大腿x轴角速度
    dataset_np[:,19] = savgol_filter(data_raw['sensor_IMU_gyro'][:,9], window_length, polyorder, deriv=1, delta=0.01)  # 右大腿x轴角加速度

    dataset_np[:,20] = data_raw['sensor_IMU_acc'][:,12]  # 右小腿x轴加速度
    dataset_np[:,21] = data_raw['sensor_IMU_acc'][:,13]  # 右小腿y轴加速度
    dataset_np[:,22] = data_raw['sensor_IMU_rpy'][:,12]  # 右小腿x轴角度   
    dataset_np[:,23] = data_raw['sensor_IMU_gyro'][:,12]  # 右小腿x轴角速度
    dataset_np[:,24] = savgol_filter(data_raw['sensor_IMU_gyro'][:,12], window_length, polyorder, deriv=1, delta=0.01)  # 右小腿x轴角加速度

    # 绘制结果,存盘
    fig1 = plt.figure(figsize=(12,8))
    t = np.linspace(0, datarows, datarows, endpoint=False)# 即 t = [0, 1, 2, ..., datarows-1]
    plt.subplot(5, 1, 1)
    plt.plot(t, dataset_np[:, 0], label='x_acc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 5], label='x_acc_lt', linewidth=2)
    plt.plot(t, dataset_np[:,10], label='x_acc_ls', linewidth=2)
    plt.plot(t, dataset_np[:,15], label='x_acc_rt', linewidth=2)
    plt.plot(t, dataset_np[:,20], label='x_acc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 2)
    plt.plot(t, dataset_np[:, 1], label='y_acc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 6], label='y_acc_lt', linewidth=2)
    plt.plot(t, dataset_np[:,11], label='y_acc_ls', linewidth=2)
    plt.plot(t, dataset_np[:,16], label='y_acc_rt', linewidth=2)
    plt.plot(t, dataset_np[:,21], label='y_acc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 3)
    plt.plot(t, dataset_np[:, 2], label='x_ang_b', linewidth=2)
    plt.plot(t, dataset_np[:, 7], label='x_ang_lt', linewidth=2)
    plt.plot(t, dataset_np[:,12], label='x_ang_ls', linewidth=2)
    plt.plot(t, dataset_np[:,17], label='x_ang_rt', linewidth=2)
    plt.plot(t, dataset_np[:,22], label='x_ang_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 4)
    plt.plot(t, dataset_np[:, 3], label='x_angvel_b', linewidth=2)
    plt.plot(t, dataset_np[:, 8], label='x_angvel_lt', linewidth=2)
    plt.plot(t, dataset_np[:,13], label='x_angvel_ls', linewidth=2)
    plt.plot(t, dataset_np[:,18], label='x_angvel_rt', linewidth=2)
    plt.plot(t, dataset_np[:,23], label='x_angvel_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 5)
    plt.plot(t, dataset_np[:, 4], label='x_angacc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 9], label='x_angacc_lt', linewidth=2)
    plt.plot(t, dataset_np[:,14], label='x_angacc_ls', linewidth=2)
    plt.plot(t, dataset_np[:,19], label='x_angacc_rt', linewidth=2)
    plt.plot(t, dataset_np[:,24], label='x_angacc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)  # 非阻塞显示
    results_dir = os.path.join(file_directory, "results_temp")
    os.makedirs(results_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    output_path = os.path.join(results_dir, data_png_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {os.path.abspath(output_path)}")
    plt.close(fig1)
    # 对每个通道应用滤波器
    # filtered_data = np.zeros_like(dataset_np)
    # for i in range(dataset_np.shape[1]):  # 对每个特征通道
    #     filtered_data[:, i] = butter_bandpass_filter(dataset_np[:, i],lowcut=0.5,highcut=20.0,fs=50)#fs: 采样频率(Hz) lowcut: 低截止频率(Hz) highcut: 高截止频率(Hz)
    
    # return filtered_data
    return dataset_np


def loaddata_preproc_v2(filepath,startpoint,endpoint,data_png_name):
    '''
    导入原始数据,添加数据的角加速度
    划分窗口,划分数据集721
    不打乱,不归一化
    5连杆生物力学模型: 躯干、左大腿、左小腿、右大腿、右小腿
    '''
    # 导入文件
    data_raw = read_bin_file_v2(filepath=filepath,startpoint=startpoint,endpoint=endpoint)
    file_directory = os.path.dirname(filepath)
    print(filepath)
    dimension = data_raw['sensor_IMU_rpy'].shape #(行 列)
    datarows = dimension[0] #行数
    # cols = dimension[1] #列数
    # print(f"data import shape:{rows},{cols}")  # 添加的打印语句
    # dimension = data_raw['sensor_IMU_rpy'].ndim # 维度
    # print(dimension) #2

    # 使用 Savitzky-Golay 滤波器平滑并求导（适合噪声数据）
    window_length, polyorder = 11, 3  # savgol_filter函数参数。Savitzky-Golay 滤波器平滑并求导（适合噪声数据）
    feature_num = 5*5
    dataset_np = np.zeros((datarows, feature_num))# rows行，列
    dataset_np[:,0] = data_raw['sensor_IMU_acc'][:,0]  # 躯干x轴加速度
    dataset_np[:,1] = data_raw['sensor_IMU_acc'][:,1]  # 躯干y轴加速度
    dataset_np[:,2] = data_raw['sensor_IMU_rpy'][:,0]  # 躯干x轴角度    #012 345 678 91011 121314
    dataset_np[:,3] = data_raw['sensor_IMU_gyro'][:,0]  # 躯干x轴角速度
    dataset_np[:,4] = savgol_filter(data_raw['sensor_IMU_gyro'][:,0], window_length, polyorder, deriv=1, delta=0.01)  # 躯干x轴角加速度

    dataset_np[:,5] = data_raw['sensor_IMU_acc'][:,3]  # 左大腿x轴加速度
    dataset_np[:,6] = data_raw['sensor_IMU_acc'][:,4]  # 左大腿y轴加速度
    dataset_np[:,7] = data_raw['sensor_IMU_rpy'][:,3]  # 左大腿x轴角度    #012 345 678 91011 121314
    dataset_np[:,8] = data_raw['sensor_IMU_gyro'][:,3]  # 左大腿x轴角速度
    dataset_np[:,9] = savgol_filter(data_raw['sensor_IMU_gyro'][:,3], window_length, polyorder, deriv=1, delta=0.01)  # 左大腿x轴角加速度

    dataset_np[:,10] = data_raw['sensor_IMU_acc'][:,6]  # 左小腿x轴加速度
    dataset_np[:,11] = data_raw['sensor_IMU_acc'][:,7]  # 左小腿y轴加速度
    dataset_np[:,12] = data_raw['sensor_IMU_rpy'][:,6]  # 左小腿x轴角度    #012 345 678 91011 121314
    dataset_np[:,13] = data_raw['sensor_IMU_gyro'][:,6]  # 左小腿x轴角速度
    dataset_np[:,14] = savgol_filter(data_raw['sensor_IMU_gyro'][:,6], window_length, polyorder, deriv=1, delta=0.01)  # 左小腿x轴角加速度

    dataset_np[:,15] = data_raw['sensor_IMU_acc'][:,9]  # 右大腿x轴加速度
    dataset_np[:,16] = data_raw['sensor_IMU_acc'][:,10]  # 右大腿y轴加速度
    dataset_np[:,17] = data_raw['sensor_IMU_rpy'][:,9]  # 右大腿x轴角度   
    dataset_np[:,18] = data_raw['sensor_IMU_gyro'][:,9]  # 右大腿x轴角速度
    dataset_np[:,19] = savgol_filter(data_raw['sensor_IMU_gyro'][:,9], window_length, polyorder, deriv=1, delta=0.01)  # 右大腿x轴角加速度

    dataset_np[:,20] = data_raw['sensor_IMU_acc'][:,12]  # 右小腿x轴加速度
    dataset_np[:,21] = data_raw['sensor_IMU_acc'][:,13]  # 右小腿y轴加速度
    dataset_np[:,22] = data_raw['sensor_IMU_rpy'][:,12]  # 右小腿x轴角度   
    dataset_np[:,23] = data_raw['sensor_IMU_gyro'][:,12]  # 右小腿x轴角速度
    dataset_np[:,24] = savgol_filter(data_raw['sensor_IMU_gyro'][:,12], window_length, polyorder, deriv=1, delta=0.01)  # 右小腿x轴角加速度

    # 绘制结果,存盘
    fig1 = plt.figure(figsize=(12,8))
    t = np.linspace(0, datarows, datarows, endpoint=False)# 即 t = [0, 1, 2, ..., datarows-1]
    plt.subplot(5, 1, 1)
    plt.plot(t, dataset_np[:, 0], label='x_acc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 5], label='x_acc_lt', linewidth=2)
    plt.plot(t, dataset_np[:,10], label='x_acc_ls', linewidth=2)
    plt.plot(t, dataset_np[:,15], label='x_acc_rt', linewidth=2)
    plt.plot(t, dataset_np[:,20], label='x_acc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 2)
    plt.plot(t, dataset_np[:, 1], label='y_acc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 6], label='y_acc_lt', linewidth=2)
    plt.plot(t, dataset_np[:,11], label='y_acc_ls', linewidth=2)
    plt.plot(t, dataset_np[:,16], label='y_acc_rt', linewidth=2)
    plt.plot(t, dataset_np[:,21], label='y_acc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 3)
    plt.plot(t, dataset_np[:, 2], label='x_ang_b', linewidth=2)
    plt.plot(t, dataset_np[:, 7], label='x_ang_lt', linewidth=2)
    plt.plot(t, dataset_np[:,12], label='x_ang_ls', linewidth=2)
    plt.plot(t, dataset_np[:,17], label='x_ang_rt', linewidth=2)
    plt.plot(t, dataset_np[:,22], label='x_ang_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 4)
    plt.plot(t, dataset_np[:, 3], label='x_angvel_b', linewidth=2)
    plt.plot(t, dataset_np[:, 8], label='x_angvel_lt', linewidth=2)
    plt.plot(t, dataset_np[:,13], label='x_angvel_ls', linewidth=2)
    plt.plot(t, dataset_np[:,18], label='x_angvel_rt', linewidth=2)
    plt.plot(t, dataset_np[:,23], label='x_angvel_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 5)
    plt.plot(t, dataset_np[:, 4], label='x_angacc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 9], label='x_angacc_lt', linewidth=2)
    plt.plot(t, dataset_np[:,14], label='x_angacc_ls', linewidth=2)
    plt.plot(t, dataset_np[:,19], label='x_angacc_rt', linewidth=2)
    plt.plot(t, dataset_np[:,24], label='x_angacc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show(block=False)  # 非阻塞显示
    results_dir = os.path.join(file_directory, "results_temp")
    os.makedirs(results_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    output_path = os.path.join(results_dir, data_png_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {os.path.abspath(output_path)}")
    plt.close(fig1)
    # 对每个通道应用滤波器
    # filtered_data = np.zeros_like(dataset_np)
    # for i in range(dataset_np.shape[1]):  # 对每个特征通道
    #     filtered_data[:, i] = butter_bandpass_filter(dataset_np[:, i],lowcut=0.5,highcut=20.0,fs=50)#fs: 采样频率(Hz) lowcut: 低截止频率(Hz) highcut: 高截止频率(Hz)
    
    # return filtered_data
    return dataset_np


def loaddata_preproc_v3(filepath,startpoint,endpoint,data_png_name):
    '''
    导入原始数据,添加数据的角加速度
    划分窗口,划分数据集721
    不打乱,不归一化
    5连杆生物力学模型: 躯干、左大腿、左小腿、右大腿、右小腿
    计算物理意义下的髋关节角度（屈区为正）、膝关节角度（伸直为正）
    IMU安装,竖直安装
    '''
    # 导入文件
    data_raw = read_bin_file_v2(filepath=filepath,startpoint=startpoint,endpoint=endpoint)
    file_directory = os.path.dirname(filepath)
    print(filepath)
    dimension = data_raw['sensor_IMU_rpy'].shape #(行 列)
    datarows = dimension[0] #行数
    # cols = dimension[1] #列数
    # print(f"data import shape:{rows},{cols}")  # 添加的打印语句
    # dimension = data_raw['sensor_IMU_rpy'].ndim # 维度
    # print(dimension) #2

    # 使用 Savitzky-Golay 滤波器平滑并求导（适合噪声数据）
    window_length, polyorder = 11, 3  # savgol_filter函数参数。Savitzky-Golay 滤波器平滑并求导（适合噪声数据）
    feature_num = 5*5
    dataset_np = np.zeros((datarows, feature_num))# rows行，列
    dataset_np[:,0] = data_raw['sensor_IMU_acc'][:,0]  # 躯干x轴加速度
    dataset_np[:,1] = data_raw['sensor_IMU_acc'][:,3]  # 左大腿x轴加速度
    dataset_np[:,2] = data_raw['sensor_IMU_acc'][:,6]  # 左小腿x轴加速度
    dataset_np[:,3] = data_raw['sensor_IMU_acc'][:,9]  # 右大腿x轴加速度
    dataset_np[:,4] = data_raw['sensor_IMU_acc'][:,12]  # 右小腿x轴加速度

    dataset_np[:,5] = data_raw['sensor_IMU_acc'][:,1]  # 躯干y轴加速度
    dataset_np[:,6] = data_raw['sensor_IMU_acc'][:,4]  # 左大腿y轴加速度
    dataset_np[:,7] = data_raw['sensor_IMU_acc'][:,7]  # 左小腿y轴加速度
    dataset_np[:,8] = data_raw['sensor_IMU_acc'][:,10]  # 右大腿y轴加速度
    dataset_np[:,9] = data_raw['sensor_IMU_acc'][:,13]  # 右小腿y轴加速度

    dataset_np[:,10] = -data_raw['sensor_IMU_rpy'][:,0]  # 躯干x轴角度    #012 345 678 91011 121314
    dataset_np[:,11] = -data_raw['sensor_IMU_rpy'][:,3]  # 左大腿x轴髋关节角度    #012 345 678 91011 121314
    dataset_np[:,12] = -data_raw['sensor_IMU_rpy'][:,6] - (dataset_np[:,11]) # 左小腿x轴膝关节角度    #012 345 678 91011 121314
    dataset_np[:,13] = -data_raw['sensor_IMU_rpy'][:,9]  # 右大腿x轴髋关节角度
    dataset_np[:,14] = -data_raw['sensor_IMU_rpy'][:,12] - (dataset_np[:,13]) # 右小腿x轴膝关节角度

    dataset_np[:,15] = -data_raw['sensor_IMU_gyro'][:,0]  # 躯干x轴角速度
    dataset_np[:,16] = -data_raw['sensor_IMU_gyro'][:,3]  # 左大腿x轴角速度
    dataset_np[:,17] = -data_raw['sensor_IMU_gyro'][:,6]  # 左小腿x轴角速度
    dataset_np[:,18] = -data_raw['sensor_IMU_gyro'][:,9]  # 右大腿x轴角速度
    dataset_np[:,19] = -data_raw['sensor_IMU_gyro'][:,12]  # 右小腿x轴角速度

    dataset_np[:,20] = savgol_filter(data_raw['sensor_IMU_gyro'][:,0], window_length, polyorder, deriv=1, delta=0.01)  # 躯干x轴角加速度
    dataset_np[:,21] = savgol_filter(data_raw['sensor_IMU_gyro'][:,3], window_length, polyorder, deriv=1, delta=0.01)  # 左大腿x轴角加速度
    dataset_np[:,22] = savgol_filter(data_raw['sensor_IMU_gyro'][:,6], window_length, polyorder, deriv=1, delta=0.01)  # 左小腿x轴角加速度
    dataset_np[:,23] = savgol_filter(data_raw['sensor_IMU_gyro'][:,9], window_length, polyorder, deriv=1, delta=0.01)  # 右大腿x轴角加速度
    dataset_np[:,24] = savgol_filter(data_raw['sensor_IMU_gyro'][:,12], window_length, polyorder, deriv=1, delta=0.01)  # 右小腿x轴角加速度

    # 绘制结果,存盘
    fig1 = plt.figure(figsize=(12,8))
    t = np.linspace(0, datarows, datarows, endpoint=False)# 即 t = [0, 1, 2, ..., datarows-1]
    plt.subplot(5, 1, 1)
    plt.plot(t, dataset_np[:, 0], label='x_acc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 1], label='x_acc_lt', linewidth=2)
    plt.plot(t, dataset_np[:, 2], label='x_acc_ls', linewidth=2)
    plt.plot(t, dataset_np[:, 3], label='x_acc_rt', linewidth=2)
    plt.plot(t, dataset_np[:, 4], label='x_acc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 2)
    plt.plot(t, dataset_np[:, 5], label='y_acc_b', linewidth=2)
    plt.plot(t, dataset_np[:, 6], label='y_acc_lt', linewidth=2)
    plt.plot(t, dataset_np[:, 7], label='y_acc_ls', linewidth=2)
    plt.plot(t, dataset_np[:, 8], label='y_acc_rt', linewidth=2)
    plt.plot(t, dataset_np[:, 9], label='y_acc_rs', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 3)
    plt.plot(t, dataset_np[:,10], label='x_ang_b', linewidth=2)
    plt.plot(t, dataset_np[:,11], label='x_ang_lhip', linewidth=2)
    plt.plot(t, dataset_np[:,12], label='x_ang_lknee', linewidth=2)
    plt.plot(t, dataset_np[:,13], label='x_ang_rhip', linewidth=2)
    plt.plot(t, dataset_np[:,14], label='x_ang_rknee', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 4)
    plt.plot(t, dataset_np[:,15], label='x_angvel_b', linewidth=2)
    plt.plot(t, dataset_np[:,16], label='x_angvel_lhip', linewidth=2)
    plt.plot(t, dataset_np[:,17], label='x_angvel_lknee', linewidth=2)
    plt.plot(t, dataset_np[:,18], label='x_angvel_rhip', linewidth=2)
    plt.plot(t, dataset_np[:,19], label='x_angvel_rknee', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 5)
    plt.plot(t, dataset_np[:,20], label='x_angacc_b', linewidth=2)
    plt.plot(t, dataset_np[:,21], label='x_angacc_lhip', linewidth=2)
    plt.plot(t, dataset_np[:,22], label='x_angacc_lknee', linewidth=2)
    plt.plot(t, dataset_np[:,23], label='x_angacc_rhip', linewidth=2)
    plt.plot(t, dataset_np[:,24], label='x_angacc_rknee', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show(block=False)  # 非阻塞显示
    results_dir = os.path.join(file_directory, "results_temp")
    os.makedirs(results_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    output_path = os.path.join(results_dir, data_png_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {os.path.abspath(output_path)}")
    plt.close(fig1)
    # 对每个通道应用滤波器
    # filtered_data = np.zeros_like(dataset_np)
    # for i in range(dataset_np.shape[1]):  # 对每个特征通道
    #     filtered_data[:, i] = butter_bandpass_filter(dataset_np[:, i],lowcut=0.5,highcut=20.0,fs=50)#fs: 采样频率(Hz) lowcut: 低截止频率(Hz) highcut: 高截止频率(Hz)
    
    # return filtered_data
    return dataset_np
# end loaddata_preproc_v3


def extract_window_data(dataset,label,window_size=20,step_size=1):

    datarows = dataset.shape[0]
    #从数据集中提取时间序列窗口数据
    num_windows = (datarows - window_size) // step_size + 1
    all_windowed_data_list = []
    all_window_labels_list = []
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = dataset[start_idx:end_idx, :]
        all_windowed_data_list.append(window)
        all_window_labels_list.append(label)

    # 转换为numpy数组
    all_windowed_data_np = np.array(all_windowed_data_list)
    all_window_labels_np = np.array(all_window_labels_list)

    # 划分数据集
    train_data_rows = int(num_windows*0.7)
    validate_data_rows = int(num_windows*0.2)
    test_data_rows = int(num_windows*0.1)

    train_data  = all_windowed_data_np[:train_data_rows]#Python切片是左闭右开,且3D数据切片不需要指定所有维度
    train_label = all_window_labels_np[:train_data_rows]

    validate_data  = all_windowed_data_np[train_data_rows:train_data_rows+validate_data_rows]
    validate_label = all_window_labels_np[train_data_rows:train_data_rows+validate_data_rows]

    test_data  = all_windowed_data_np[train_data_rows+validate_data_rows:train_data_rows+validate_data_rows+test_data_rows]
    test_label = all_window_labels_np[train_data_rows+validate_data_rows:train_data_rows+validate_data_rows+test_data_rows]

    print(f"{label}数据集划分结果:")
    print(f"训练集: {len(train_data)},{train_data.shape} 样本")
    print(f"验证集: {len(validate_data)},{validate_data.shape} 样本")
    print(f"测试集: {len(test_data)},{test_data.shape} 样本")

    return train_data,train_label,validate_data,validate_label,test_data,test_label


def loaddata_preproc_netdata(filepath,window_size=20,step_size=1):
    '''
    导入原始数据,在线数据集ENABL3S,添加数据的角加速度
    划分窗口,划分数据集721
    不打乱,不归一化
    5连杆生物力学模型: 躯干、左大腿、左小腿、右大腿、右小腿
    '''
    # 获取当前目录下的所有文件和子目录名并按字母顺序排序
    print(f"打开目录: {filepath}")
    files_dict = {}
    all_files = sorted(os.listdir(filepath))
    files = [f for f in all_files if os.path.isfile(os.path.join(filepath, f))]# 过滤出文件（排除目录）
    # 初始化用于存储所有数据的列表
    all_train_data = []
    all_train_labels = []
    all_validate_data = []
    all_validate_labels = []
    all_test_data = []
    all_test_labels = []
    train_data_nowindow = []
    for file in files:
        df = pd.read_csv(f'{filepath}/{file}')
        unique_labels = df.iloc[:, 52].unique()# 获取所有唯一标签值
        print(f"读取文件:{file}, 发现标签:{sorted(unique_labels)}")
        grouped = df.groupby(df.columns[52])# 按照label值分组
        separated_data = {# 为当前文件创建存储结构
            'labels': unique_labels,
            'group_data_np': {},
            'group_counts': {},
        }
        for label,group in grouped:
            separated_data['group_data_np'][label] = group.values# 转为NumPy数组
            separated_data['group_counts'][label] = len(group)
            dimension = separated_data['group_data_np'][label].shape
            datarows = dimension[0] #行数
            cols = dimension[1] #列数
            # print("数组形状：", dimension)  # (4000, 61)
            # 使用 Savitzky-Golay 滤波器平滑并求导（适合噪声数据）
            window_length, polyorder = 11, 3  # savgol_filter函数参数。Savitzky-Golay 滤波器平滑并求导（适合噪声数据）
            feature_num = 5*5
            dataset_np = np.zeros((datarows, feature_num))# rows行，列
            dataset_np[:,0] = separated_data['group_data_np'][label][:,24]  # 躯干x轴加速度，朝前，对应imux轴
            dataset_np[:,1] = separated_data['group_data_np'][label][:,25]  # 躯干y轴加速度，朝上，对应imuy轴
            dataset_np[:,2] = separated_data['group_data_np'][label][:,28]  # 躯干x轴角度    
            dataset_np[:,3] = separated_data['group_data_np'][label][:,28]  # 躯干x轴角速度，人体u关节旋转轴，对应imuz轴
            dataset_np[:,4] = savgol_filter(separated_data['group_data_np'][label][:,28], window_length, polyorder, deriv=1, delta=0.01)  # 躯干x轴角加速度

            dataset_np[:,5] = separated_data['group_data_np'][label][:,20]  # 左大腿x轴加速度，朝前，对应imuz轴
            dataset_np[:,6] = separated_data['group_data_np'][label][:,18]  # 左大腿y轴加速度，朝上，对应imux轴
            dataset_np[:,7] = separated_data['group_data_np'][label][:,21]  # 左大腿x轴角度   ，关节旋转轴，人体矢状面
            dataset_np[:,8] = separated_data['group_data_np'][label][:,21]  # 左大腿x轴角速度，人体u关节旋转轴，对应imuy轴
            dataset_np[:,9] = savgol_filter(separated_data['group_data_np'][label][:,21], window_length, polyorder, deriv=1, delta=0.01)  # 左大腿x轴角加速度

            dataset_np[:,10] = separated_data['group_data_np'][label][:,14]  # 左小腿x轴加速度，朝前，对应imuz轴
            dataset_np[:,11] = separated_data['group_data_np'][label][:,12]  # 左小腿y轴加速度，朝上，对应imux轴
            dataset_np[:,12] = separated_data['group_data_np'][label][:,15]  # 左小腿x轴角度    
            dataset_np[:,13] = separated_data['group_data_np'][label][:,15]  # 左小腿x轴角速度，人体u关节旋转轴，对应imuy轴
            dataset_np[:,14] = savgol_filter(separated_data['group_data_np'][label][:,15], window_length, polyorder, deriv=1, delta=0.01)  # 左小腿x轴角加速度

            dataset_np[:,15] = separated_data['group_data_np'][label][:,8]  # 右大腿x轴加速度，朝前，对应imuz轴
            dataset_np[:,16] = separated_data['group_data_np'][label][:,6]  # 右大腿y轴加速度，朝上，对应imux轴
            dataset_np[:,17] = separated_data['group_data_np'][label][:,9]  # 右大腿x轴角度   ，关节旋转轴，人体矢状面
            dataset_np[:,18] = separated_data['group_data_np'][label][:,9]  # 右大腿x轴角速度，人体u关节旋转轴，对应imuy轴
            dataset_np[:,19] = savgol_filter(separated_data['group_data_np'][label][:,9], window_length, polyorder, deriv=1, delta=0.01)  # 右大腿x轴角加速度

            dataset_np[:,20] = separated_data['group_data_np'][label][:,2]  # 右小腿x轴加速度，朝前，对应imuz轴
            dataset_np[:,21] = separated_data['group_data_np'][label][:,0]  # 右小腿y轴加速度，朝上，对应imux轴
            dataset_np[:,22] = separated_data['group_data_np'][label][:,3]  # 右小腿x轴角度   ，关节旋转轴，人体矢状面
            dataset_np[:,23] = separated_data['group_data_np'][label][:,3]  # 右小腿x轴角速度，人体u关节旋转轴，对应imuy轴
            dataset_np[:,24] = savgol_filter(separated_data['group_data_np'][label][:,3], window_length, polyorder, deriv=1, delta=0.01)  # 右小腿x轴角加速度

            train_data,train_label,validate_data,validate_label,test_data,test_label = extract_window_data(dataset_np,label,window_size=window_size,step_size=step_size)

            # 将当前标签的数据添加到总列表中
            train_data_nowindow.append(dataset_np[:math.floor(datarows*0.7),:])
            all_train_data.append(train_data)
            all_train_labels.append(train_label)
            all_validate_data.append(validate_data)
            all_validate_labels.append(validate_label)
            all_test_data.append(test_data)
            all_test_labels.append(test_label)
        
    # 使用np.vstack合并所有数据
    train_data_raw_nowindow = np.vstack(train_data_nowindow)#(1139870,25)
    train_data_raw = np.vstack(all_train_data)#(798387,20,25)
    train_label_raw = np.concatenate(all_train_labels)
    validate_data_raw = np.vstack(all_validate_data)
    validate_label_raw = np.concatenate(all_validate_labels)
    test_data_raw = np.vstack(all_test_data)
    test_label_raw = np.concatenate(all_test_labels)

    print(f"合并后无窗口划分数据维度：{train_data_raw_nowindow.shape}")
    print(f"合并后数据维度：{train_data_raw.shape}")

    #应该对训练集进行归一化
    #数据标准化，并标准化参数保存至文件
    #scaler = StandardScaler()  # StandardScaler (Z-score 标准化)
    scaler = MinMaxScaler(feature_range=(-1, 1))  # (归一化到 [-1,1])
    scaler.fit_transform(train_data_raw_nowindow)#计算均值、标准差等参数
    # print("均值:", scaler.mean_)
    # print("标准差:", scaler.scale_)  # 对于 StandardScaler
    # print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    # print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler
    # 保存最大值，最小值，缩放比例和偏移量
    # 保存每个特征均值，标准差，方差，已处理的样本总数

    # 应用标准化(划分窗口之后)
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw

    # files_dict[file] = separated_data
    

    # separated_data = {label: group for label, group in grouped}# 创建字典保存分组结果
    # np_data0 = separated_data['group_data'][0].values
    # np_data1 = separated_data['group_data'][1].values
    # np_data2 = separated_data[2].values
    # np_data3 = separated_data[3].values
    # np_data4 = separated_data[4].values

    # print(df.columns.tolist())  # 以列表形式输出所有列名
    # ['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az',  012
    #  'Right_Shank_Gy', 'Right_Shank_Gz', 'Right_Shank_Gx',  345
    #  'Right_Thigh_Ax', 'Right_Thigh_Ay', 'Right_Thigh_Az',  678
    #  'Right_Thigh_Gy', 'Right_Thigh_Gz', 'Right_Thigh_Gx',  91011
    #  'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az',     121314
    #  'Left_Shank_Gy', 'Left_Shank_Gz', 'Left_Shank_Gx',     151617
    #  'Left_Thigh_Ax', 'Left_Thigh_Ay', 'Left_Thigh_Az',     181920
    #  'Left_Thigh_Gy', 'Left_Thigh_Gz', 'Left_Thigh_Gx',     212223
    #  'Waist_Ax', 'Waist_Ay', 'Waist_Az',                    242526
    #  'Waist_Gy', 'Waist_Gz', 'Waist_Gx',                    272829
    #  'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL', 'Right_RF',  30-36
    #  'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST', 'Left_VL', 'Left_RF',         37-43
    #  'Right_Ankle', 'Right_Knee',                                        4445
    #  'Left_Ankle', 'Left_Knee',                                          4647
    #  'Right_Ankle_Velocity', 'Right_Knee_Velocity',                      4849
    #  'Left_Ankle_Velocity', 'Left_Knee_Velocity',                        5051
    #  'Mode',  0=Sitting, 6=Standing, 1=Level ground walking, 2=Ramp ascent, 3=Ramp descent, 4=Stair ascent, 5=Stair descent  52
    #  'Right_Heel_Contact', 'Right_Heel_Contact_Trigger', 'Right_Toe_Off', 'Right_Toe_Off_Trigger',  53545556
    #  'Left_Heel_Contact', 'Left_Heel_Contact_Trigger', 'Left_Toe_Off', 'Left_Toe_Off_Trigger']      57585960

    

    

    # # 绘制结果,存盘
    # fig1 = plt.figure(figsize=(12,8))
    # t = np.linspace(0, datarows, datarows, endpoint=False)# 即 t = [0, 1, 2, ..., datarows-1]
    # plt.subplot(5, 1, 1)
    # plt.plot(t, dataset_np[:, 0], label='x_acc_b', linewidth=2)
    # plt.plot(t, dataset_np[:, 5], label='x_acc_lt', linewidth=2)
    # plt.plot(t, dataset_np[:,10], label='x_acc_ls', linewidth=2)
    # plt.plot(t, dataset_np[:,15], label='x_acc_rt', linewidth=2)
    # plt.plot(t, dataset_np[:,20], label='x_acc_rs', linewidth=2)
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(5, 1, 2)
    # plt.plot(t, dataset_np[:, 1], label='y_acc_b', linewidth=2)
    # plt.plot(t, dataset_np[:, 6], label='y_acc_lt', linewidth=2)
    # plt.plot(t, dataset_np[:,11], label='y_acc_ls', linewidth=2)
    # plt.plot(t, dataset_np[:,16], label='y_acc_rt', linewidth=2)
    # plt.plot(t, dataset_np[:,21], label='y_acc_rs', linewidth=2)
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(5, 1, 3)
    # plt.plot(t, dataset_np[:, 2], label='x_ang_b', linewidth=2)
    # plt.plot(t, dataset_np[:, 7], label='x_ang_lt', linewidth=2)
    # plt.plot(t, dataset_np[:,12], label='x_ang_ls', linewidth=2)
    # plt.plot(t, dataset_np[:,17], label='x_ang_rt', linewidth=2)
    # plt.plot(t, dataset_np[:,22], label='x_ang_rs', linewidth=2)
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(5, 1, 4)
    # plt.plot(t, dataset_np[:, 3], label='x_angvel_b', linewidth=2)
    # plt.plot(t, dataset_np[:, 8], label='x_angvel_lt', linewidth=2)
    # plt.plot(t, dataset_np[:,13], label='x_angvel_ls', linewidth=2)
    # plt.plot(t, dataset_np[:,18], label='x_angvel_rt', linewidth=2)
    # plt.plot(t, dataset_np[:,23], label='x_angvel_rs', linewidth=2)
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(5, 1, 5)
    # plt.plot(t, dataset_np[:, 4], label='x_angacc_b', linewidth=2)
    # plt.plot(t, dataset_np[:, 9], label='x_angacc_lt', linewidth=2)
    # plt.plot(t, dataset_np[:,14], label='x_angacc_ls', linewidth=2)
    # plt.plot(t, dataset_np[:,19], label='x_angacc_rt', linewidth=2)
    # plt.plot(t, dataset_np[:,24], label='x_angacc_rs', linewidth=2)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show(block=False)  # 非阻塞显示
    # results_dir = os.path.join(filepath, "results_temp")
    # os.makedirs(results_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    # output_path = os.path.join(results_dir, data_png_name)
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"结果已保存至: {os.path.abspath(output_path)}")
    # plt.close(fig1)
    # # 对每个通道应用滤波器
    # # filtered_data = np.zeros_like(dataset_np)
    # # for i in range(dataset_np.shape[1]):  # 对每个特征通道
    # #     filtered_data[:, i] = butter_bandpass_filter(dataset_np[:, i],lowcut=0.5,highcut=20.0,fs=50)#fs: 采样频率(Hz) lowcut: 低截止频率(Hz) highcut: 高截止频率(Hz)
    
    # # return filtered_data
    # return dataset_np,dataset_np_label


if __name__ == "__main__":
    results = test_butter_bandpass_filter()
    print("\n程序继续执行...")
    print("可以进行其他处理，图像窗口保持打开状态")
    
    # 这里可以添加其他处理代码
    # 例如计算滤波前后信号的RMS值
    original_rms = np.sqrt(np.mean(results['original_signal']**2))
    filtered_rms = np.sqrt(np.mean(results['filtered_signal']**2))
    print(f"\n原始信号RMS值: {original_rms:.4f}")
    print(f"滤波后信号RMS值: {filtered_rms:.4f}")