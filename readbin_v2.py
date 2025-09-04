import numpy as np
import struct
from tkinter import filedialog
from tkinter import Tk
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 读取文件，*.bin

def safe_read(fd, size):
    """安全读取指定字节数，不足时抛出异常"""
    data = fd.read(size)
    if len(data) < size:
        raise IOError(f"需要 {size} 字节，但只读取到 {len(data)} 字节")
    return data

def read_bin_file_v2(filepath=None,startpoint=0,endpoint=None):

    file_path = filepath

    # 打开文件选择对话框
    if filepath is None:
        root = Tk()# 创建隐藏的Tkinter窗口用于文件选择
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Binary files", "*.bin")])
        if not file_path:
            print("未选择文件")
            return None

    # print(f"正在读取文件: {os.path.basename(file_path)}")

    try:
        file_size = os.path.getsize(file_path)
        # print(f"文件大小: {file_size} 字节")
        
        # 计算完整帧数（每帧512字节）
        frame_size = 512
        total_frames = file_size // frame_size #向下取整
        remaining_bytes = file_size % frame_size
        # print(f"文件总帧数: {total_frames}")
        
        if remaining_bytes != 0:
            warnings.warn(f"文件包含 {remaining_bytes} 字节不完整数据，将被忽略")
        
        # 预定义数据结构
        data_structures = {
            'DATA_frame': np.arange(total_frames),
            'DATA_time': np.arange(total_frames) * 0.05, # 20Hz
            # 传感器数据
            'sensor_IMU_acc': np.zeros((total_frames, 15)),  # 5x3 （back\left thigh\left shank\right thigh\right shank）
            'sensor_IMU_gyro': np.zeros((total_frames, 15)),  # 5x3
            'sensor_IMU_rpy': np.zeros((total_frames, 15)),  # 5x3
            'sensor_IMU_rpy_foot': np.zeros((total_frames, 6)),  # 2x3
            'sensor_IMU_q4': np.zeros((total_frames, 20)),   # 5x4
            'sensor_mpos': np.zeros((total_frames, 4)),  #电机位置(left hip\left knee\right hip\right knee)
            'sensor_mspeed': np.zeros((total_frames, 4)),
            'sensor_mcurrent': np.zeros((total_frames, 4)),
            'sensor_mIqSet': np.zeros((total_frames, 4)),
            'sensor_adc': np.zeros((total_frames, 8), dtype=np.uint16),
            'sensor_io': np.zeros((total_frames, 1), dtype=np.uint16),
            'sensor_left_GaitPhase': np.zeros((total_frames, 1), dtype=np.uint8),
            'sensor_right_GaitPhase': np.zeros((total_frames, 1), dtype=np.uint8),
            'sensor_timestamp': np.zeros((total_frames, 2), dtype=np.uint16),#左右脚传感器时间(left\right)
            'sensor_init_IMU_rpy': np.zeros((total_frames, 15)),  # 5x3
            'sensor_init_IMU_rpy_foot': np.zeros((total_frames, 6)),  # 2x3
            'sensor_init_mpos': np.zeros((total_frames, 4)),  #电机位置(left hip\left knee\right hip\right knee)
            'sensor_init_adc': np.zeros((total_frames, 8), dtype=np.uint16),
            'sensor_isvalid': np.zeros((total_frames, 6), dtype=np.uint16),#(io\adc\acc\q4\rpy\gyro)
            'sensor_vbus': np.zeros((total_frames, 1)),
            'sensor_torqHipMotor': np.zeros((total_frames, 1)),
            'sensor_torqKneeMotor': np.zeros((total_frames, 1)),
        }

        with open(file_path, 'rb') as fidin:
            for frame_idx in range(total_frames):
                try:
                    # 读取传感器数据 (512Bits)  exo_sensor_t sensor
                    data_structures['sensor_IMU_acc'][frame_idx] = struct.unpack('15f', safe_read(fidin, 60))#IMU加速度 (15f=60Bits)
                    data_structures['sensor_IMU_gyro'][frame_idx] = struct.unpack('15f', safe_read(fidin, 60))#IMU角速度 (15f=60Bits)
                    data_structures['sensor_IMU_rpy'][frame_idx] = struct.unpack('15f', safe_read(fidin, 60))#IMU角度 (15f=60Bits)
                    data_structures['sensor_IMU_rpy_foot'][frame_idx] = struct.unpack('6f', safe_read(fidin, 24))#脚IMU角度 (5f=24Bits)
                    data_structures['sensor_IMU_q4'][frame_idx]  = struct.unpack('20f', safe_read(fidin, 80))#四元数 (20f=80Bits)
                    data_structures['sensor_mpos'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机位置(4f=16Bits)
                    data_structures['sensor_mspeed'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机速度(15f=60Bits)
                    data_structures['sensor_mcurrent'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机电流(15f=60Bits)
                    data_structures['sensor_mIqSet'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机电流(15f=60Bits)
                    data_structures['sensor_adc'][frame_idx] = struct.unpack('8H', safe_read(fidin, 16))#adc(8H=16Bits) #H无符号短整数(unsigned short)
                    data_structures['sensor_io'][frame_idx] = struct.unpack('1H', safe_read(fidin, 2))#io(1H=2Bits)
                    data_structures['sensor_left_GaitPhase'][frame_idx] = struct.unpack('1B', safe_read(fidin, 1))#步态相位(1B=1Bits)
                    data_structures['sensor_right_GaitPhase'][frame_idx] = struct.unpack('1B', safe_read(fidin, 1))#步态相位(1B=1Bits)
                    data_structures['sensor_timestamp'][frame_idx] = struct.unpack('2H', safe_read(fidin, 4))#左右脚时间戳(2H=4Bits)
                    data_structures['sensor_init_IMU_rpy'][frame_idx] = struct.unpack('15f', safe_read(fidin, 60))#初始IMU角度 (15f=60Bits)
                    data_structures['sensor_init_IMU_rpy_foot'][frame_idx] = struct.unpack('6f', safe_read(fidin, 24))#初始脚IMU角度 (5f=24Bits)
                    data_structures['sensor_init_mpos'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#初始电机位置(4f=16Bits)
                    data_structures['sensor_init_adc'][frame_idx] = struct.unpack('8H', safe_read(fidin, 16))#初始adc(8H=16Bits) #H无符号短整数(unsigned short)
                    data_structures['sensor_isvalid'][frame_idx] = struct.unpack('6H', safe_read(fidin, 12))#有效判断(6H=12Bits)
                    data_structures['sensor_vbus'][frame_idx] = struct.unpack('1f', safe_read(fidin, 4))#电池电压(1f=4Bits)
                    data_structures['sensor_torqHipMotor'][frame_idx] = struct.unpack('1f', safe_read(fidin, 4))#电机力矩(1f=4Bits)
                    data_structures['sensor_torqKneeMotor'][frame_idx] = struct.unpack('1f', safe_read(fidin, 4))#电机力矩(1f=4Bits)
                except (IOError, struct.error) as e:
                    print(f"第 {frame_idx} 帧读取错误: {str(e)}")
                    break
        print(f"读取 {frame_idx+1}/{total_frames}帧数据")
        if endpoint is None:
            endpoint = total_frames
        if startpoint > endpoint:
            print('坐标点选择错误！')
            return
        
        # 返回切片数据
        result = {}
        for key in data_structures:
            if key == 'DATA_time' or 'DATA_frame':  # 特殊处理时间序列
                result[key] = data_structures[key][startpoint:endpoint]
            else:
                result[key] = data_structures[key][startpoint:endpoint, ...]
        return result
    
    except Exception as e:
        print(f"文件处理错误: {str(e)}")
        return None



def plot_sensor_data_v2(data):
    if data is None:
        print("无有效数据可绘制")
        return
    
    time = data['DATA_time']
    
    # 创建一个大图，包含多个子图
    plt.figure(figsize=(15, 20))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # 1. 绘制IMU加速度数据
    ax1 = plt.subplot(gs[0, 0])
    for i in range(5):  # 5个IMU
        ax1.plot(time, data['sensor_IMU_acc'][:, i*3], label=f'IMU{i+1}acc-X')
        ax1.plot(time, data['sensor_IMU_acc'][:, i*3+1], label=f'IMU{i+1}acc-Y')
        ax1.plot(time, data['sensor_IMU_acc'][:, i*3+2], label=f'IMU{i+1}acc-Z')
    # ax1.set_title('IMU acc')
    # ax1.set_xlabel('time (s)')
    ax1.set_ylabel('acc (m/s²)')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True)
    
    # 2. 绘制IMU角速度数据
    ax2 = plt.subplot(gs[0, 1])
    for i in range(5):  # 5个IMU
        ax2.plot(time, data['sensor_IMU_gyro'][:, i*3], label=f'IMU{i+1}angVel-X')
        ax2.plot(time, data['sensor_IMU_gyro'][:, i*3+1], label=f'IMU{i+1}angVel-Y')
        ax2.plot(time, data['sensor_IMU_gyro'][:, i*3+2], label=f'IMU{i+1}angVel-Z')
    # ax2.set_title('IMU angleVel')
    # ax2.set_xlabel('time (s)')
    ax2.set_ylabel('angVel (du/s)')
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True)
    
    # 3. 绘制关节角度
    ax3 = plt.subplot(gs[1, 0])
    joint_names = ['angBack', 'angThighL', 'angCalfL', 'angThighR', 'angCalfR']
    # joint_names = ['angBack', 'angHipL', 'angKneeL', 'angAnkleL', 'angHipR', 'angKneeR', 'angAnkleR']
    for i in range(5):
        ax3.plot(time, data['sensor_IMU_rpy'][:, i*3], label=f'IMU{i+1}ang-X')
        ax3.plot(time, data['sensor_IMU_rpy'][:, i*3+1], label=f'IMU{i+1}ang-Y')
        ax3.plot(time, data['sensor_IMU_rpy'][:, i*3+2], label=f'IMU{i+1}ang-Z')
    # ax3.set_title('joint angle')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('ang (du)')
    ax3.legend()
    ax3.grid(True)

    # 4. 绘制足底压力
    ax4 = plt.subplot(gs[1, 1])
    adc_names = ['UpL','DownL','FrontL','BackL','UpR','DownR','FrontR','BackR']
    for i in range(8):
        ax4.plot(time, data['sensor_adc'][:, i], label=adc_names[i])
    ax4.set_title('foot pressure')
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('pressure (mv)')
    ax4.legend()
    ax4.grid(True)
    
    # # 4. 绘制电机电流
    # ax4 = plt.subplot(gs[1, 1])
    # motor_names = ['mposHipL', 'mposKneeL', 'mposHipR', 'mposKneeR']
    # for i in range(4):
    #     ax4.plot(time, data['sensor_mcurrent'][:, i], label=motor_names[i])
    # # ax4.set_title('motor current')
    # ax4.set_xlabel('time (s)')
    # ax4.set_ylabel('current (A)')
    # ax4.legend()
    # ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = read_bin_file_v2(startpoint=1,endpoint=None)
    if data is not None:
        print("\n数据摘要:")
        print(f"总帧数: {len(data['DATA_frame'])}")
        print("第一帧加速度均值:", np.mean(data['sensor_IMU_acc'][0]))
        print("第一帧四元数:", data['sensor_IMU_q4'][0][:4])  # 显示前4个值
        #print(bin_data[:100].hex(' '))# 十六进制
    # 绘制数据
    plot_sensor_data_v2(data)


