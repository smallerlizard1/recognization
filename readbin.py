import numpy as np
import struct
from tkinter import filedialog
from tkinter import Tk
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 读取bin文件，msg_bt_*.bin

def safe_read(fd, size):
    """安全读取指定字节数，不足时抛出异常"""
    data = fd.read(size)
    if len(data) < size:
        raise IOError(f"需要 {size} 字节，但只读取到 {len(data)} 字节")
    return data

def read_bin_file(filepath=None,startpoint=0,endpoint=None):

    file_path = filepath

    # 打开文件选择对话框
    if filepath is None:
        root = Tk()# 创建隐藏的Tkinter窗口用于文件选择
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Binary files", "*.bin")])
        if not file_path:
            print("未选择文件")
            return None

    print(f"正在读取文件: {os.path.basename(file_path)}")

    try:
        file_size = os.path.getsize(file_path)
        print(f"文件大小: {file_size} 字节")
        
        # 计算完整帧数（每帧512字节）
        frame_size = 512
        total_frames = file_size // frame_size
        remaining_bytes = file_size % frame_size
        print(f"文件总帧数: {total_frames}")
        
        if remaining_bytes != 0:
            warnings.warn(f"文件包含 {remaining_bytes} 字节不完整数据，将被忽略")
        
        # 预定义数据结构
        data_structures = {
            'HEADER': np.zeros((total_frames, 4), dtype=np.uint8),
            'DATA_frame': np.arange(total_frames),
            'DATA_time': np.arange(total_frames) * 0.02,
            # 传感器数据
            'sensor_IMU_acc': np.zeros((total_frames, 15)),  # 5x3 （back\left thigh\left shank\right thigh\right shank）
            'sensor_IMU_q4': np.zeros((total_frames, 20)),   # 5x4
            'sensor_IMU_rpy': np.zeros((total_frames, 15)),  # 5x3
            'sensor_IMU_rpy_foot': np.zeros((total_frames, 6)),  # 2x3
            'sensor_IMU_gyro': np.zeros((total_frames, 15)),  # 5x3
            'sensor_mpos': np.zeros((total_frames, 4)),  #电机位置(left hip\left knee\right hip\right knee)
            'sensor_mspeed': np.zeros((total_frames, 4)),
            'sensor_mcurrent': np.zeros((total_frames, 4)),
            'sensor_adc': np.zeros((total_frames, 8), dtype=np.uint16),
            'sensor_io': np.zeros((total_frames, 1), dtype=np.uint16),
            'sensor_reserved': np.zeros((total_frames, 1), dtype=np.uint16),
            'sensor_timestamp': np.zeros((total_frames, 2), dtype=np.uint16),#左右脚传感器时间(left\right)
            'sensor_isvalid': np.zeros((total_frames, 6), dtype=np.uint16),#(io\adc\acc\q4\rpy\gyro)
            # 状态数据
            'status_angle': np.zeros((total_frames, 7)), # 关节角度
            'status_force': np.zeros((total_frames, 8)), # 关节力
            'status_motortorque': np.zeros((total_frames, 4)),
            'status_leftratio': np.zeros((total_frames, 1)),
            'status_leftconfidence': np.zeros((total_frames, 1)),
            'status_rightratio': np.zeros((total_frames, 1)),
            'status_rightconfidence': np.zeros((total_frames, 1)),
            'status_motionratio': np.zeros((total_frames, 1)),
            'status_motionconfidence': np.zeros((total_frames, 1)),
            'status_exostate': np.zeros((total_frames, 1), dtype=np.uint8),
            'status_leftstate': np.zeros((total_frames, 1), dtype=np.uint8),
            'status_rightstate': np.zeros((total_frames, 1), dtype=np.uint8),
            'status_motionstate': np.zeros((total_frames, 1), dtype=np.uint8),
            'status_errorcode1': np.zeros((total_frames, 1), dtype=np.uint16),
            'status_isvalid1': np.zeros((total_frames, 1), dtype=np.uint16),
            'status_debug_time': np.zeros((total_frames, 1), dtype=np.uint32),
            'status_debug_motionstate': np.zeros((total_frames, 1), dtype=np.uint8),
            'status_debug_robotversion': np.zeros((total_frames, 1), dtype=np.uint16),
            'status_debug_flag': np.zeros((total_frames, 1), dtype=np.uint8),
            # 控制数据
            'control_timestamp': np.zeros((total_frames, 1), dtype=np.uint16),
            'control_period': np.zeros((total_frames, 1), dtype=np.uint16),
            'control_timeused': np.zeros((total_frames, 1), dtype=np.uint16),
            'control_recordnum': np.zeros((total_frames, 1), dtype=np.uint16)
        }

        with open(file_path, 'rb') as fidin:
            for frame_idx in range(total_frames):
                try:
                    # 1. 读取帧头 (4Bits)
                    data_structures['HEADER'][frame_idx] = struct.unpack('4B', safe_read(fidin, 4))
                    
                    # 2. 读取传感器数据 (368Bits)  exo_sensor_t sensor
                    data_structures['sensor_IMU_acc'][frame_idx] = struct.unpack('15f', safe_read(fidin, 60))#IMU加速度 (15f=60Bits)                   
                    data_structures['sensor_IMU_q4'][frame_idx]  = struct.unpack('20f', safe_read(fidin, 80))#四元数 (20f=80Bits)
                    data_structures['sensor_IMU_rpy'][frame_idx] = struct.unpack('15f', safe_read(fidin, 60))#IMU角度 (15f=60Bits)
                    data_structures['sensor_IMU_rpy_foot'][frame_idx] = struct.unpack('6f', safe_read(fidin, 24))#脚IMU角度 (5f=24Bits)
                    data_structures['sensor_IMU_gyro'][frame_idx] = struct.unpack('15f', safe_read(fidin, 60))#IMU角速度 (15f=60Bits)
                    data_structures['sensor_mpos'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机位置(4f=16Bits)
                    data_structures['sensor_mspeed'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机速度(15f=60Bits)
                    data_structures['sensor_mcurrent'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机电流(15f=60Bits)
                    data_structures['sensor_adc'][frame_idx] = struct.unpack('8H', safe_read(fidin, 16))#adc(8H=16Bits) #H无符号短整数(unsigned short)
                    data_structures['sensor_io'][frame_idx] = struct.unpack('1H', safe_read(fidin, 2))#io(1H=2Bits)
                    data_structures['sensor_reserved'][frame_idx] = struct.unpack('1H', safe_read(fidin, 2))#reserved(1H=2Bits)
                    data_structures['sensor_timestamp'][frame_idx] = struct.unpack('2H', safe_read(fidin, 4))#左右脚时间戳(2H=4Bits)
                    data_structures['sensor_isvalid'][frame_idx] = struct.unpack('6H', safe_read(fidin, 12))#有效判断(6H=12Bits)

                    # 3. 读取状态数据 (116Bits)
                    data_structures['status_angle'][frame_idx] = struct.unpack('7f', safe_read(fidin, 28))#关节角度 (7f, 28Bits)                   
                    data_structures['status_force'][frame_idx] = struct.unpack('8f', safe_read(fidin, 32))#力数据 (8f, 32Bits)                    
                    data_structures['status_motortorque'][frame_idx] = struct.unpack('4f', safe_read(fidin, 16))#电机力矩 (4f, 16Bits)                   
                    data_structures['status_leftratio'][frame_idx] = struct.unpack('f', safe_read(fidin, 4))#步态进度和置信度 (1f, 4Bits)
                    data_structures['status_leftconfidence'][frame_idx] = struct.unpack('f', safe_read(fidin, 4))
                    data_structures['status_rightratio'][frame_idx] = struct.unpack('f', safe_read(fidin, 4))
                    data_structures['status_rightconfidence'][frame_idx] = struct.unpack('f', safe_read(fidin, 4))
                    data_structures['status_motionratio'][frame_idx] = struct.unpack('f', safe_read(fidin, 4))
                    data_structures['status_motionconfidence'][frame_idx] = struct.unpack('f', safe_read(fidin, 4))              
                    data_structures['status_exostate'][frame_idx] = struct.unpack('B', safe_read(fidin, 1))#状态标志 (1B, 1B)
                    data_structures['status_leftstate'][frame_idx] = struct.unpack('B', safe_read(fidin, 1))
                    data_structures['status_rightstate'][frame_idx] = struct.unpack('B', safe_read(fidin, 1))
                    data_structures['status_motionstate'][frame_idx] = struct.unpack('B', safe_read(fidin, 1))    
                    data_structures['status_errorcode1'][frame_idx] = struct.unpack('H', safe_read(fidin, 2))#错误代码和有效性 (1H,2Bits)
                    data_structures['status_isvalid1'][frame_idx] = struct.unpack('H', safe_read(fidin, 2))
                    data_structures['status_debug_time'][frame_idx] = struct.unpack('I', safe_read(fidin, 4))#调试信息 (4Bits) 32位无符号整数
                    data_structures['status_debug_motionstate'][frame_idx] = struct.unpack('B', safe_read(fidin, 1)) 
                    data_structures['status_debug_robotversion'][frame_idx] = struct.unpack('H', safe_read(fidin, 2))
                    data_structures['status_debug_flag'][frame_idx] = struct.unpack('B', safe_read(fidin, 1))

                    # 4. 控制数据 (8Bits)
                    data_structures['control_timestamp'][frame_idx] = struct.unpack('H', safe_read(fidin, 2))
                    data_structures['control_period'][frame_idx] = struct.unpack('H', safe_read(fidin, 2))
                    data_structures['control_timeused'][frame_idx] = struct.unpack('H', safe_read(fidin, 2))
                    data_structures['control_recordnum'][frame_idx] = struct.unpack('H', safe_read(fidin, 2))
                    
                    # 跳过剩余字段 (优化性能时可逐字段读取)
                    fidin.seek(frame_size - 4 - 368 - 116 - 8, os.SEEK_CUR)  # 跳过frame_size - 4 - 352 - 91 - 8=57Bits

                    # # 6. 读取payload (15Bits)
                    # payload_size = frame_size - 4 - 352 - 91 - 8 - 1
                    # data_structures['payload'][frame_idx] = (safe_read(fidin, payload_size))
                    
                    # # 7. 校验位 (1B)
                    # data_structures['checksum'][frame_idx] = struct.unpack('B', safe_read(fidin, 1))
                    
                except (IOError, struct.error) as e:
                    print(f"第 {frame_idx} 帧读取错误: {str(e)}")
                    break
                
        print(f"读取 {frame_idx+1}帧数据")
        if endpoint is None:
            endpoint = total_frames
        if startpoint > endpoint:
            print('坐标点选择错误！')
            return
        
        # 返回切片数据
        result = {}
        for key in data_structures:
            if key == 'time':  # 特殊处理时间序列
                result[key] = data_structures[key][startpoint:endpoint]
            else:
                result[key] = data_structures[key][startpoint:endpoint, ...]
        
        return result
    
    except Exception as e:
        print(f"文件处理错误: {str(e)}")
        return None



def plot_sensor_data(data):
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
    joint_names = ['angBack', 'angHipL', 'angKneeL', 'angAnkleL', 'angHipR', 'angKneeR', 'angAnkleR']
    for i in range(7):
        ax3.plot(time, data['status_angle'][:, i], label=joint_names[i])
    # ax3.set_title('joint angle')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('ang (du)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 绘制电机电流
    ax4 = plt.subplot(gs[1, 1])
    motor_names = ['mposHipL', 'mposKneeL', 'mposHipR', 'mposKneeR']
    for i in range(4):
        ax4.plot(time, data['sensor_mcurrent'][:, i], label=motor_names[i])
    # ax4.set_title('motor current')
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('current (A)')
    ax4.legend()
    ax4.grid(True)
    
    # # 8. 绘制电机位置
    # ax8 = plt.subplot(gs[3, 1])
    # for i in range(4):
    #     ax8.plot(time, data['sensor_mpos'][:, i], label=motor_names[i])
    # ax8.set_title('电机位置')
    # ax8.set_xlabel('时间 (s)')
    # ax8.set_ylabel('位置 (rad)')
    # ax8.legend()
    # ax8.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = read_bin_file(startpoint=129,endpoint=1047)
    if data is not None:
        print("\n数据摘要:")
        print(f"总帧数: {len(data['DATA_frame'])}")
        print("第一帧加速度均值:", np.mean(data['sensor_IMU_acc'][0]))
        print("第一帧四元数:", data['sensor_IMU_q4'][0][:4])  # 显示前4个值
        #print(bin_data[:100].hex(' '))# 十六进制
    # 绘制数据
    plot_sensor_data(data)


