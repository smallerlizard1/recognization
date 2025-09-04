import serial
import serial.tools.list_ports
import threading
import time

class SerialCommunication:
    def __init__(self,port_name=None,baud_rate=115200):
        self.serial_port = None
        self.is_running = False
        self.receive_thread = None
        self.buffer = bytearray()  # 初始化缓冲区
        self.newframes = []  # 最新的数据帧
        self.frame_size = 36  # 固定帧长度
        self.header_byte = 0xAA  # 帧头标识
        self.baudrate = baud_rate
        self.portname = port_name
    
    def list_ports(self):
        """列出所有可用的串口"""
        ports = serial.tools.list_ports.comports()
        print("可用串口:", ports)
        return [port.device for port in ports]
    
    def open_port(self, timeout=1):
        """打开串口
        阻塞读取,超时时间为1s
        """
        try:
            self.serial_port = serial.Serial(
                port=self.portname,
                baudrate=self.baudrate,
                timeout=timeout
            )
            self.is_running = True
            # 启动守护接收线程,当主线程结束时，所有守护线程会自动强制终止，而不管它们是否执行完毕。
            self.receive_thread = threading.Thread(target=self._receive_data, daemon=True)
            self.receive_thread.start()
            print(f"串口 {self.portname} 已打开，波特率 {self.baudrate}")
            return True
        except Exception as e:
            print(f"打开串口失败: {e}")
            return False
    
    def close_port(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            self.is_running = False
            # 等待接收线程结束（最多等待1秒）
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=1.0)
                if self.receive_thread.is_alive():
                    print("警告: 接收线程未正常退出")
            self.serial_port.close()
            print("串口已关闭")
    
    def send_data(self, data):
        """发送数据"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(data)
                # print(f"发送: {data}")
            except Exception as e:
                print(f"发送数据失败: {e}")
        else:
            print("串口未打开")
    
    def _receive_data(self):
        """接收数据的线程函数
        self.is_running: 线程运行标志位,当设置为False时线程会退出
        使用缓冲区累积数据，不一定接受几个字节的数据，因此读取至少一个完整帧后，处理数据
        """
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0: # 读取所有可用数据
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    # data = self.serial_port.read(35)  # read(size):从串口阻塞读取指定数量的字节                   
                    print(f"实际读取的字节数: {len(data)}")  # 输出真实读取的字节数
                    self.buffer.extend(data)                    
                    # self._process_frames()# 帧处理逻辑
            except Exception as e:
                print(f"接收数据出错: {e}")
                break
    
    def _process_frames(self):
        """从缓冲区提取完整帧(保证以0xAA开头)"""
        while True:
            # 1. 检查缓冲区是否足够长
            if len(self.buffer) < self.frame_size:
                return
            
            # 2. 查找帧头0xAA的位置
            header_pos = -1
            for i in range(len(self.buffer) - self.frame_size + 1):
                if self.buffer[i] == self.header_byte:
                    header_pos = i
                    break
            
            # 3. 未找到有效帧头时清理缓冲区
            if header_pos == -1:
                # 保留最后 (frame_size-1) 字节（可能包含半个帧头）
                keep_len = min(len(self.buffer), self.frame_size - 1)
                self.buffer = self.buffer[-keep_len:] if keep_len > 0 else bytearray()
                return
            
            # 4. 丢弃帧头前的无效数据
            if header_pos > 0:
                print(f"丢弃无效数据: {self.buffer[:header_pos]}")
                self.buffer = self.buffer[header_pos:]
            
            # 5. 提取完整帧
            if len(self.buffer) >= self.frame_size:
                self.newframes.append(self.buffer[:self.frame_size]) #对缓冲区进行切片，取出前35字节
                self.buffer = self.buffer[self.frame_size:]#移除已处理的帧数据
                # print(self.newframes[-1])

if __name__ == "__main__":
    print("主线程:", threading.main_thread())  # 输出主线程信息
    print("当前线程:", threading.current_thread())  # 获取当前运行的线程
    # serial_comm = SerialCommunication(port_name='/dev/ttyACM0', baud_rate=115200)# 新建串口对象，并设置接收回调
    serial_comm = SerialCommunication(port_name='/dev/ttyUSB0', baud_rate=115200)# 新建串口对象，并设置接收回调
    # ports = serial_comm.list_ports()# 列出可用串口    
    # if ports:
    if serial_comm.open_port():
        try:
            while True:
                # 从控制台读取输入并发送
                # user_input = input("输入要发送的数据(输入'quit'退出): ")
                # if user_input.lower() == 'quit':
                #     break
                # serial_comm.send_data(user_input)
                a=1
        finally:
            serial_comm.close_port()
    # else:
    #     print("没有找到可用的串口")
