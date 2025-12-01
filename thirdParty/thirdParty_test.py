#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
肌电数据采集和存储 + Tk 实时波形显示示例程序
- 使用 WearLab 设备采集肌电数据并保存到 HDF5 文件
- 同时使用 tkinter Canvas 实时显示所有通道的波形（颜色各不相同）
"""

import time
import logging
import signal
import sys
from collections import deque
from typing import Dict, Optional

import h5py
import numpy as np
import threading
import tkinter as tk
import colorsys

from wearlab_protocolX.meta import (
    PackageType,
    EmgDataPacket,
    HardwareDiscoveryPacket,
)
from wearlab_protocolX.wearlab_device import DeviceCommunication, WearLabDevice

# 全局变量
devices = []
device_comm: Optional[DeviceCommunication] = None
emg_handler = None
is_running = True

# 实时绘图相关全局
live_buffer = None       # EMGLivePlotBuffer 实例
tk_root = None           # Tk root 实例


class EMGLivePlotBuffer:
    """用于 Tk 实时波形显示的环形缓冲区（只用于显示，不用于存盘）"""

    def __init__(self, channels: int = 64, window_size: int = 1000):
        """
        Args:
            channels: 通道数
            window_size: 显示窗口内的最大点数
        """
        self.channels = channels
        self.window_size = window_size
        # shape: (window_size, channels)
        self.buffer = np.zeros((window_size, channels), dtype=np.float32)
        self.index = 0
        self.len = 0
        self.lock = threading.Lock()

    def append(self, emg_data: np.ndarray):
        """
        追加一帧数据到缓冲区（线程安全）
        emg_data: shape (channels,)
        """
        if emg_data.shape[0] != self.channels:
            return
        with self.lock:
            self.buffer[self.index] = emg_data
            self.index = (self.index + 1) % self.window_size
            self.len = min(self.len + 1, self.window_size)

    def get_window(self):
        """
        返回按时间顺序排列的最近 window_size 个数据: shape (L, channels)
        L <= window_size
        """
        with self.lock:
            if self.len == 0:
                return None
            if self.len < self.window_size:
                return self.buffer[:self.len].copy()
            # 环形展开
            idx = self.index
            return np.vstack((self.buffer[idx:], self.buffer[:idx])).copy()


def generate_distinct_colors(n: int):
    """
    生成 n 个颜色各不相同的十六进制颜色字符串列表（#RRGGBB）
    使用 HSV 均匀分布，然后转换到 RGB。
    """
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        s = 0.8
        v = 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append("#{:02X}{:02X}{:02X}".format(
            int(r * 255), int(g * 255), int(b * 255)
        ))
    return colors


class EMGLivePlotWindow:
    """Tk 实时波形窗口"""

    def __init__(self,
                 root: tk.Tk,
                 buffer: EMGLivePlotBuffer,
                 channels_to_show=None,
                 refresh_ms: int = 50,
                 bg: str = "black"):
        """
        Args:
            root: Tk 根窗口
            buffer: 实时数据缓冲区
            channels_to_show: 要显示的通道索引列表；如为 None，则显示所有通道 (0..channels-1)
            refresh_ms: 刷新间隔（毫秒）
            bg: 背景颜色
        """
        self.root = root
        self.buffer = buffer
        self.refresh_ms = refresh_ms
        self.bg = bg

        # 默认显示所有通道
        if channels_to_show is None:
            channels_to_show = list(range(self.buffer.channels))
        self.channels_to_show = channels_to_show

        # 为每个通道生成一种独特的颜色
        self.channel_colors = generate_distinct_colors(len(self.channels_to_show))

        self.canvas_width = 1000
        self.canvas_height = 600

        self.root.title("EMG 实时波形")
        self.canvas = tk.Canvas(root,
                                width=self.canvas_width,
                                height=self.canvas_height,
                                bg=self.bg)
        self.canvas.pack(fill="both", expand=True)

        # 缩放参数
        self.y_scale = 1.0    # 垂直缩放乘子
        self.y_offset = 0.0   # 垂直平移（像素）

        # 绑定键盘控制缩放
        self.root.bind("<Up>", self.increase_scale)
        self.root.bind("<Down>", self.decrease_scale)

        # 启动周期刷新
        self.update_plot()

    def increase_scale(self, event=None):
        self.y_scale *= 1.2

    def decrease_scale(self, event=None):
        self.y_scale /= 1.2
        if self.y_scale <= 0:
            self.y_scale = 0.1

    def update_plot(self):
        data = self.buffer.get_window()
        if data is None:
            self.root.after(self.refresh_ms, self.update_plot)
            return

        L, C = data.shape
        if L < 2:
            self.root.after(self.refresh_ms, self.update_plot)
            return

        # 获取画布实际大小（允许用户拉伸窗口）
        w = self.canvas.winfo_width() or self.canvas_width
        h = self.canvas.winfo_height() or self.canvas_height

        self.canvas.delete("all")

        x_step = w / max(L - 1, 1)

        ch_count = len(self.channels_to_show)
        # 多通道时，使用通道 index 做垂直分层显示，避免严重重叠
        # 每个通道占据总高度的一小部分区间
        for idx, ch in enumerate(self.channels_to_show):
            if ch >= C:
                continue
            ch_data = data[:, ch]

            # 自动根据当前窗口数据确定振幅范围
            dmin, dmax = float(ch_data.min()), float(ch_data.max())
            if dmax == dmin:
                dmax = dmin + 1.0

            # 当前通道的中心线（分层）
            # 在画布中将多个通道从上到下均匀分布
            # 预留一些上下边距
            margin_top = h * 0.05
            margin_bottom = h * 0.05
            avaliable_h = h - margin_top - margin_bottom
            per_ch_height = avaliable_h / max(ch_count, 1)

            center_line = margin_top + per_ch_height * (idx + 0.5)

            # 把 dmin~dmax 映射到当前通道垂直区域的 80% 高度
            scale = (per_ch_height * 0.4 / (dmax - dmin)) * self.y_scale

            points = []
            for i, v in enumerate(ch_data):
                x = i * x_step
                norm = v - (dmin + dmax) / 2.0
                y = center_line - norm * scale + self.y_offset
                points.extend((x, y))

            color = self.channel_colors[idx]
            self.canvas.create_line(points, fill=color, width=1.0)

        self.root.after(self.refresh_ms, self.update_plot)


class EMGHDF5Recorder:
    """EMG 数据 HDF5 记录器类"""

    def __init__(self, filename: str, device_id: str, channels_size: int = 64, buffer_size: int = 1000):
        """
        初始化记录器

        Args:
            filename: HDF5 文件名
            device_id: 设备 ID
            channels_size: 通道数
            buffer_size: 内部写盘缓冲区大小（单位：采样点）
        """
        self.channels_size = channels_size
        self.filename = filename
        self.device_id = device_id
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.timestamps_buffer = deque(maxlen=buffer_size)
        self.is_recording = False
        self.file_handle: Optional[h5py.File] = None
        self.logger = logging.getLogger(__name__)

    def start_recording(self, device_info: dict = None):
        """开始记录数据"""
        if self.is_recording:
            self.logger.warning("Recording is already in progress")
            return

        try:
            # 打开 HDF5 文件（追加模式）
            self.file_handle = h5py.File(self.filename, 'a')

            # 创建设备组
            if self.device_id in self.file_handle:
                device_group = self.file_handle[self.device_id]
            else:
                device_group = self.file_handle.create_group(self.device_id)

            # 存储设备信息
            if device_info:
                info_group = device_group.require_group("device_info")
                for key, value in device_info.items():
                    info_group.attrs[key] = value

            # 创建数据集
            if "emg_data" not in device_group:
                device_group.create_dataset(
                    "emg_data",
                    shape=(0, self.channels_size),
                    maxshape=(None, self.channels_size),
                    dtype=np.float32,
                    chunks=True,
                    compression='gzip'
                )

            if "timestamps" not in device_group:
                device_group.create_dataset(
                    "timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=True,
                    compression='gzip'
                )

            self.is_recording = True
            self.logger.info(f"Started recording for device {self.device_id} to {self.filename}")

        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            raise

    def add_data(self, emg_packet: EmgDataPacket):
        """添加数据到缓冲区"""
        if not self.is_recording:
            self.logger.warning("Recording not started, ignoring data")
            return

        # 添加数据到缓冲区
        self.data_buffer.append(emg_packet.emg_data)
        self.timestamps_buffer.append(emg_packet.pack_time)
        self.logger.debug(f"Added data for device {self.device_id}")

        # 如果缓冲区满了，刷新到文件
        if len(self.data_buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self.is_recording or not self.file_handle:
            return

        num_points = len(self.data_buffer)
        if num_points == 0:
            return

        try:
            device_group = self.file_handle[self.device_id]
            emg_dataset = device_group["emg_data"]
            timestamp_dataset = device_group["timestamps"]

            current_size = emg_dataset.shape[0]
            new_size = current_size + num_points

            emg_dataset.resize((new_size, self.channels_size))
            timestamp_dataset.resize((new_size,))

            emg_dataset[current_size:new_size, :] = np.array(self.data_buffer)
            timestamp_dataset[current_size:new_size] = np.array(self.timestamps_buffer)

            self.data_buffer.clear()
            self.timestamps_buffer.clear()

            self.file_handle.flush()

            self.logger.debug(f"Flushed {num_points} data points to file")
        except Exception as e:
            self.logger.error(f"Failed to flush buffer: {e}")

    def stop_recording(self):
        """停止记录并保存所有数据"""
        if not self.is_recording:
            return

        # 刷新剩余数据
        self._flush_buffer()

        # 关闭文件
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

        self.is_recording = False
        self.logger.info(f"Stopped recording for device {self.device_id}")
        self.logger.info(f"File is saved to {self.filename}")


class EMGDataHandler:
    """EMG 数据处理器类"""

    def __init__(self, base_filename: str, live_buffer: Optional[EMGLivePlotBuffer] = None):
        """
        Args:
            base_filename: 基础文件名，用于生成 HDF5 文件名
            live_buffer: 实时绘图缓冲区（可选）
        """
        self.base_filename = base_filename
        self.recorders: Dict[str, EMGHDF5Recorder] = {}
        self.device_info: Dict[str, dict] = {}
        self.logger = logging.getLogger(__name__)
        self.live_buffer = live_buffer

    def data_callback(self, data_packet):
        """数据回调函数，处理来自设备的数据包"""
        try:
            if isinstance(data_packet, EmgDataPacket):
                self._handle_emg_data(data_packet)
            elif isinstance(data_packet, HardwareDiscoveryPacket):
                self._handle_device_info(data_packet)
        except Exception as e:
            self.logger.error(f"Error processing data packet: {e}")

    def _handle_emg_data(self, emg_packet: EmgDataPacket):
        """处理 EMG 数据包"""
        device_id = emg_packet.hardware_identifier

        # 实时绘图缓冲写入（轻量操作）
        if self.live_buffer is not None:
            try:
                arr = np.array(emg_packet.emg_data, dtype=np.float32)
                self.live_buffer.append(arr)
            except Exception as e:
                self.logger.error(f"Error appending to live buffer: {e}")

        # HDF5 记录处理
        if device_id not in self.recorders:
            filename = f"{self.base_filename}.h5"
            self.recorders[device_id] = EMGHDF5Recorder(filename, device_id)
            device_info = self.device_info.get(device_id, {})
            self.recorders[device_id].start_recording(device_info)

        self.recorders[device_id].add_data(emg_packet)

    def _handle_device_info(self, info_packet: HardwareDiscoveryPacket):
        """处理设备信息包"""
        device_id = info_packet.hardware_identifier

        self.device_info[device_id] = {
            "display_name": info_packet.display_name,
            "model": info_packet.model,
            "manufacturer": info_packet.manufacturer,
            "hardware_revision": info_packet.hardware_revision,
            "firmware_version": info_packet.firmware_version,
            "mcu_temp": info_packet.mcu_temp,
            "battery_voltage": info_packet.battery_voltage
        }

        self.logger.info(f"Received device info for {device_id}: {info_packet.display_name}")

    def stop_all_recorders(self):
        """停止所有记录器"""
        for recorder in self.recorders.values():
            recorder.stop_recording()

        self.recorders.clear()
        self.logger.info("All recorders stopped")


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("emg_recording.log")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting EMG data recording demo")
    return logger


def device_disconnect_callback(device: WearLabDevice):
    """设备断开连接回调"""
    global devices
    if device in devices:
        devices.remove(device)
    print(f"Device {device.hardware_identifier} disconnected")
    logging.info(f"Device {device.hardware_identifier} disconnected")


def device_discovery_callback(device: WearLabDevice):
    """设备发现回调"""
    global device_comm
    print(f"Device {device.hardware_identifier} discovered, connecting...")
    logging.info(f"Device {device.hardware_identifier} discovered")
    device_comm.connect_device(device.hardware_identifier)


def device_connect_callback(device: WearLabDevice, *args):
    """设备连接回调"""
    global devices, device_comm, emg_handler
    devices.append(device)
    print(f"Device {device.hardware_identifier} connected")
    logging.info(f"Device {device.hardware_identifier} connected")

    # 注册数据变化回调
    device_comm.register_data_change_callback(emg_handler.data_callback)

    # 设置采样周期（根据你的设备协议，此处 1000 代表 1000ms 或 1000Hz 请自行确认）
    time.sleep(3)  # 等待设备稳定
    device_comm.send_command(device.hardware_identifier, PackageType.SetCycleSEMG, 1000)
    print(f"Set SEMG cycle for device {device.hardware_identifier}")


def signal_handler(sig, frame):
    """信号处理函数，用于优雅地退出程序"""
    global is_running
    print("\nReceived interrupt signal. Stopping recording...")
    is_running = False
    cleanup()
    sys.exit(0)


def cleanup():
    """清理资源"""
    global device_comm, emg_handler, tk_root

    print("Cleaning up resources...")

    # 停止所有记录器
    if emg_handler:
        emg_handler.stop_all_recorders()

    # 停止设备通信
    if device_comm:
        device_comm.stop_communication()

    # 退出 Tk 主循环
    if tk_root is not None:
        try:
            tk_root.quit()
        except Exception:
            pass

    print("Cleanup completed.")


def main():
    """主函数"""
    global device_comm, emg_handler, is_running, live_buffer, tk_root

    logger = setup_logging()

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 创建实时显示缓冲区（64 通道，显示最近 1000 点）
    live_buffer = EMGLivePlotBuffer(channels=64, window_size=1000)

    # 创建数据处理器，注入实时缓冲区
    emg_handler = EMGDataHandler("emg_data", live_buffer=live_buffer)

    # 创建设备通信对象
    device_comm = DeviceCommunication(target_vid=0x0483, target_pid=0x5740)
    device_comm.register_device_discovery_callback(device_discovery_callback)
    device_comm.register_device_connect_callback(device_connect_callback)
    device_comm.register_device_disconnect_callback(device_disconnect_callback)

    # 启动通信
    print("Starting device communication...")
    device_comm.start_communication()
    print("Waiting for devices to connect...")
    print("Press Ctrl+C to stop recording and exit")

    # 创建 Tk 窗口并启动实时绘图
    tk_root = tk.Tk()
    # 不指定 channels_to_show，默认显示所有通道并自动上色
    plot_window = EMGLivePlotWindow(
        tk_root,
        buffer=live_buffer,
        channels_to_show=None,  # None -> 显示所有通道
        refresh_ms=120,
        bg="black"
    )

    try:
        tk_root.mainloop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt in Tk mainloop")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
