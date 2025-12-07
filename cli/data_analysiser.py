# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

import mplcursors
import numpy as np
import pandas as pd
from scipy.signal import welch
import neurokit2 as nk

# Matplotlib 中文 & 样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


class EMGApp(tk.Tk):
    """
    EMG 多设备 / 多通道分析（无 IMU）

    H5 数据格式要求：
      ['device_id', 'timestamp', 'ch_0', ..., 'ch_63']

    特性：
    - 多设备，按 timestamp 通过全局时间轴对齐
    - 多通道同时事件对齐分析
    - 多设备联合事件分析（可选）
    - 频域分析：MPF 随时间（Welch PSD）
    """
    def __init__(self):
        super().__init__()
        self.title("EMG Data Analysis (Tk, H5, Multi-device)")
        self.geometry("1400x900")

        # ====== 整体数据（跨设备） ======
        self.full_df = None              # 整个 H5 中的数据 DataFrame
        self.device_ids = []             # 所有 device_id
        self.current_device_id = None    # 当前选择的设备

        # ====== 时间轴全局对齐参数 ======
        self.global_time_origin = None   # 全局时间原点（timestamp）
        self.time_scale = None           # 时间单位比例（例如：1e6 表示 timestamp 是微秒）

        # ====== 当前设备数据 ======
        self.data_df = None              # 当前 device 的 DataFrame
        self.time_raw = None             # 原始 timestamp
        self.time_s = None               # 换算为秒的全局时间轴（不同设备共享同一原点）
        self.signals = None              # shape: (N, C)，只包含 ch_* 通道
        self.channel_names = []          # ['ch_0', 'ch_1', ...]
        self.sampling_rate = 1000.0      # 采样率，自动估计，可手动改

        # ====== EMG 相关 ======
        self.emg_cleaned = {}            # ch -> cleaned
        self.emg_amplitude = {}          # ch -> amplitude
        self.emg_signals_result = None   # 多设备+多通道 EMG 时序特征
        self.event_result_df = None      # 多设备+多通道 事件相关分析结果

        # ====== 事件 & 通道选择 ======
        self.events = []                 # 双击添加的事件（全局时间轴的秒数）
        self.selected_channels = []      # 当前选中的通道列表

        # ====== GUI: Notebook + Tabs ======
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_preprocess = DataPreprocessingTab(self, self.notebook)
        self.notebook.add(self.tab_preprocess, text="数据预处理")

        self.tab_label = DataLabelingTab(self, self.notebook)
        self.notebook.add(self.tab_label, text="数据标记")

        self.tab_analysis = None         # 数据分析 Tab 在分析后动态创建

    # ===================== H5 加载 & 设备管理 =====================

    def load_h5_data(self, file_path: str):
        """
        从 H5 读取数据 (基于 h5py 解析层级结构)。
        结构要求：
          - Group (device_id)
            - Dataset: emg_data (shape: N x Channels)
            - Dataset: timestamps (shape: N)
        """
        import h5py
        import os

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        frames = []

        try:
            with h5py.File(file_path, "r") as f:
                # 遍历 H5 文件下的第一层 keys (视为 device_id)
                for device_id in f.keys():
                    grp = f[device_id]

                    # 确保是 Group
                    if not isinstance(grp, h5py.Group):
                        continue

                    # 检查必要的数据集是否存在
                    if "emg_data" not in grp or "timestamps" not in grp:
                        print(f"警告: Group {device_id} 缺少 'emg_data' 或 'timestamps'，已跳过。")
                        continue

                    # 读取数据
                    emg_data = grp["emg_data"][:]
                    timestamps = grp["timestamps"][:]

                    # 校验维度
                    if emg_data.ndim != 2:
                        print(f"警告: Group {device_id} emg_data 维度为 {emg_data.ndim} (预期 2)，已跳过。")
                        continue

                    n_samples, data_dim = emg_data.shape

                    if timestamps.shape[0] != n_samples:
                        print(
                            f"警告: Group {device_id} 长度不匹配 (emg={n_samples}, time={timestamps.shape[0]})，已跳过。")
                        continue

                    # 构造通道列名: ch_0, ch_1, ...
                    channel_cols = [f"ch_{i}" for i in range(data_dim)]

                    # 构造该设备的 DataFrame
                    df_dev = pd.DataFrame(emg_data, columns=channel_cols)
                    df_dev.insert(0, "timestamp", timestamps)
                    df_dev.insert(0, "device_id", device_id)

                    frames.append(df_dev)

        except Exception as e:
            raise ValueError(f"解析 H5 文件结构失败: {e}")

        if not frames:
            raise ValueError("H5 文件中未找到任何合法的 EMG 数据 (需包含 emg_data 和 timestamps)")

        # 合并所有设备的数据
        df = pd.concat(frames, ignore_index=True)

        # --- 以下逻辑保持原样，用于校验和更新 UI 状态 ---

        expected_cols = {"device_id", "timestamp"}
        if not expected_cols.issubset(df.columns):
            # 理论上前面的构造逻辑保证了这一点，但保留作为双重检查
            raise ValueError(f"构造的数据缺少必要列 {expected_cols}")

        # 通道列：所有以 ch_ 开头的列
        channel_cols = [c for c in df.columns if c.startswith("ch_")]
        if not channel_cols:
            raise ValueError("未找到任何 ch_* 通道列")

        self.full_df = df
        self.device_ids = sorted(df["device_id"].unique().tolist())

        if not self.device_ids:
            raise ValueError("device_id 列表为空")

        # 默认选择第一个设备
        self.current_device_id = self.device_ids[0]
        self.set_current_device(self.current_device_id)

        # 通知 Tab 更新
        if hasattr(self, 'tab_preprocess'):
            self.tab_preprocess.update_after_load(file_path)
        if hasattr(self, 'tab_label'):
            self.tab_label.update_after_load()


    def set_current_device(self, device_id):
        """
        按 device_id 过滤数据，并提取：
          - timestamp -> time_raw, time_s（全局对齐）
          - ch_* -> signals
        """
        if self.full_df is None:
            return
        if device_id not in self.device_ids:
            raise ValueError(f"设备 {device_id} 不存在")

        self.current_device_id = device_id
        df_dev = self.full_df[self.full_df["device_id"] == device_id].copy()
        if df_dev.empty:
            raise ValueError(f"设备 {device_id} 没有数据")

        df_dev = df_dev.sort_values("timestamp")

        self.time_raw = df_dev["timestamp"].astype(float).values
        self.data_df = df_dev

        self.channel_names = [c for c in df_dev.columns if c.startswith("ch_")]
        if not self.channel_names:
            raise ValueError(f"设备 {device_id} 没有 ch_* 通道")

        self.signals = df_dev[self.channel_names].astype(float).values

        # 估计时间轴 & 采样率 & 初步 EMG 清理
        self.estimate_time_and_fs()
        self.initial_emg_clean()

    # ===================== 时间 & 采样率 =====================

    def estimate_time_and_fs(self):
        """
        使用全局时间原点和时间单位：
        - time_s = (timestamp - global_time_origin) / time_scale
        - 采样率使用当前设备自己的时间间隔估计
        """
        t = self.time_raw
        if t is None or len(t) < 2:
            self.time_s = None
            self.sampling_rate = 1000.0
            return

        if self.global_time_origin is None:
            self.global_time_origin = t[0]
        if self.time_scale is None:
            # 再估一个，保险
            dt_raw = np.median(np.diff(t))
            if dt_raw <= 0:
                self.time_scale = 1.0
            elif dt_raw > 1e5:
                self.time_scale = 1e6
            elif dt_raw > 100:
                self.time_scale = 1e3
            else:
                self.time_scale = 1.0

        origin = self.global_time_origin
        scale = self.time_scale

        self.time_s = (t - origin) / scale

        dt_raw_dev = np.median(np.diff(t))
        if dt_raw_dev > 0:
            self.sampling_rate = float(1.0 / (dt_raw_dev / scale))
        else:
            self.sampling_rate = 1000.0

    def initial_emg_clean(self):
        """
        对当前设备的所有 ch_* 通道进行 EMG 清理和振幅计算。
        """
        self.emg_cleaned.clear()
        self.emg_amplitude.clear()
        if self.signals is None:
            return

        fs = self.sampling_rate
        for i, ch in enumerate(self.channel_names):
            sig = self.signals[:, i].astype(float)
            try:
                emg_c = nk.emg_clean(sig, sampling_rate=fs)
                emg_a = nk.emg_amplitude(emg_c)
            except Exception:
                # 某些通道如果不是 EMG 格式，直接用原始和绝对值
                emg_c = sig
                emg_a = np.abs(sig)
            self.emg_cleaned[ch] = emg_c
            self.emg_amplitude[ch] = emg_a

    # ===================== 剪裁 =====================

    def apply_trim(self, start_idx: int, end_idx: int):
        """
        将当前设备的 time_s / time_raw / signals / data_df 剪裁到 [start_idx, end_idx)
        """
        if self.time_s is None or self.signals is None:
            return

        start_idx = max(0, int(start_idx))
        end_idx = min(len(self.time_s), int(end_idx))
        if end_idx <= start_idx:
            messagebox.showwarning("警告", "剪裁范围不合法")
            return

        self.time_s = self.time_s[start_idx:end_idx]
        self.time_raw = self.time_raw[start_idx:end_idx]
        self.signals = self.signals[start_idx:end_idx, :]

        if self.data_df is not None:
            self.data_df = self.data_df.iloc[start_idx:end_idx].reset_index(drop=True)

        # 裁剪后重新做 EMG 清理
        self.initial_emg_clean()

        # 更新各个 Tab 的图
        self.tab_preprocess.redraw()
        self.tab_label.redraw()

    # ===================== EMG 事件 & 频域分析（多通道 + 多设备） =====================

    def run_emg_event_analysis(self, channels, all_devices=False):
        """
        对 channels（可以是单个或列表）做 EMG 特征 + 事件相关分析。
        - all_devices=False: 仅当前设备
        - all_devices=True : 所有设备联合分析

        使用全局时间轴对齐：
        - self.events 存储的是全局 time_s 中的秒数
        - 每个设备内部会把事件转换为“相对该设备起点的时间”传给 neurokit2
        """
        if isinstance(channels, str):
            channels = [channels]
        channels = [ch for ch in channels if ch in self.channel_names or ch in self.emg_cleaned]
        if not channels:
            messagebox.showwarning("警告", "没有有效的通道被选中")
            return
        if not self.events:
            messagebox.showerror("错误", "尚未标记任何事件（在图上双击添加）")
            return

        devices = self.device_ids if all_devices else [self.current_device_id]

        all_emg_signals = []
        all_events_df = []

        # 遍历设备
        for dev in devices:
            self.set_current_device(dev)  # 切换设备（不会自动重画 UI）

            fs = self.sampling_rate
            t_global = self.time_s
            if t_global is None or len(t_global) == 0:
                continue

            # 当前设备的起始时间（全局轴），用于把全局事件转为相对时间给 neurokit2
            dev_start = t_global[0]
            dev_end = t_global[-1]

            # 只保留落在该设备数据范围内的事件
            dev_events_global = [e for e in self.events if dev_start <= e <= dev_end]
            if not dev_events_global:
                continue

            # 相对事件时间（秒）
            dev_events_rel = [e - dev_start for e in dev_events_global]

            # 遍历通道
            for ch in channels:
                if ch not in self.emg_cleaned:
                    continue
                emg_clean = self.emg_cleaned[ch]

                # neurokit2 emg_process
                emg_signals, info = nk.emg_process(
                    emg_clean, sampling_rate=fs, method_activation="pelt"
                )

                # --------- RMS（频域相关统计之一） ---------
                window_size_ms = 50
                step_size_ms = 25
                window_size_samples = int(window_size_ms / 1000 * fs)
                step_size_samples = int(step_size_ms / 1000 * fs)

                rms_values = []
                sig_series = emg_signals["EMG_Clean"]
                N = len(sig_series)
                for i in range(0, N, window_size_samples):
                    window = sig_series.iloc[i:i + window_size_samples]
                    if len(window) == 0:
                        continue
                    rms = np.sqrt((window ** 2).mean())
                    rms_values.extend([rms] * len(window))

                if len(rms_values) < len(emg_signals) and len(rms_values) > 0:
                    rms_values.extend([rms_values[-1]] * (len(emg_signals) - len(rms_values)))
                rms_values = rms_values[:len(emg_signals)]
                emg_signals["EMG_RMS"] = rms_values

                # --------- 振幅差分 ---------
                emg_signals["EMG_Amplitude_Diff"] = emg_signals["EMG_Amplitude"].diff()

                # --------- MPF（Mean Power Frequency：频域分析） ---------
                signal_arr = emg_signals["EMG_Clean"].values
                mpf_values = []
                window_centers = []
                N = len(signal_arr)
                for start in range(0, N - window_size_samples + 1, step_size_samples):
                    end = start + window_size_samples
                    w = signal_arr[start:end]
                    freqs, psd = welch(
                        w, fs=fs,
                        nperseg=max(16, window_size_samples // 2),
                        noverlap=max(8, window_size_samples // 4)
                    )
                    if np.sum(psd) == 0:
                        mpf = 0
                    else:
                        mpf = np.sum(freqs * psd) / np.sum(psd)
                    mpf_values.append(mpf)
                    # 使用全局 time_s 的窗口中心时间，保证多设备对齐
                    t_window = t_global[start:end]
                    window_centers.append(float(t_window.mean()))

                # 全局时间轴
                t_full = t_global[:len(signal_arr)]
                if len(mpf_values) >= 2:
                    emg_signals["EMG_MPF"] = np.interp(t_full, window_centers, mpf_values)
                else:
                    emg_signals["EMG_MPF"] = np.zeros_like(t_full)

                # 填充额外信息（使用全局时间）
                emg_signals["Time_s"] = t_full
                emg_signals["Channel"] = ch
                emg_signals["device_id"] = dev
                all_emg_signals.append(emg_signals)

                # 事件相关分析：events 使用相对时间
                epochs = nk.epochs_create(
                    emg_signals,
                    events=dev_events_rel,
                    sampling_rate=fs,
                    epochs_start=-0.1,
                    epochs_end=1.9
                )
                df_events = nk.emg_eventrelated(epochs)
                df_events["Channel"] = ch
                df_events["device_id"] = dev
                all_events_df.append(df_events)

        if not all_emg_signals:
            messagebox.showwarning("警告", "在任何设备上都没有匹配的事件或通道数据。")
            return

        self.emg_signals_result = pd.concat(all_emg_signals, ignore_index=True)
        if all_events_df:
            self.event_result_df = pd.concat(all_events_df, ignore_index=True)
        else:
            self.event_result_df = None

        # 创建或更新分析 Tab
        if self.tab_analysis is None:
            self.tab_analysis = DataAnalysisTab(self, self.notebook)
            self.notebook.add(self.tab_analysis, text="数据分析")
        self.tab_analysis.update_content()


# =====================================================================
# Tab1：数据预处理（加载 H5 + 选择设备 + 剪裁）
# =====================================================================

class DataPreprocessingTab(ttk.Frame):
    def __init__(self, app: EMGApp, master=None):
        super().__init__(master)
        self.app = app

        # 顶部区域：加载、设备选择、采样率
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.load_btn = ttk.Button(top, text="加载 H5 数据", command=self.on_load_clicked)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(top, text="设备：").pack(side=tk.LEFT)
        self.dev_var = tk.StringVar()
        self.dev_combo = ttk.Combobox(top, textvariable=self.dev_var, state="readonly", width=15)
        self.dev_combo.pack(side=tk.LEFT, padx=5)
        self.dev_combo.bind("<<ComboboxSelected>>", self.on_device_changed)

        ttk.Label(top, text="采样率(Hz)：").pack(side=tk.LEFT)
        self.fs_var = tk.StringVar(value="未知")
        self.fs_entry = ttk.Entry(top, textvariable=self.fs_var, width=12)
        self.fs_entry.pack(side=tk.LEFT, padx=5)

        self.fs_update_btn = ttk.Button(top, text="更新采样率", command=self.update_fs_from_entry)
        self.fs_update_btn.pack(side=tk.LEFT, padx=5)

        self.file_label_var = tk.StringVar(value="未加载文件")
        ttk.Label(top, textvariable=self.file_label_var).pack(side=tk.LEFT, padx=10)

        # 剪裁按钮区域
        trim_frame = ttk.Frame(self)
        trim_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.select_trim_btn = ttk.Button(trim_frame, text="选择剪裁范围", command=self.on_select_trim_range)
        self.select_trim_btn.pack(side=tk.LEFT, padx=5)

        self.apply_trim_btn = ttk.Button(trim_frame, text="应用剪裁", command=self.on_apply_trim)
        self.apply_trim_btn.pack(side=tk.LEFT, padx=5)

        # Matplotlib 图
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        self.span_selector = None
        self.trim_start_idx = None
        self.trim_end_idx = None

        self.cursor = None

    def on_load_clicked(self):
        file_path = filedialog.askopenfilename(
            title="选择 HDF5 数据文件",
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.app.load_h5_data(file_path)
        except Exception:
            # 错误已经在 app 里提示过了
            return

    def update_after_load(self, file_path: str):
        self.file_label_var.set(f"已加载：{file_path}")

        # 更新设备列表
        self.dev_combo["values"] = self.app.device_ids
        if self.app.device_ids:
            self.dev_var.set(self.app.current_device_id)

        # 更新采样率显示
        self.fs_var.set(f"{self.app.sampling_rate:.2f}")

        self.redraw()

    def on_device_changed(self, event):
        dev = self.dev_var.get()
        if not dev:
            return
        try:
            self.app.set_current_device(dev)
            self.fs_var.set(f"{self.app.sampling_rate:.2f}")
            self.redraw()
            self.app.tab_label.update_after_load()
        except Exception as e:
            messagebox.showerror("错误", f"切换设备失败：\n{e}")

    def redraw(self):
        self.ax.clear()
        if self.app.time_s is None or self.app.signals is None:
            self.ax.set_title("尚未加载数据")
            self.canvas.draw()
            return

        t = self.app.time_s
        sigs = self.app.signals
        for i, ch in enumerate(self.app.channel_names):
            self.ax.plot(t, sigs[:, i], label=ch)

        self.ax.set_xlabel("全局时间 (秒)")
        self.ax.set_ylabel("振幅")
        self.ax.set_title(f"设备 {self.app.current_device_id}：所有通道原始信号（全局对齐）")
        self.ax.legend(loc='upper right', fontsize='small')
        self.fig.tight_layout()
        self.canvas.draw()

        if self.cursor:
            self.cursor.disconnect()
        self.cursor = mplcursors.cursor(self.ax.lines, hover=True)
        self.cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"t = {sel.target[0]:.3f} s\n值 = {sel.target[1]:.3f}"
            )
        )

    def update_fs_from_entry(self):
        try:
            fs = float(self.fs_var.get())
            if fs <= 0:
                raise ValueError
            self.app.sampling_rate = fs
            self.app.initial_emg_clean()
            messagebox.showinfo("信息", f"采样率已更新为 {fs:.2f} Hz，并已重新清理 EMG")
        except Exception:
            messagebox.showerror("错误", "采样率输入不合法")

    def on_select_trim_range(self):
        if self.app.time_s is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        if self.span_selector:
            self.span_selector.disconnect_events()
            self.span_selector = None

        def on_select(xmin, xmax):
            t = self.app.time_s
            start_idx = np.searchsorted(t, xmin, side="left")
            end_idx = np.searchsorted(t, xmax, side="right")
            self.trim_start_idx = start_idx
            self.trim_end_idx = end_idx
            messagebox.showinfo(
                "提示",
                f"已选择剪裁范围：索引 {start_idx} ~ {end_idx}\n时间 {xmin:.3f} s ~ {xmax:.3f} s"
            )

        self.span_selector = SpanSelector(
            self.ax,
            on_select,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.2, facecolor='tab:blue')
        )
        messagebox.showinfo("提示", "在图上拖动鼠标选择剪裁范围")

    def on_apply_trim(self):
        if self.trim_start_idx is None or self.trim_end_idx is None:
            messagebox.showwarning("警告", "请先选择剪裁范围")
            return
        self.app.apply_trim(self.trim_start_idx, self.trim_end_idx)


# =====================================================================
# Tab2：数据标记（多通道选择 + 双击添加事件 + 多设备联合分析开关）
# =====================================================================

class DataLabelingTab(ttk.Frame):
    def __init__(self, app: EMGApp, master=None):
        super().__init__(master)
        self.app = app

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top, text="选择 EMG 通道（可多选）：").pack(side=tk.LEFT)
        # 多选 Listbox
        self.ch_listbox = tk.Listbox(
            top, selectmode=tk.EXTENDED, height=5, exportselection=False
        )
        self.ch_listbox.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(top, text="保存当前图像", command=self.save_current_plot)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 多设备联合分析选项
        self.all_devices_var = tk.BooleanVar(value=False)
        self.all_devices_check = ttk.Checkbutton(
            top, text="多设备对齐分析", variable=self.all_devices_var
        )
        self.all_devices_check.pack(side=tk.LEFT, padx=5)

        self.run_btn = ttk.Button(top, text="运行分析", command=self.on_run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # 图
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        self.event_lines = []
        self.cursor = None

        # 双击添加事件（全局时间）
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def update_after_load(self):
        self.ch_listbox.delete(0, tk.END)
        for ch in self.app.channel_names:
            self.ch_listbox.insert(tk.END, ch)
        # 默认选中第一个
        if self.app.channel_names:
            self.ch_listbox.selection_set(0)
        self.redraw()

    def get_selected_channels(self):
        idxs = self.ch_listbox.curselection()
        if not idxs:
            return []
        return [self.ch_listbox.get(i) for i in idxs]

    def redraw(self):
        self.ax.clear()
        channels = self.get_selected_channels()
        if not channels:
            self.ax.set_title("请选择至少一个通道")
            self.canvas.draw()
            return

        t = self.app.time_s
        if t is None:
            self.ax.set_title("尚未加载数据")
            self.canvas.draw()
            return

        # 叠加绘制多个通道的 EMG_Clean 和 Amplitude
        for ch in channels:
            if ch not in self.app.emg_cleaned:
                continue
            emg_c = self.app.emg_cleaned[ch]
            emg_a = self.app.emg_amplitude[ch]
            if len(emg_c) != len(t):
                # 理论上不会，但保险
                N = min(len(emg_c), len(t))
                emg_c = emg_c[:N]
                emg_a = emg_a[:N]
                tt = t[:N]
            else:
                tt = t
            self.ax.plot(tt, emg_c, label=f"{ch} Clean")
            self.ax.plot(tt, emg_a, label=f"{ch} Amp", linestyle="--")

        self.ax.set_xlabel("全局时间 (秒)")
        self.ax.set_ylabel("振幅")
        self.ax.set_title(f"设备 {self.app.current_device_id} 多通道 EMG（全局对齐）")
        self.ax.legend(loc='upper right', fontsize='small')

        # 重画事件线
        for line in self.event_lines:
            line.remove()
        self.event_lines.clear()
        for ev in self.app.events:
            line = self.ax.axvline(x=ev, color='g', linestyle='--')
            self.event_lines.append(line)

        self.fig.tight_layout()
        self.canvas.draw()

        if self.cursor:
            self.cursor.disconnect()
        self.cursor = mplcursors.cursor(self.ax.lines, hover=True)
        self.cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"t = {sel.target[0]:.3f} s\n值 = {sel.target[1]:.3f}"
            )
        )

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.dblclick and event.xdata is not None:
            t = float(event.xdata)
            self.app.events.append(t)
            line = self.ax.axvline(x=t, color='g', linestyle='--')
            self.event_lines.append(line)
            self.canvas.draw()

    def save_current_plot(self):
        file_path = filedialog.asksaveasfilename(
            title="保存图像",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.fig.savefig(file_path)
            messagebox.showinfo("成功", f"图像已保存到：\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：\n{e}")

    def on_run_analysis(self):
        channels = self.get_selected_channels()
        if not channels:
            messagebox.showwarning("警告", "请先选择至少一个通道")
            return
        try:
            self.app.run_emg_event_analysis(
                channels,
                all_devices=self.all_devices_var.get()
            )
        except Exception as e:
            messagebox.showerror("错误", f"分析失败：\n{e}")


# =====================================================================
# Tab3：数据分析（5 子图 + 多设备多通道事件结果表）
# =====================================================================

class DataAnalysisTab(ttk.Frame):
    def __init__(self, app: EMGApp, master=None):
        super().__init__(master)
        self.app = app

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.export_fig_btn = ttk.Button(top, text="导出图像", command=self.export_figure)
        self.export_fig_btn.pack(side=tk.LEFT, padx=5)

        self.export_emg_btn = ttk.Button(top, text="导出 EMG 时序数据",
                                         command=lambda: self.export_data("emg"))
        self.export_emg_btn.pack(side=tk.LEFT, padx=5)

        self.export_event_btn = ttk.Button(top, text="导出事件结果",
                                           command=lambda: self.export_data("event"))
        self.export_event_btn.pack(side=tk.LEFT, padx=5)

        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=400)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # 5 子图
        self.fig, self.axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, left)
        self.toolbar.update()

        # 事件结果表
        self.tree = ttk.Treeview(right, show="headings")
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

    def update_content(self):
        emg_signals = self.app.emg_signals_result
        df_events = self.app.event_result_df
        if emg_signals is None:
            return

        for ax in self.axes:
            ax.clear()

        # 多设备 + 多通道，一起 groupby 绘图
        groups = emg_signals.groupby(["device_id", "Channel"])
        for (dev, ch), df_sub in groups:
            t = df_sub["Time_s"].values
            self.axes[0].plot(t, df_sub["EMG_Clean"], label=f"{dev}-{ch}")
            self.axes[1].plot(t, df_sub["EMG_Amplitude"], label=f"{dev}-{ch}")
            self.axes[2].plot(t, df_sub["EMG_RMS"], label=f"{dev}-{ch}")
            self.axes[3].plot(t, df_sub["EMG_Amplitude_Diff"], label=f"{dev}-{ch}")
            self.axes[4].plot(t, df_sub["EMG_MPF"], label=f"{dev}-{ch}")

        self.axes[0].set_title("Original Signal")
        self.axes[0].set_ylabel("Amplitude")

        self.axes[1].set_title("EMG Activation")
        self.axes[1].set_ylabel("Amplitude")

        self.axes[2].set_title("Root Mean Square (RMS)")
        self.axes[2].set_ylabel("Amplitude")

        self.axes[3].set_title("Amplitude Change Rate Over Time (ARF)")
        self.axes[3].set_ylabel("Change")

        self.axes[4].set_title("Mean Power Frequency Over Time (MPF)")
        self.axes[4].set_xlabel("Global Time (s)")
        self.axes[4].set_ylabel("MPF (Hz)")

        for ax in self.axes:
            ax.legend(loc="upper right", fontsize="x-small")

        self.fig.tight_layout()
        self.canvas.draw()

        # 更新表
        self.tree.delete(*self.tree.get_children())
        if df_events is not None and not df_events.empty:
            self.tree["columns"] = list(df_events.columns)
            for col in df_events.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, anchor=tk.CENTER)

            for _, row in df_events.iterrows():
                vals = [str(row[c]) for c in df_events.columns]
                self.tree.insert("", tk.END, values=vals)
        else:
            self.tree["columns"] = []
            # 如果没有事件结果，就显示空表

    def export_figure(self):
        file_path = filedialog.asksaveasfilename(
            title="保存图像",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.fig.savefig(file_path)
            messagebox.showinfo("成功", f"图像已保存到：\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：\n{e}")

    def export_data(self, which: str):
        if which == "emg":
            df = self.app.emg_signals_result
            title = "导出 EMG 时序数据"
        else:
            df = self.app.event_result_df
            title = "导出事件结果"

        if df is None or df.empty:
            messagebox.showwarning("警告", "没有可导出的数据")
            return

        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".csv",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx"),
                ("JSON Files", "*.json"),
                ("Pickle Files", "*.pkl"),
                ("All Files", "*.*"),
            ]
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith(".csv"):
                df.to_csv(file_path, index=False)
            elif file_path.lower().endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            elif file_path.lower().endswith(".json"):
                df.to_json(file_path, orient="records", lines=True)
            elif file_path.lower().endswith(".pkl"):
                df.to_pickle(file_path)
            else:
                df.to_csv(file_path, index=False)
            messagebox.showinfo("成功", f"数据已导出到：\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：\n{e}")


def main():
    app = EMGApp()
    app.mainloop()


if __name__ == "__main__":
    main()
