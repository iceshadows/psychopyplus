import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
import h5py
import scipy.signal as signal
import neurokit2 as nk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ======== 设置中文字体为 SimHei，避免负号变成方块 ========
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 全局中文字体
matplotlib.rcParams["axes.unicode_minus"] = False     # 解决坐标轴负号显示问题
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages


# ======================
# 日志
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================
# H5 -> DataFrame
# ======================
def h5_to_dataframe(h5_path: str) -> pd.DataFrame:
    """
    将 HDF5 文件转换为 pandas.DataFrame

    约定结构：
        /<device_id>/emg_data  : (n_samples, n_channels)
        /<device_id>/timestamps: (n_samples,)

    返回列：
        device_id, timestamp, ch_0, ch_1, ...
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    frames = []

    with h5py.File(h5_path, "r") as f:
        # 遍历每个设备 group
        for device_id in f.keys():
            grp = f[device_id]

            if not isinstance(grp, h5py.Group):
                continue

            if "emg_data" not in grp or "timestamps" not in grp:
                logger.warning(f"Group {device_id} missing 'emg_data' or 'timestamps', skip.")
                continue

            emg_data = grp["emg_data"][:]
            timestamps = grp["timestamps"][:]

            if emg_data.ndim != 2:
                logger.warning(f"Group {device_id} emg_data ndim={emg_data.ndim}, expect 2, skip.")
                continue

            n_samples, data_dim = emg_data.shape

            if timestamps.shape[0] != n_samples:
                logger.warning(
                    f"Group {device_id} size mismatch: emg_data={n_samples}, "
                    f"timestamps={timestamps.shape[0]}, skip."
                )
                continue

            # 构造通道列名
            channel_cols = [f"ch_{i}" for i in range(data_dim)]

            df_dev = pd.DataFrame(emg_data, columns=channel_cols)
            df_dev.insert(0, "timestamp", timestamps)
            df_dev.insert(0, "device_id", device_id)

            frames.append(df_dev)

    if not frames:
        logger.warning("No valid EMG data found in file, return empty DataFrame.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return df


# ======================
# 主 GUI 应用
# ======================
class SignalAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("同步生理信号分析 GUI Demo")
        self.geometry("1200x800")

        # 数据相关
        self.df: pd.DataFrame | None = None
        self.fs = None  # 采样率
        self.current_device = None

        # 当前通道 & 信号类型
        self.current_channel_var = tk.StringVar(value="ch_0")
        self.signal_type_var = tk.StringVar(value="EMG")  # EMG / ECG / EEG

        # 患者信息
        self.patient_name_var = tk.StringVar()
        self.patient_id_var = tk.StringVar()
        self.patient_age_var = tk.StringVar()
        self.patient_sex_var = tk.StringVar()
        self.patient_note_var = tk.StringVar()

        # 时间段截取（单位：秒）
        self.segment_start_var = tk.DoubleVar(value=0.0)
        self.segment_end_var = tk.DoubleVar(value=0.0)  # 0 表示“到末尾”

        # 滤波器列表
        # 每个元素为 dict: {"type": "lowpass/highpass/bandpass", "order": int, "f1": float, "f2": float | None}
        self.filters = []

        # 滤波后信号缓存
        self.filtered_signal = None
        self.filtered_time = None
        self.filtered_fs = None

        # 特征缓存（用于导出报告）
        self.last_features = {}

        self._build_layout()

    # ======================
    # GUI 布局
    # ======================
    def _build_layout(self):
        # 顶部：患者信息 + 数据导入
        top_frame = ttk.LabelFrame(self, text="患者信息 & 数据导入")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 患者信息
        ttk.Label(top_frame, text="姓名:").grid(row=0, column=0, sticky="e")
        ttk.Entry(top_frame, textvariable=self.patient_name_var, width=15).grid(row=0, column=1, padx=2)

        ttk.Label(top_frame, text="ID:").grid(row=0, column=2, sticky="e")
        ttk.Entry(top_frame, textvariable=self.patient_id_var, width=15).grid(row=0, column=3, padx=2)

        ttk.Label(top_frame, text="年龄:").grid(row=0, column=4, sticky="e")
        ttk.Entry(top_frame, textvariable=self.patient_age_var, width=8).grid(row=0, column=5, padx=2)

        ttk.Label(top_frame, text="性别:").grid(row=0, column=6, sticky="e")
        ttk.Entry(top_frame, textvariable=self.patient_sex_var, width=8).grid(row=0, column=7, padx=2)

        ttk.Label(top_frame, text="备注:").grid(row=0, column=8, sticky="e")
        ttk.Entry(top_frame, textvariable=self.patient_note_var, width=30).grid(row=0, column=9, padx=2)

        # 第二行：信号类型 / 采样率 / 打开文件 / device / channel
        ttk.Label(top_frame, text="信号类型:").grid(row=1, column=0, sticky="e", pady=3)
        tt = ttk.Combobox(top_frame, textvariable=self.signal_type_var, values=["EMG", "ECG", "EEG"], width=10, state="readonly")
        tt.grid(row=1, column=1, padx=2)

        ttk.Label(top_frame, text="采样率 (Hz):").grid(row=1, column=2, sticky="e")
        self.fs_entry = ttk.Entry(top_frame, width=10)
        self.fs_entry.grid(row=1, column=3, padx=2)

        ttk.Button(top_frame, text="打开 HDF5 文件", command=self.load_h5_file).grid(row=1, column=4, padx=5)

        ttk.Label(top_frame, text="设备 ID:").grid(row=1, column=5, sticky="e")
        self.device_combo = ttk.Combobox(top_frame, values=[], state="readonly", width=18)
        self.device_combo.grid(row=1, column=6, padx=2)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        ttk.Label(top_frame, text="通道:").grid(row=1, column=7, sticky="e")
        self.channel_combo = ttk.Combobox(top_frame, textvariable=self.current_channel_var, values=[], state="readonly", width=10)
        self.channel_combo.grid(row=1, column=8, padx=2)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_change)

        ttk.Button(top_frame, text="更新波形", command=self.update_plots).grid(row=1, column=9, padx=5)

        # 中间：左侧控制，右侧图形
        middle_frame = ttk.Frame(self)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧控制
        control_frame = ttk.Frame(middle_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 滤波器列表
        filter_frame = ttk.LabelFrame(control_frame, text="滤波器列表（可叠加）")
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="类型:").grid(row=0, column=0, sticky="e")
        self.filter_type_var = tk.StringVar(value="bandpass")
        ft_combo = ttk.Combobox(filter_frame, textvariable=self.filter_type_var,
                                values=["lowpass", "highpass", "bandpass"], state="readonly", width=10)
        ft_combo.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(filter_frame, text="f1 (Hz):").grid(row=1, column=0, sticky="e")
        self.filter_f1_var = tk.StringVar(value="20.0")
        ttk.Entry(filter_frame, textvariable=self.filter_f1_var, width=8).grid(row=1, column=1, padx=2, pady=2)

        ttk.Label(filter_frame, text="f2 (Hz):").grid(row=1, column=2, sticky="e")
        self.filter_f2_var = tk.StringVar(value="450.0")
        ttk.Entry(filter_frame, textvariable=self.filter_f2_var, width=8).grid(row=1, column=3, padx=2, pady=2)

        ttk.Label(filter_frame, text="阶数:").grid(row=0, column=2, sticky="e")
        self.filter_order_var = tk.StringVar(value="4")
        ttk.Entry(filter_frame, textvariable=self.filter_order_var, width=5).grid(row=0, column=3, padx=2, pady=2)

        ttk.Button(filter_frame, text="添加滤波器", command=self.add_filter).grid(row=0, column=4, padx=5)
        ttk.Button(filter_frame, text="删除选中", command=self.remove_filter).grid(row=1, column=4, padx=5)

        self.filter_listbox = tk.Listbox(filter_frame, height=6, width=50)
        self.filter_listbox.grid(row=2, column=0, columnspan=5, sticky="we", padx=2, pady=2)

        ttk.Button(filter_frame, text="应用滤波", command=self.apply_filters).grid(row=3, column=0, columnspan=5, pady=3)

        # 数据截取
        segment_frame = ttk.LabelFrame(control_frame, text="数据截取（秒）")
        segment_frame.pack(fill=tk.X, pady=5)

        ttk.Label(segment_frame, text="起始 t_start:").grid(row=0, column=0, sticky="e")
        ttk.Entry(segment_frame, textvariable=self.segment_start_var, width=10).grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(segment_frame, text="结束 t_end:").grid(row=1, column=0, sticky="e")
        ttk.Entry(segment_frame, textvariable=self.segment_end_var, width=10).grid(row=1, column=1, padx=2, pady=2)

        ttk.Label(segment_frame, text="(0 或空 = 到末尾)").grid(row=1, column=2, padx=2)

        ttk.Button(segment_frame, text="应用时间段", command=self.apply_segment).grid(row=2, column=0, columnspan=3, pady=3)

        # 特征
        feature_ctrl_frame = ttk.LabelFrame(control_frame, text="特征 & 频谱")
        feature_ctrl_frame.pack(fill=tk.X, pady=5)

        ttk.Button(feature_ctrl_frame, text="计算特征 & 更新功率谱", command=self.compute_features).pack(padx=5, pady=5, fill=tk.X)

        # 右侧绘图区域
        plot_frame = ttk.Frame(middle_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_time = self.fig.add_subplot(211)
        self.ax_freq = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 底部：特征文本 & 导出 PDF
        bottom_frame = ttk.LabelFrame(self, text="特征结果 & 报告")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.feature_text = ScrolledText(bottom_frame, height=8)
        self.feature_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_bottom_frame = ttk.Frame(bottom_frame)
        right_bottom_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        ttk.Button(right_bottom_frame, text="导出 PDF 报告", command=self.export_pdf).pack(fill=tk.X, pady=5)

    # ======================
    # 事件处理 / 工具函数
    # ======================
    def load_h5_file(self):
        path = filedialog.askopenfilename(
            title="选择 HDF5 数据文件",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            df = h5_to_dataframe(path)
        except Exception as e:
            messagebox.showerror("错误", f"读取 HDF5 失败:\n{e}")
            return

        if df.empty:
            messagebox.showwarning("提示", "HDF5 中未找到合法 EMG 数据。")
            return

        self.df = df
        device_ids = sorted(df["device_id"].unique().tolist())
        self.device_combo["values"] = device_ids
        self.device_combo.set(device_ids[0])
        self.current_device = device_ids[0]

        # 更新通道列表
        self.update_channel_list()
        self.filtered_signal = None

        messagebox.showinfo("成功", f"成功加载 HDF5 数据，共 {len(df)} 行。")
        self.update_plots()

    def update_channel_list(self):
        if self.df is None:
            return
        # 所有列中以 ch_ 开头的
        channel_cols = [c for c in self.df.columns if c.startswith("ch_")]
        if not channel_cols:
            channel_cols = []
        self.channel_combo["values"] = channel_cols
        if channel_cols:
            self.channel_combo.set(channel_cols[0])
            self.current_channel_var.set(channel_cols[0])

    def on_device_change(self, event=None):
        self.current_device = self.device_combo.get()
        self.filtered_signal = None
        self.update_plots()

    def on_channel_change(self, event=None):
        self.filtered_signal = None
        self.update_plots()

    def _get_fs(self):
        # 解析采样率
        txt = self.fs_entry.get().strip()
        if not txt:
            raise ValueError("采样率为空")
        fs = float(txt)
        if fs <= 0:
            raise ValueError("采样率必须为正数")
        return fs

    def get_current_signal(self):
        """
        从当前 DataFrame 中按照 device_id, channel & 时间段取出信号段。
        返回: (signal_1d, time_axis, fs)
        """
        if self.df is None:
            messagebox.showwarning("提示", "请先加载 HDF5 数据。")
            return None, None, None
        if self.current_device is None:
            messagebox.showwarning("提示", "请先选择设备 ID。")
            return None, None, None

        try:
            fs = self._get_fs()
            self.fs = fs
        except Exception as e:
            messagebox.showerror("错误", f"采样率错误: {e}")
            return None, None, None

        dev_df = self.df[self.df["device_id"] == self.current_device].copy()
        if dev_df.empty:
            messagebox.showerror("错误", "当前设备 ID 无数据。")
            return None, None, None

        dev_df = dev_df.sort_values("timestamp")
        ch = self.current_channel_var.get()
        if ch not in dev_df.columns:
            messagebox.showerror("错误", f"通道 {ch} 不存在。")
            return None, None, None

        y = dev_df[ch].values.astype(float)
        n = len(y)
        t = np.arange(n) / fs  # 用采样率构造时间轴

        # 时间段截取
        t_start = self.segment_start_var.get()
        t_end = self.segment_end_var.get()
        if t_start < 0:
            t_start = 0
        if t_end <= 0 or t_end > t[-1]:
            t_end = t[-1]

        mask = (t >= t_start) & (t <= t_end)
        if not mask.any():
            # 如果时间段非法，则使用全段
            logger.warning("时间段选择为空，使用全段数据。")
            t_start, t_end = t[0], t[-1]
            mask = (t >= t_start) & (t <= t_end)

        t_seg = t[mask]
        y_seg = y[mask]

        return y_seg, t_seg, fs

    # ======================
    # 滤波器列表
    # ======================
    def add_filter(self):
        ftype = self.filter_type_var.get()
        try:
            order = int(self.filter_order_var.get())
            if order <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("错误", "滤波器阶数必须为正整数。")
            return

        try:
            f1 = float(self.filter_f1_var.get())
        except Exception:
            messagebox.showerror("错误", "f1 必须为数字。")
            return

        f2 = None
        if ftype == "bandpass":
            try:
                f2 = float(self.filter_f2_var.get())
            except Exception:
                messagebox.showerror("错误", "带通滤波器需要合法的 f2。")
                return
            if f2 <= f1:
                messagebox.showerror("错误", "带通滤波器要求 f2 > f1。")
                return

        filt = {"type": ftype, "order": order, "f1": f1, "f2": f2}
        self.filters.append(filt)
        self.refresh_filter_listbox()

    def refresh_filter_listbox(self):
        self.filter_listbox.delete(0, tk.END)
        for idx, f in enumerate(self.filters):
            if f["type"] == "bandpass":
                txt = f"{idx+1}. {f['type']} {f['f1']}~{f['f2']} Hz, n={f['order']}"
            else:
                txt = f"{idx+1}. {f['type']} fc={f['f1']} Hz, n={f['order']}"
            self.filter_listbox.insert(tk.END, txt)

    def remove_filter(self):
        sel = self.filter_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.filters):
            self.filters.pop(idx)
            self.refresh_filter_listbox()

    def apply_filters(self):
        y, t, fs = self.get_current_signal()
        if y is None:
            return

        if not self.filters:
            messagebox.showinfo("提示", "滤波器列表为空，将不进行滤波。")
            self.filtered_signal = None
            self.update_plots()
            return

        nyq = 0.5 * fs
        y_filt = y.copy()

        for f in self.filters:
            try:
                if f["type"] == "lowpass":
                    Wn = f["f1"] / nyq
                    b, a = signal.butter(f["order"], Wn, btype="low")
                elif f["type"] == "highpass":
                    Wn = f["f1"] / nyq
                    b, a = signal.butter(f["order"], Wn, btype="high")
                elif f["type"] == "bandpass":
                    Wn = [f["f1"] / nyq, f["f2"] / nyq]
                    b, a = signal.butter(f["order"], Wn, btype="band")
                else:
                    continue
                y_filt = signal.filtfilt(b, a, y_filt)
            except Exception as e:
                messagebox.showerror("错误", f"应用滤波器失败: {e}")
                return

        self.filtered_signal = y_filt
        self.filtered_time = t
        self.filtered_fs = fs

        self.update_plots()

    def apply_segment(self):
        # 修改时间段，清除滤波缓存
        self.filtered_signal = None
        self.update_plots()

    # ======================
    # 绘图
    # ======================
    def update_plots(self):
        y, t, fs = self.get_current_signal()
        if y is None:
            return

        # 使用滤波后的信号（如果长度匹配），否则原始
        if self.filtered_signal is not None and len(self.filtered_signal) == len(y):
            y_plot_raw = y
            y_plot = self.filtered_signal
            use_filtered = True
        else:
            y_plot_raw = y
            y_plot = y
            use_filtered = False

        self.ax_time.clear()
        self.ax_freq.clear()

        if use_filtered:
            self.ax_time.plot(t, y_plot_raw, color="gray", alpha=0.4, label="原始")
            self.ax_time.plot(t, y_plot, color="C0", label="滤波后")
            self.ax_time.legend(loc="upper right", fontsize=8)
        else:
            self.ax_time.plot(t, y_plot, color="C0", label="信号")
            self.ax_time.legend(loc="upper right", fontsize=8)

        self.ax_time.set_xlabel("时间 (s)")
        self.ax_time.set_ylabel("幅值")
        ch = self.current_channel_var.get()
        self.ax_time.set_title(f"时间域波形 - 设备 {self.current_device}, 通道 {ch}")

        # Welch 功率谱
        if len(y_plot) > 10:
            f, Pxx = signal.welch(y_plot, fs=fs, nperseg=min(2048, len(y_plot)))
            self.ax_freq.semilogy(f, Pxx)
            self.ax_freq.set_xlabel("频率 (Hz)")
            self.ax_freq.set_ylabel("功率谱密度")
            self.ax_freq.set_title("Welch 功率谱")

        self.fig.tight_layout()
        self.canvas.draw()

    # ======================
    # 特征计算
    # ======================
    def compute_features(self):
        y, t, fs = self.get_current_signal()
        if y is None:
            return

        if self.filtered_signal is not None and len(self.filtered_signal) == len(y):
            sig = self.filtered_signal
        else:
            sig = y

        if len(sig) < 5:
            messagebox.showwarning("提示", "数据长度太短，无法计算特征。")
            return

        features = {}
        duration = t[-1] - t[0]
        features["采样率(Hz)"] = fs
        features["时长(s)"] = duration
        features["均值"] = float(np.mean(sig))
        features["标准差"] = float(np.std(sig))
        features["RMS"] = float(np.sqrt(np.mean(sig ** 2)))
        features["峰-峰值"] = float(np.max(sig) - np.min(sig))

        # 功率谱特征
        f, Pxx = signal.welch(sig, fs=fs, nperseg=min(2048, len(sig)))
        total_power = float(np.trapz(Pxx, f))
        features["总功率"] = total_power

        signal_type = self.signal_type_var.get().upper()

        # EEG：经典频带功率
        if signal_type == "EEG":
            bands = {
                "delta(0.5-4Hz)": (0.5, 4),
                "theta(4-8Hz)": (4, 8),
                "alpha(8-13Hz)": (8, 13),
                "beta(13-30Hz)": (13, 30),
                "gamma(30-45Hz)": (30, 45)
            }
            for name, (f1, f2) in bands.items():
                band_mask = (f >= f1) & (f <= f2)
                if band_mask.any():
                    bp = float(np.trapz(Pxx[band_mask], f[band_mask]))
                else:
                    bp = 0.0
                features[f"功率_{name}"] = bp

        # ECG：用 neurokit2 算心率等
        if signal_type == "ECG":
            try:
                signals, info = nk.ecg_process(sig, sampling_rate=fs)
                hr_mean = float(signals["ECG_Rate"].mean())
                hr_std = float(signals["ECG_Rate"].std())
                features["平均心率(bpm)"] = hr_mean
                features["心率标准差(bpm)"] = hr_std

                # 进一步 HRV 指标（如果可用）
                try:
                    hrv = nk.hrv(info, sampling_rate=fs, show=False)
                    if "HRV_SDNN" in hrv.columns:
                        features["HRV_SDNN(ms)"] = float(hrv["HRV_SDNN"].iloc[0])
                    if "HRV_RMSSD" in hrv.columns:
                        features["HRV_RMSSD(ms)"] = float(hrv["HRV_RMSSD"].iloc[0])
                except Exception as e:
                    logger.warning(f"HRV 计算失败: {e}")
            except Exception as e:
                logger.warning(f"ECG 特征计算失败: {e}")

        # EMG：简单指标，可按需扩展（包络、肌电积分等）
        if signal_type == "EMG":
            # 绝对值积分（简易 IEMG）
            features["IEMG(∑|x|)"] = float(np.sum(np.abs(sig)))

        # 更新文本框
        self.feature_text.delete("1.0", tk.END)
        self.feature_text.insert(tk.END, f"信号类型: {signal_type}\n")
        self.feature_text.insert(tk.END, f"设备: {self.current_device}, 通道: {self.current_channel_var.get()}\n\n")

        for k, v in features.items():
            # 数字统一格式一下
            if isinstance(v, float):
                self.feature_text.insert(tk.END, f"{k}: {v:.4g}\n")
            else:
                self.feature_text.insert(tk.END, f"{k}: {v}\n")

        self.last_features = features
        # 顺便刷新一下功率谱（这里已经计算过 welch，可以复用，但为简洁起见就直接让 update_plots 再画一遍）
        self.update_plots()

    # ======================
    # 导出 PDF 报告
    # ======================
    def export_pdf(self):
        if self.df is None:
            messagebox.showwarning("提示", "没有数据，无法导出报告。")
            return

        # 确保有最新特征
        self.compute_features()

        path = filedialog.asksaveasfilename(
            title="导出 PDF 报告",
            defaultextension=".pdf",
            filetypes=[("PDF 文件", "*.pdf")]
        )
        if not path:
            return

        try:
            with PdfPages(path) as pdf:
                # 第一页：波形+功率谱（当前 matplotlib Figure）
                pdf.savefig(self.fig)

                # 第二页：文字报告（用一个纯文本 Figure 实现）
                fig_report = plt.figure(figsize=(8.27, 11.69))  # A4 纵向
                fig_report.clf()

                y = 0.95
                line_space = 0.03

                def add_line(text, fontsize=10, weight=None):
                    nonlocal y
                    fig_report.text(0.1, y, text, fontsize=fontsize,
                                    weight=weight if weight else "normal")
                    y -= line_space

                add_line("实验报告", fontsize=16, weight="bold")
                y -= 0.02

                add_line("【患者信息】", weight="bold")
                add_line(f"姓名: {self.patient_name_var.get()}")
                add_line(f"ID: {self.patient_id_var.get()}")
                add_line(f"年龄: {self.patient_age_var.get()}")
                add_line(f"性别: {self.patient_sex_var.get()}")
                add_line(f"备注: {self.patient_note_var.get()}")
                y -= 0.02

                add_line("【数据设定】", weight="bold")
                add_line(f"信号类型: {self.signal_type_var.get()}")
                add_line(f"采样率(Hz): {self.fs_entry.get()}")
                add_line(f"设备 ID: {self.current_device}")
                add_line(f"通道: {self.current_channel_var.get()}")
                add_line(f"时间段: {self.segment_start_var.get()}s  ~  {self.segment_end_var.get()}s (0=到末尾)")
                y -= 0.02

                add_line("【滤波配置】", weight="bold")
                if not self.filters:
                    add_line("未使用数字滤波。")
                else:
                    for idx, f in enumerate(self.filters):
                        if f["type"] == "bandpass":
                            txt = f"{idx+1}. {f['type']} {f['f1']}~{f['f2']} Hz, n={f['order']}"
                        else:
                            txt = f"{idx+1}. {f['type']} fc={f['f1']} Hz, n={f['order']}"
                        add_line(txt)
                y -= 0.02

                add_line("【特征结果】", weight="bold")
                for k, v in self.last_features.items():
                    if isinstance(v, float):
                        add_line(f"{k}: {v:.4g}")
                    else:
                        add_line(f"{k}: {v}")

                pdf.savefig(fig_report)
                plt.close(fig_report)

            messagebox.showinfo("成功", f"PDF 报告已导出到:\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出 PDF 失败:\n{e}")


if __name__ == "__main__":
    app = SignalAnalysisApp()
    app.mainloop()
