#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Session management for WearLab EMG recording in PsychoPy.

This module contains:

- EMGLivePlotBuffer
    Thread-safe circular buffer for live visualization.

- EMGLivePlotTkWindow
    Tk-based live visualization window for EMG data.

- EMGHDF5Recorder
    Handles writing EMG samples, timestamps, device info and events to HDF5.

- EMGDataHandler
    Receives parsed WearLab protocol packets and forwards them to recorders
    and (optionally) to the live buffer.

- WearLabEMGSession
    High-level controller used by PsychoPy-generated scripts. It:
    - starts DeviceCommunication
    - wires callbacks
    - exposes a log_event(...) method to record PsychoPy events
    - optionally opens a separate Tk window for EMG live visualization.
"""

import os
import time
import math
import colorsys
import logging
import threading
from collections import deque
from typing import Dict, Optional, Deque, Any, List

import tkinter as tk

import h5py
import numpy as np

from wearlab_protocolX.meta import (
    PackageType,
    EmgDataPacket,
    HardwareDiscoveryPacket,
)
from wearlab_protocolX.wearlab_device import DeviceCommunication, WearLabDevice


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Live buffer for visualization
# --------------------------------------------------------------------------- #

class EMGLivePlotBuffer:
    """Thread-safe circular buffer for EMG live preview."""

    def __init__(self, channels: int, window_size: int = 1000) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of channels per EMG frame.
        window_size : int
            Maximum number of time points kept in the buffer.
        """
        self.channels = channels
        self.window_size = window_size
        self.buffer = np.zeros((window_size, channels), dtype=np.float32)
        self.index = 0
        self.length = 0
        self._lock = threading.Lock()

    def append(self, emg_frame: np.ndarray) -> None:
        """
        Append a new EMG frame to the circular buffer.

        Parameters
        ----------
        emg_frame : np.ndarray
            Shape (channels,). Frames with mismatched shape are ignored.
        """
        if emg_frame.shape[0] != self.channels:
            return
        with self._lock:
            self.buffer[self.index] = emg_frame
            self.index = (self.index + 1) % self.window_size
            self.length = min(self.length + 1, self.window_size)

    def get_window(self) -> Optional[np.ndarray]:
        """
        Return data in temporal order: shape (L, channels), L <= window_size.

        Returns
        -------
        np.ndarray or None
            Copy of the live window, or None if buffer is empty.
        """
        with self._lock:
            if self.length == 0:
                return None
            if self.length < self.window_size:
                return self.buffer[: self.length].copy()
            # Unroll the circular buffer
            idx = self.index
            return np.vstack((self.buffer[idx:], self.buffer[:idx])).copy()


# --------------------------------------------------------------------------- #
# Tk live visualization window (multi-channel, vertically separated)
# --------------------------------------------------------------------------- #

class EMGLivePlotTkWindow(threading.Thread):
    """
    Tk-based EMG live preview window.

    This window runs in its own thread and periodically pulls a snapshot
    from EMGLivePlotBuffer to draw multi-channel waveforms.

    Visualization style:
    - Each channel is drawn in its own horizontal "track", vertically stacked.
    - Tracks are evenly distributed from top to bottom with small margins.
    - Each channel has its own color (HSV wheel).
    - Vertical scale can be adjusted via Up/Down keys.

    Performance notes:
    - Only the last `max_points` samples are drawn.
    - All channels share the same X coordinates.
    """

    def __init__(
        self,
        live_buffer: EMGLivePlotBuffer,
        update_interval_ms: int = 50,
        max_points: int = 800,
        title: str = "WearLab EMG Live",
        line_width: int = 1,
        channels_to_show: Optional[List[int]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.live_buffer = live_buffer
        self.update_interval_ms = update_interval_ms
        self.max_points = max_points
        self.title = title
        self.line_width = line_width

        # Which channels to display. If None, display all.
        if channels_to_show is None:
            channels_to_show = list(range(self.live_buffer.channels))
        self.channels_to_show = channels_to_show

        self._running = threading.Event()
        self._running.set()

        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.width = 1000
        self.height = 600

        # Global vertical scaling and offset (applied to all channels)
        self.y_scale = 1.0
        self.y_offset = 0.0

        # Pre-generate one color per channel to show
        self.channel_colors = self._generate_channel_colors(len(self.channels_to_show))

    def _generate_channel_colors(self, n_channels: int) -> list[str]:
        """
        Generate a distinct RGB color (in hex string) for each channel
        by evenly sampling the HSV color wheel.
        """
        colors: list[str] = []
        if n_channels <= 0:
            return ["#00FF00"]
        for i in range(n_channels):
            h = float(i) / float(n_channels)
            s = 0.8
            v = 0.9
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
        return colors

    # ------------------------------ #
    # Thread entry
    # ------------------------------ #

    def run(self) -> None:
        """Thread entry: create Tk root and start main loop."""
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Update stored width/height when the window is resized
        self.canvas.bind("<Configure>", self._on_resize)

        # Bind keys for vertical scaling
        self.root.bind("<Up>", self._on_key_up)
        self.root.bind("<Down>", self._on_key_down)

        # Start periodic updates
        self._schedule_update()

        self.root.mainloop()

    def _on_resize(self, event: tk.Event) -> None:
        """Update internal width/height when the canvas is resized."""
        self.width = max(100, int(event.width))
        self.height = max(100, int(event.height))

    def _on_close(self) -> None:
        """
        Callback when user closes the window.

        This will stop the update loop and destroy the Tk root.
        """
        self._running.clear()
        if self.root is not None:
            self.root.after(0, self.root.destroy)

    def _on_key_up(self, event: tk.Event) -> None:
        """Increase global vertical scale."""
        self.y_scale *= 1.2

    def _on_key_down(self, event: tk.Event) -> None:
        """Decrease global vertical scale (with lower bound)."""
        self.y_scale /= 1.2
        if self.y_scale < 0.1:
            self.y_scale = 0.1

    # ------------------------------ #
    # External stop API
    # ------------------------------ #

    def stop(self) -> None:
        """
        Request the live window to stop and close.

        Safe to call multiple times and from other threads.
        """
        self._running.clear()
        if self.root is not None:
            try:
                self.root.after(0, self.root.destroy)
            except Exception:
                # Window might already be closed
                pass

    # ------------------------------ #
    # Drawing logic
    # ------------------------------ #

    def _schedule_update(self) -> None:
        """Schedule the next update if still running."""
        if self.root is None:
            return
        if not self._running.is_set():
            return
        self._update_plot()
        self.root.after(self.update_interval_ms, self._schedule_update)

    def _update_plot(self) -> None:
        """
        Fetch latest window from buffer and redraw.

        Only the last `max_points` samples are used to limit the workload.
        """
        if self.canvas is None:
            return

        window = self.live_buffer.get_window()
        if window is None or window.shape[0] < 2:
            # No data: clear canvas
            self.canvas.delete("emg")
            return

        # Use only the last max_points samples for performance
        num_samples = window.shape[0]
        if num_samples > self.max_points:
            step = math.ceil(num_samples / self.max_points)
            window = window[::step, :]

        # Ensure at least 2 points remain
        if window.shape[0] < 2:
            self.canvas.delete("emg")
            return

        self._draw_waveforms(window)

    def _draw_waveforms(self, window: np.ndarray) -> None:
        """
        Draw multi-channel EMG waveforms with vertical separation.

        Each channel is drawn in its own horizontal band.

        Parameters
        ----------
        window : np.ndarray
            Shape (L, C) where L is time and C is channels.
        """
        if self.canvas is None:
            return

        self.canvas.delete("emg")

        num_points, num_channels = window.shape
        if num_points < 2 or num_channels == 0:
            return

        w = float(self.width)
        h = float(self.height)

        # Current canvas size may be zero before first layout
        if w <= 1.0 or h <= 1.0:
            return

        # Margins at top and bottom
        margin_top = h * 0.05
        margin_bottom = h * 0.05
        available_h = h - margin_top - margin_bottom
        if available_h <= 0:
            return

        # Determine how many channels we actually show
        ch_indices = [ch for ch in self.channels_to_show if ch < num_channels]
        ch_count = len(ch_indices)
        if ch_count == 0:
            return

        # Height per channel track
        per_ch_height = available_h / float(ch_count)

        # Pre-compute X coordinates (shared by all channels)
        # X goes from 0 to w with num_points samples
        xs = np.linspace(0.0, w, num_points, dtype=np.float32)

        # Draw each channel in its own track
        for idx, ch in enumerate(ch_indices):
            ch_data = window[:, ch]

            # Auto-range for this channel
            dmin = float(np.min(ch_data))
            dmax = float(np.max(ch_data))
            if dmax == dmin:
                dmax = dmin + 1.0

            # Center line for this channel's track
            center_line = margin_top + per_ch_height * (idx + 0.5)

            # Map dmin..dmax into 80% of track height
            scale = (per_ch_height * 0.4 / (dmax - dmin)) * self.y_scale

            # Midpoint of the data range
            mid = 0.5 * (dmin + dmax)

            # Compute Y coordinates: center_line - (value - mid) * scale
            ys = center_line - (ch_data - mid) * scale + self.y_offset

            # Interleave X/Y into a flat list
            points: List[float] = []
            for x, y in zip(xs, ys):
                points.append(float(x))
                points.append(float(y))

            color = self.channel_colors[idx % len(self.channel_colors)]
            self.canvas.create_line(
                *points,
                fill=color,
                width=self.line_width,
                tags="emg",
                smooth=False,
            )


# --------------------------------------------------------------------------- #
# HDF5 recorder
# --------------------------------------------------------------------------- #

class EMGHDF5Recorder:
    """Recorder that writes EMG data and events for a single device into HDF5."""

    def __init__(
        self,
        filename: str,
        device_id: str,
        channels_size: int = 64,
        buffer_size: int = 1000,
    ) -> None:
        """
        Initialize the recorder.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.
        device_id : str
            Hardware identifier of the device.
        channels_size : int
            Number of EMG channels per sample.
        buffer_size : int
            Max number of samples/events stored in memory before flushing.
        """
        self.filename = filename
        self.device_id = device_id
        self.channels_size = channels_size
        self.buffer_size = buffer_size

        self.data_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.timestamps_buffer: Deque[int] = deque(maxlen=buffer_size)
        self.events_buffer: Deque[np.void] = deque(maxlen=buffer_size)

        self.is_recording: bool = False
        self.file_handle: Optional[h5py.File] = None

        self.log = logging.getLogger(f"{__name__}.EMGHDF5Recorder")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start_recording(self, device_info: Optional[dict] = None) -> None:
        """Open the HDF5 file and create datasets if needed."""
        if self.is_recording:
            self.log.warning("Recording already in progress.")
            return

        try:
            # Ensure the parent directory exists, e.g. "data/..."
            dir_path = os.path.dirname(self.filename)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            # Open in append mode so multiple runs or devices can share a file
            self.file_handle = h5py.File(self.filename, "a")

            # Create or get device group
            if self.device_id in self.file_handle:
                device_group = self.file_handle[self.device_id]
            else:
                device_group = self.file_handle.create_group(self.device_id)

            # Store device info as attributes
            if device_info:
                info_group = device_group.require_group("device_info")
                for key, value in device_info.items():
                    # Use attributes for simple metadata
                    info_group.attrs[key] = value

            # Create EMG dataset
            if "emg_data" not in device_group:
                device_group.create_dataset(
                    "emg_data",
                    shape=(0, self.channels_size),
                    maxshape=(None, self.channels_size),
                    dtype=np.float32,
                    chunks=True,
                    compression="gzip",
                )

            # Create timestamps dataset
            if "timestamps" not in device_group:
                device_group.create_dataset(
                    "timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=True,
                    compression="gzip",
                )

            # Create events dataset (structured array)
            if "events" not in device_group:
                dt = np.dtype(
                    [
                        ("time", "f8"),   # PsychoPy time (seconds from globalClock)
                        ("name", "S64"),  # Event name (e.g., "trial_begin")
                        ("info", "S512"), # JSON-encoded extra info
                    ]
                )
                device_group.create_dataset(
                    "events",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dt,
                    chunks=True,
                    compression="gzip",
                )

            self.is_recording = True
            self.log.info(
                "Started recording for device %s to %s",
                self.device_id,
                self.filename,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("Failed to start recording: %s", exc)
            self.file_handle = None
            self.is_recording = False
            raise

    def stop_recording(self) -> None:
        """Flush any remaining buffers and close the HDF5 file."""
        if not self.is_recording:
            return

        # Flush remaining data and events
        self._flush_data_buffer()
        self._flush_events_buffer()

        # Close file
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        self.is_recording = False
        self.log.info("Stopped recording for device %s", self.device_id)
        self.log.info("File saved to %s", self.filename)

    # ------------------------------------------------------------------ #
    # EMG samples
    # ------------------------------------------------------------------ #

    def add_data(self, emg_packet: EmgDataPacket) -> None:
        """Add a single EMG packet to the buffers."""
        if not self.is_recording:
            self.log.debug("Recording not started; ignoring EMG data packet.")
            return

        # Append EMG samples and timestamps
        self.data_buffer.append(np.asarray(emg_packet.emg_data, dtype=np.float32))
        self.timestamps_buffer.append(int(emg_packet.pack_time))

        if len(self.data_buffer) >= self.buffer_size:
            self._flush_data_buffer()

    def _flush_data_buffer(self) -> None:
        """Write buffered EMG samples to file."""
        if not self.is_recording or self.file_handle is None:
            return

        num_points = len(self.data_buffer)
        if num_points == 0:
            return

        try:
            device_group = self.file_handle[self.device_id]
            emg_dataset = device_group["emg_data"]
            ts_dataset = device_group["timestamps"]

            current_size = emg_dataset.shape[0]
            new_size = current_size + num_points

            emg_dataset.resize((new_size, self.channels_size))
            ts_dataset.resize((new_size,))

            emg_dataset[current_size:new_size, :] = np.stack(self.data_buffer)
            ts_dataset[current_size:new_size] = np.array(self.timestamps_buffer)

            self.data_buffer.clear()
            self.timestamps_buffer.clear()

            self.file_handle.flush()
            self.log.debug("Flushed %d EMG samples to file", num_points)
        except Exception as exc:  # noqa: BLE001
            self.log.error("Failed to flush EMG buffer: %s", exc)

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def log_event(self, time_sec: float, name: str, info_json: str) -> None:
        """
        Add an event to the event buffer.

        Parameters
        ----------
        time_sec : float
            PsychoPy time (e.g., globalClock.getTime()).
        name : str
            Short event label.
        info_json : str
            JSON-encoded string with additional event information.
        """
        if not self.is_recording:
            return

        # Encode as fixed-length bytes
        name_bytes = name.encode("utf-8")[:64]
        info_bytes = info_json.encode("utf-8")[:512]

        # The actual dtype is defined in start_recording
        event = np.array(
            (float(time_sec), name_bytes, info_bytes),
            dtype=[
                ("time", "f8"),
                ("name", "S64"),
                ("info", "S512"),
            ],
        )
        self.events_buffer.append(event[()])  # store scalar np.void

        if len(self.events_buffer) >= self.buffer_size:
            self._flush_events_buffer()

    def _flush_events_buffer(self) -> None:
        """Write buffered events to file."""
        if not self.is_recording or self.file_handle is None:
            return
        if not self.events_buffer:
            return

        try:
            device_group = self.file_handle[self.device_id]
            ev_ds = device_group["events"]

            num = len(self.events_buffer)
            current = ev_ds.shape[0]
            new_size = current + num

            ev_ds.resize((new_size,))
            ev_ds[current:new_size] = np.array(self.events_buffer, dtype=ev_ds.dtype)

            self.events_buffer.clear()
            self.file_handle.flush()
            self.log.debug("Flushed %d events to file", num)
        except Exception as exc:  # noqa: BLE001
            self.log.error("Failed to flush event buffer: %s", exc)


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #

class EMGDataHandler:
    """Handle WearLab protocol packets and dispatch them to HDF5 recorders."""

    def __init__(self, base_filename: str, live_buffer: Optional[EMGLivePlotBuffer] = None) -> None:
        """
        Parameters
        ----------
        base_filename : str
            Base filename (without extension). The final HDF5 filename is
            `base_filename + ".h5"`.
        live_buffer : EMGLivePlotBuffer, optional
            Optional buffer for live visualization.
        """
        self.base_filename = base_filename
        self.recorders: Dict[str, EMGHDF5Recorder] = {}
        self.device_info: Dict[str, dict] = {}
        self.live_buffer = live_buffer
        self.log = logging.getLogger(f"{__name__}.EMGDataHandler")

    # ------------------------------------------------------------------ #
    # Packet dispatch
    # ------------------------------------------------------------------ #

    def data_callback(self, data_packet: Any) -> None:
        """Entry point for DeviceCommunication data callbacks."""
        try:
            if isinstance(data_packet, EmgDataPacket):
                self._handle_emg_data(data_packet)
            elif isinstance(data_packet, HardwareDiscoveryPacket):
                self._handle_device_info(data_packet)
        except Exception as exc:  # noqa: BLE001
            self.log.error("Error processing data packet: %s", exc)

    def _handle_emg_data(self, emg_packet: EmgDataPacket) -> None:
        """Handle EMG data packets."""
        device_id = emg_packet.hardware_identifier

        # Feed live buffer (if enabled)
        if self.live_buffer is not None:
            try:
                arr = np.asarray(emg_packet.emg_data, dtype=np.float32)
                self.live_buffer.append(arr)
            except Exception as exc:  # noqa: BLE001
                self.log.error("Error appending to live buffer: %s", exc)

        # Lazily create recorder for each device
        if device_id not in self.recorders:
            filename = f"{self.base_filename}.h5"
            recorder = EMGHDF5Recorder(
                filename=filename,
                device_id=device_id,
                channels_size=len(emg_packet.emg_data),
            )
            device_info = self.device_info.get(device_id, {})
            recorder.start_recording(device_info)
            self.recorders[device_id] = recorder
            self.log.info("Created recorder for device %s -> %s", device_id, filename)

        self.recorders[device_id].add_data(emg_packet)

    def _handle_device_info(self, info_packet: HardwareDiscoveryPacket) -> None:
        """Store device info for later writing into HDF5 attributes."""
        device_id = info_packet.hardware_identifier

        self.device_info[device_id] = {
            "display_name": info_packet.display_name,
            "model": info_packet.model,
            "manufacturer": info_packet.manufacturer,
            "hardware_revision": info_packet.hardware_revision,
            "firmware_version": info_packet.firmware_version,
            "mcu_temp": info_packet.mcu_temp,
            "battery_voltage": info_packet.battery_voltage,
        }
        self.log.info("Received device info for %s: %s", device_id, info_packet.display_name)

    # ------------------------------------------------------------------ #
    # Session-level helpers
    # ------------------------------------------------------------------ #

    def log_event_for_all_devices(self, time_sec: float, name: str, info_json: str) -> None:
        """Log the same event for all active devices."""
        for recorder in self.recorders.values():
            recorder.log_event(time_sec, name, info_json)

    def stop_all_recorders(self) -> None:
        """Stop all recorders and clear registry."""
        for recorder in list(self.recorders.values()):
            recorder.stop_recording()
        self.recorders.clear()
        self.log.info("All recorders stopped.")


# --------------------------------------------------------------------------- #
# High-level session
# --------------------------------------------------------------------------- #

class WearLabEMGSession:
    """
    High-level controller used by PsychoPy scripts via the Builder component.

    Typical usage (in generated code):

        emgRecorder = WearLabEMGSession(
            base_filename=thisExp.dataFileName + "_emg",
            target_vid=0x0483,
            target_pid=0x5740,
            semg_cycle_ms=1000,
            channels=64,
            enable_live_buffer=True,
        )
        emgRecorder.start()
        ...
        emgRecorder.log_event(...)
        ...
        emgRecorder.stop()
    """

    def __init__(
        self,
        base_filename: str,
        target_vid: int = 0x0483,
        target_pid: int = 0x5740,
        semg_cycle_ms: int = 1000,
        channels: int = 64,
        enable_live_buffer: bool = False,
        live_buffer_window_size: int = 1000,
    ) -> None:
        """
        Parameters
        ----------
        base_filename : str
            Base filename (without extension) for EMG HDF5 files.
        target_vid : int
            USB VID for WearLab device.
        target_pid : int
            USB PID for WearLab device.
        semg_cycle_ms : int
            Sampling period value passed to the device (unit depends on firmware).
        channels : int
            Expected number of EMG channels (used mainly for info/logging).
        enable_live_buffer : bool
            Whether to enable the internal live buffer for visualization.
        live_buffer_window_size : int
            Number of time points kept in the live buffer.
        """
        self.base_filename = base_filename
        self.target_vid = target_vid
        self.target_pid = target_pid
        self.semg_cycle_ms = semg_cycle_ms
        self.channels = channels

        self.devices: list[WearLabDevice] = []
        self.device_comm: Optional[DeviceCommunication] = None
        self.emg_handler: Optional[EMGDataHandler] = None
        self._running: bool = False

        self.enable_live_buffer = enable_live_buffer
        self.live_buffer_window_size = live_buffer_window_size
        self.live_buffer: Optional[EMGLivePlotBuffer] = None
        self.live_window_thread: Optional[EMGLivePlotTkWindow] = None

        self.log = logging.getLogger(f"{__name__}.WearLabEMGSession")

    # ------------------------------------------------------------------ #
    # Device callbacks
    # ------------------------------------------------------------------ #

    def _device_disconnect_callback(self, device: WearLabDevice) -> None:
        """Called when a device disconnects."""
        if device in self.devices:
            self.devices.remove(device)
        self.log.info("Device %s disconnected", device.hardware_identifier)

    def _device_discovery_callback(self, device: WearLabDevice) -> None:
        """Called when a new device is discovered."""
        self.log.info("Device %s discovered, connecting...", device.hardware_identifier)
        if self.device_comm is not None:
            self.device_comm.connect_device(device.hardware_identifier)

    def _device_connect_callback(self, device: WearLabDevice, *args: Any) -> None:
        """Called after a device is connected and ready."""
        self.devices.append(device)
        self.log.info("Device %s connected", device.hardware_identifier)

        if self.device_comm is None or self.emg_handler is None:
            self.log.error("DeviceCommunication or EMGDataHandler not initialized.")
            return

        # Register data callback and configure SEMG sampling
        self.device_comm.register_data_change_callback(self.emg_handler.data_callback)

        # Small delay to let the device settle (tweak if needed)
        time.sleep(0.5)

        self.device_comm.send_command(
            device.hardware_identifier,
            PackageType.SetCycleSEMG,
            self.semg_cycle_ms,
        )
        self.log.info(
            "Set SEMG cycle=%d for device %s",
            self.semg_cycle_ms,
            device.hardware_identifier,
        )

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start device communication and EMG recording."""
        if self._running:
            return

        self.log.info("Starting WearLab EMG session with base filename '%s'", self.base_filename)

        # Prepare live buffer and Tk visualization window if requested
        if self.enable_live_buffer:
            self.live_buffer = EMGLivePlotBuffer(
                channels=self.channels,
                window_size=self.live_buffer_window_size,
            )
            try:
                self.live_window_thread = EMGLivePlotTkWindow(
                    live_buffer=self.live_buffer,
                    title=f"WearLab EMG Live ({os.path.basename(self.base_filename)})",
                    update_interval_ms=50,
                    max_points=800,
                    line_width=1,
                    channels_to_show=None,  # None -> show all channels
                )
                self.live_window_thread.start()
            except Exception as exc:  # noqa: BLE001
                self.log.error("Failed to start Tk live window: %s", exc)
                self.live_window_thread = None
        else:
            self.live_buffer = None
            self.live_window_thread = None

        self.emg_handler = EMGDataHandler(self.base_filename, live_buffer=self.live_buffer)
        self.device_comm = DeviceCommunication(
            target_vid=self.target_vid,
            target_pid=self.target_pid,
        )

        self.device_comm.register_device_discovery_callback(self._device_discovery_callback)
        self.device_comm.register_device_connect_callback(self._device_connect_callback)
        self.device_comm.register_device_disconnect_callback(self._device_disconnect_callback)

        self.device_comm.start_communication()
        self._running = True

    def stop(self) -> None:
        """Stop EMG recording and device communication."""
        if not self._running:
            return

        self.log.info("Stopping WearLab EMG session.")

        if self.emg_handler is not None:
            self.emg_handler.stop_all_recorders()

        if self.device_comm is not None:
            self.device_comm.stop_communication()

        # Stop Tk live window if running
        if self.live_window_thread is not None:
            try:
                self.live_window_thread.stop()
            except Exception as exc:  # noqa: BLE001
                self.log.error("Error stopping live window thread: %s", exc)
            self.live_window_thread = None

        self._running = False

    # ------------------------------------------------------------------ #
    # Event logging API for PsychoPy
    # ------------------------------------------------------------------ #

    def log_event(self, time_sec: float, name: str, info: Dict[str, Any] | None = None) -> None:
        """
        Log an experimental event for all active devices.

        Parameters
        ----------
        time_sec : float
            PsychoPy time in seconds (e.g., globalClock.getTime()).
        name : str
            Short event name, e.g. "trial_begin", "stim_onset", "experiment_end".
        info : dict, optional
            Extra metadata which will be JSON-encoded.
        """
        if self.emg_handler is None:
            return

        import json  # local import to avoid mandatory json dependency in weird contexts

        if info is None:
            info = {}

        info_json = json.dumps(info, ensure_ascii=False)
        self.emg_handler.log_event_for_all_devices(time_sec, name, info_json)

    # ------------------------------------------------------------------ #
    # Live buffer access
    # ------------------------------------------------------------------ #

    def get_live_window(self) -> Optional[np.ndarray]:
        """
        Return a copy of the current live window, or None if disabled or empty.

        This is intended for visualization only, not for analysis.
        """
        if self.live_buffer is None:
            return None
        return self.live_buffer.get_window()
