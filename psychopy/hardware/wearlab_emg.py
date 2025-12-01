#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Session management for WearLab EMG recording in PsychoPy.

This module contains three main classes:

- EMGHDF5Recorder
    Handles writing EMG samples, timestamps, device info and events to HDF5.

- EMGDataHandler
    Receives parsed WearLab protocol packets and forwards them to recorders.

- WearLabEMGSession
    High-level controller used by PsychoPy-generated scripts. It:
    - starts DeviceCommunication
    - wires callbacks
    - exposes a log_event(...) method to record PsychoPy events.
"""
import os
import time
import logging
from collections import deque
from typing import Dict, Optional, Deque, Any

import h5py
import numpy as np

from wearlab_protocolX.meta import (
    PackageType,
    EmgDataPacket,
    HardwareDiscoveryPacket,
)
from wearlab_protocolX.wearlab_device import DeviceCommunication, WearLabDevice


logger = logging.getLogger(__name__)


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
            # Ensure that the parent directory exists (e.g. "data/...")
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


class EMGDataHandler:
    """Handle WearLab protocol packets and dispatch them to HDF5 recorders."""

    def __init__(self, base_filename: str) -> None:
        """
        Parameters
        ----------
        base_filename : str
            Base filename (without extension). The final HDF5 filename is
            `base_filename + ".h5"`.
        """
        self.base_filename = base_filename
        self.recorders: Dict[str, EMGHDF5Recorder] = {}
        self.device_info: Dict[str, dict] = {}
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
        import os, h5py
        dir_path = os.path.dirname(self.base_filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with h5py.File(self.base_filename + ".h5", "a"):
            pass
        self.emg_handler = EMGDataHandler(self.base_filename)
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
