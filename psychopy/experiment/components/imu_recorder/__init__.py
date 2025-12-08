#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PsychoPy Builder component for IMU (Inertial Measurement Unit) recording.

This component:
- Creates and starts an IMUSession at experiment start.
- Captures Accelerometer, Gyroscope, Magnetometer, and optionally Quaternions.
- Allows configuration of sensitivity (G-range) and sensor fusion mode.
- Automatically derives the base filename from thisExp.dataFileName.
- Supports 3D visualization options (handled by the driver session).
"""
from pathlib import Path
from psychopy.experiment.components import BaseComponent, Param, _translate


class IMURecorderComponent(BaseComponent):
    """Builder component which controls an IMU recording session."""

    # Component categorization
    categories = ['Motion', 'Physiology']
    targets = ['PsychoPy']

    # Icons (Ensure these exist or comment them out)
    iconFile = Path(__file__).parent / 'IMU.png'
    iconSVG = Path(__file__).parent / 'IMU.svg'

    tooltip = _translate(
        'IMURecorder: A component for recording 6-axis or 9-axis motion data '
        '(Accel, Gyro, Mag) and orientation (Quaternions/Euler).'
    )

    def __init__(self, exp, parentName, name="imuRecorder"):
        """
        Parameters
        ----------
        exp :
            The Experiment to which this component belongs.
        parentName : str
            The name of the parent Routine.
        name : str
            The name of this component instance.
        """
        super().__init__(exp, parentName, name=name)

        # Type used internally by PsychoPy
        self.type = "IMURecorder"
        self.url = "https://example.com/psychopy-imu-drivers"

        # ------------------------------------------------------------------ #
        # Custom Builder parameters (dialog fields)
        # ------------------------------------------------------------------ #

        # Device Model
        self.params["model"] = Param(
            "WitMotion WT901",
            valType="str",
            inputType="choice",
            allowedVals=[
                'WitMotion WT901 (Serial)',
                'WitMotion BLE',
                'Xsens DOT',
                'Shimmer3 IMU',
                'MbientLab MetaMotion',
                'Generic UART IMU'
            ],
            hint=_translate(
                "The hardware model of the IMU sensor."
            ),
            label=_translate("Device Model"),
        )

        # Port / Address
        self.params["serialPort"] = Param(
            "COM3",
            valType="str",
            inputType="entry",
            allowedTypes=[],
            hint=_translate(
                "The Serial Port (e.g., 'COM3') or Bluetooth MAC Address."
            ),
            label=_translate("Port / Address"),
        )

        # Sampling Rate
        self.params["sampleRateHz"] = Param(
            "100",
            valType="code",
            inputType="choice",
            allowedVals=['50', '60', '100', '200', '400', '1000'],
            hint=_translate(
                "Sampling rate in Hz. 100Hz is standard for human motion analysis."
            ),
            label=_translate("Sample Rate (Hz)"),
        )

        # Accelerometer Range (Sensitivity)
        self.params["accelRange"] = Param(
            "4g",
            valType="str",
            inputType="choice",
            allowedVals=['2g', '4g', '8g', '16g'],
            hint=_translate(
                "Measurement range. Choose lower (2g) for fine sedentary movements, "
                "higher (16g) for impacts/running."
            ),
            label=_translate("Accel Range"),
        )

        # Data Mode (Raw vs Fusion)
        self.params["fusionMode"] = Param(
            "Raw 9-Axis + Quaternions",
            valType="str",
            inputType="choice",
            allowedVals=[
                'Raw 9-Axis Only',
                'Raw 6-Axis (No Mag)',
                'Quaternions Only (Fusion)',
                'Raw 9-Axis + Quaternions'
            ],
            hint=_translate(
                "Select 'Quaternions' if you need absolute orientation. "
                "Select 'Raw' for standard Accel/Gyro analysis."
            ),
            label=_translate("Data Mode"),
        )

        # Filename suffix
        self.params["fileSuffix"] = Param(
            '"_imu"',
            valType="code",
            allowedTypes=[],
            hint=_translate(
                "Suffix for the output file (e.g., .csv or .hdf5)."
            ),
            label=_translate("File suffix"),
        )

        # Enable live visualization
        self.params["enableVisualization"] = Param(
            False,
            valType="bool",
            allowedTypes=[],
            hint=_translate(
                "If checked, shows a live plot of Accel data or a 3D orientation cube "
                "(driver dependent)."
            ),
            label=_translate("Enable visualization"),
        )

        # Extend the order
        self.order += ["model", "serialPort", "sampleRateHz", "accelRange", "fusionMode", "fileSuffix",
                       "enableVisualization"]

    # ------------------------------------------------------------------ #
    # Code generation helpers
    # ------------------------------------------------------------------ #

    def writeInitCode(self, buff):
        """
        Called by PsychoPy when generating the script.
        """
        name = self.params["name"].val
        model = self.params["model"].val
        port = self.params["serialPort"].val
        rate = self.params["sampleRateHz"].val
        g_range = self.params["accelRange"].val
        mode = self.params["fusionMode"].val
        suffix = self.params["fileSuffix"].val
        enable_viz = self.params["enableVisualization"].val

        # Generates code assuming a 'psychopy.hardware.imu' module exists
        code = f"""
# Initialize IMU recording session for component '{name}'
# Note: Requires a fictional 'psychopy.hardware.imu' driver.
from psychopy.hardware.imu import IMUSession

# Resolve visualization flag
{name}_enableViz = bool({enable_viz})
{name}_fileBase = thisExp.dataFileName + {suffix}

# Map settings to driver constants
{name}_settings = {{
    'accel_range': '{g_range}',
    'data_mode': '{mode}'
}}

# Initialize the session
{name} = IMUSession(
    model='{model}',
    port='{port}',
    sample_rate_hz={rate},
    settings={name}_settings,
    base_filename={name}_fileBase,
    enable_live_view={name}_enableViz
)

# Connect and Calibrate
# Note: IMUs often require a few seconds of stillness on startup for gyro bias calc
print(f"Connecting to IMU ({model}) on {port}... Please keep sensor still.")
{name}.start()

# Log experiment_start event
{name}.log_event(
    time_sec=0.0,
    name="experiment_start",
    info={{
        "component": "{name}", 
        "routine": "{self.parentName}", 
        "settings": {name}_settings
    }}
)
"""
        buff.writeIndentedLines(code)

    def writeRoutineEachFrameCode(self, buff):
        """
        Called once per frame.
        """
        name = self.params["name"].val
        code = f"""
# IMURecorderComponent '{name}': 
# Live 3D visualization or plotting is handled by the IMUSession background thread.
"""
        buff.writeIndentedLines(code)

    def writeRoutineEndCode(self, buff):
        """
        Called when the parent Routine ends.
        """
        name = self.params["name"].val
        code = f"""
# Optional: Log routine end marker
# {name}.log_event(time_sec=globalClock.getTime(), name="routine_end")
"""
        buff.writeIndentedLines(code)

    def writeExperimentEndCode(self, buff):
        """
        Called at experiment end.
        """
        name = self.params["name"].val

        code = f"""
# Stop IMU session for '{name}' and flush data to disk
try:
    if '{name}' in locals():
        {name}.log_event(
            time_sec=globalClock.getTime(),
            name="experiment_end",
            info={{"component": "{name}"}}
        )
        print(f"Stopping IMU recording for {{name}}...")
        {name}.stop()
        {name}.close()
except Exception as _imu_err:
    print("Error while stopping IMU session '{{}}': {{}}".format("{name}", _imu_err))
"""
        buff.writeIndentedLines(code)
