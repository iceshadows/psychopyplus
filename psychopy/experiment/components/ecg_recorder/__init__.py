#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PsychoPy Builder component for ECG (Electrocardiogram) recording.

This component:
- Creates and starts an ECGSession at experiment start.
- Captures raw electrical heart signals (mV) and/or RR-Intervals.
- Includes specific settings for Mains Noise Filtering (50/60Hz).
- Automatically derives the base filename from thisExp.dataFileName.
- Supports live strip-chart visualization of the ECG waveform.
"""
from pathlib import Path
from psychopy.experiment.components import BaseComponent, Param, _translate


class ECGRecorderComponent(BaseComponent):
    """Builder component which controls an ECG recording session."""

    # Component categorization
    categories = ['Physiology', 'Electrophysiology']
    targets = ['PsychoPy']

    # Icons (Ensure these exist or comment them out)
    iconFile = Path(__file__).parent / 'HR.png'
    iconSVG = Path(__file__).parent / 'HR.svg'

    tooltip = _translate(
        'ECGRecorder: A component for recording Electrocardiogram signals, '
        'including Raw Waveform (mV) and RR-Intervals.'
    )

    def __init__(self, exp, parentName, name="ecgRecorder"):
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
        self.type = "ECGRecorder"
        self.url = "https://example.com/psychopy-ecg-drivers"

        # ------------------------------------------------------------------ #
        # Custom Builder parameters (dialog fields)
        # ------------------------------------------------------------------ #

        # Device Model
        self.params["model"] = Param(
            "Polar H10 (BLE)",
            valType="str",
            inputType="choice",
            allowedVals=[
                'Polar H10 (BLE)',
                'Shimmer3 ECG',
                'Biopac MP160',
                'Bittium Faros',
                'OpenBCI Cyton',
                'Generic Serial ECG'
            ],
            hint=_translate(
                "The hardware model of the ECG sensor."
            ),
            label=_translate("Device Model"),
        )

        # Port / ID
        self.params["serialPort"] = Param(
            "COM3",
            valType="str",
            inputType="entry",
            allowedTypes=[],
            hint=_translate(
                "Serial Port (e.g., 'COM3') or Bluetooth MAC ID (e.g., 'A0:B1:C2...') "
                "depending on the device type."
            ),
            label=_translate("Port / Device ID"),
        )

        # Sampling Rate
        self.params["sampleRateHz"] = Param(
            "500",
            valType="code",
            inputType="choice",
            allowedVals=['130', '250', '500', '1000', '2000'],
            hint=_translate(
                "Sampling rate in Hz. 500Hz is recommended for precise QRS detection "
                "and HRV analysis."
            ),
            label=_translate("Sample Rate (Hz)"),
        )

        # Mains Noise Filter (Notch Filter)
        # Crucial for ECG because the signal is small and prone to power line noise.
        self.params["mainsFilter"] = Param(
            "None",
            valType="str",
            inputType="choice",
            allowedVals=[
                'None',
                '50Hz (EU/Asia)',
                '60Hz (US/JP)'
            ],
            hint=_translate(
                "Apply a hardware or software notch filter to remove power line interference."
            ),
            label=_translate("Mains Filter"),
        )

        # Data Mode
        self.params["dataMode"] = Param(
            "Raw Waveform + RR",
            valType="str",
            inputType="choice",
            allowedVals=[
                'Raw Waveform + RR',
                'Raw Waveform Only',
                'RR-Intervals Only'
            ],
            hint=_translate(
                "Select 'Raw Waveform' for full signal analysis. "
                "Select 'RR-Intervals' for lightweight HRV logging."
            ),
            label=_translate("Data Mode"),
        )

        # Filename suffix
        self.params["fileSuffix"] = Param(
            '"_ecg"',
            valType="code",
            allowedTypes=[],
            hint=_translate(
                "Suffix for the output file."
            ),
            label=_translate("File suffix"),
        )

        # Enable live visualization
        self.params["enableVisualization"] = Param(
            False,
            valType="bool",
            allowedTypes=[],
            hint=_translate(
                "If checked, opens a window displaying the live ECG strip chart."
            ),
            label=_translate("Enable visualization"),
        )

        # Extend the order
        self.order += ["model", "serialPort", "sampleRateHz", "mainsFilter", "dataMode", "fileSuffix",
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
        mains = self.params["mainsFilter"].val
        mode = self.params["dataMode"].val
        suffix = self.params["fileSuffix"].val
        enable_viz = self.params["enableVisualization"].val

        # Generates code assuming a 'psychopy.hardware.ecg' module exists
        code = f"""
# Initialize ECG recording session for component '{name}'
# Note: Requires a fictional 'psychopy.hardware.ecg' driver.
from psychopy.hardware.ecg import ECGSession

# Resolve visualization flag
{name}_enableViz = bool({enable_viz})
{name}_fileBase = thisExp.dataFileName + {suffix}

# Map Mains Filter selection
{name}_notch_freq = None
if '50Hz' in '{mains}':
    {name}_notch_freq = 50
elif '60Hz' in '{mains}':
    {name}_notch_freq = 60

# Initialize the session
{name} = ECGSession(
    model='{model}',
    port='{port}',
    sample_rate_hz={rate},
    notch_filter_hz={name}_notch_freq,
    data_mode='{mode}',
    base_filename={name}_fileBase,
    enable_live_view={name}_enableViz
)

# Connect to sensor
print(f"Connecting to ECG Device ({model}) on {port}...")
# Note: Ensure skin contact is good before starting for clean signal
{name}.start()

# Log experiment_start event
{name}.log_event(
    time_sec=0.0,
    name="experiment_start",
    info={{
        "component": "{name}", 
        "routine": "{self.parentName}", 
        "filter": "{mains}"
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
# ECGRecorderComponent '{name}': 
# Live visualization typically runs in a separate thread.
# Access {name}.get_current_hr() here if you need to trigger stimuli based on heart rate.
"""
        buff.writeIndentedLines(code)

    def writeRoutineEndCode(self, buff):
        """
        Called when the parent Routine ends.
        """
        name = self.params["name"].val
        code = f"""
# Optional: Log routine boundary
# {name}.log_event(time_sec=globalClock.getTime(), name="routine_end")
"""
        buff.writeIndentedLines(code)

    def writeExperimentEndCode(self, buff):
        """
        Called at experiment end.
        """
        name = self.params["name"].val

        code = f"""
# Stop ECG session for '{name}'
try:
    if '{name}' in locals():
        {name}.log_event(
            time_sec=globalClock.getTime(),
            name="experiment_end",
            info={{"component": "{name}"}}
        )
        print(f"Stopping ECG recording for {{name}}...")
        {name}.stop()
        {name}.close()
except Exception as _ecg_err:
    print("Error while stopping ECG session '{{}}': {{}}".format("{name}", _ecg_err))
"""
        buff.writeIndentedLines(code)
