# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PsychoPy Builder component for SpO2 (Pulse Oximetry) recording.

This component:
- Creates and starts an SpO2Session at experiment start.
- Supports recording of processed values (SpO2%, BPM) and raw PPG waveforms.
- Automatically derives the base filename from thisExp.dataFileName.
- Logs "experiment_start" and "experiment_end" events automatically.
- Optionally enables a live visualization of the PPG waveform or numeric display.
"""
from pathlib import Path
from psychopy.experiment.components import BaseComponent, Param, _translate


class SpO2RecorderComponent(BaseComponent):
    """Builder component which controls a Pulse Oximeter (SpO2) recording session."""

    # Component categorization in PsychoPy Builder interface
    categories = ['Physiology', 'Oximetry','SpO2']
    targets = ['PsychoPy']

    # Icons (Pointing to generic paths, ensure these files exist or remove these lines)
    iconFile = Path(__file__).parent / 'SpO2.png'
    iconSVG = Path(__file__).parent / 'SpO2.svg'

    tooltip = _translate(
        'SpO2Recorder: A component for recording Blood Oxygen Saturation (SpO2), '
        'Pulse Rate (BPM), and Photoplethysmogram (PPG) waveforms.'
    )

    def __init__(self, exp, parentName, name="spo2Recorder"):
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
        self.type = "SpO2Recorder"
        self.url = "https://example.com/psychopy-spo2-drivers"

        # ------------------------------------------------------------------ #
        # Custom Builder parameters (dialog fields)
        # ------------------------------------------------------------------ #

        # Device Model
        self.params["model"] = Param(
            "Contec CMS50",
            valType="str",
            inputType="choice",
            allowedVals=[
                'Contec CMS50',
                'Nonin Xpod',
                'Nonin 3150 BLE',
                'Berry Med BLE',
                'Generic Serial Protocol'
            ],
            hint=_translate(
                "The hardware model of the Pulse Oximeter."
            ),
            label=_translate("Device Model"),
        )

        # Serial Port / Connection String
        self.params["serialPort"] = Param(
            "COM3",
            valType="str",
            inputType="entry",
            allowedTypes=[],
            hint=_translate(
                "The Serial Port (e.g., 'COM3', '/dev/ttyUSB0') for wired devices "
                "or MAC Address for Bluetooth LE devices."
            ),
            label=_translate("Port / Address"),
        )

        # Recording Mode
        self.params["recordingMode"] = Param(
            "SpO2 + BPM + Raw PPG",
            valType="str",
            inputType="choice",
            allowedVals=[
                'SpO2 + BPM + Raw PPG',
                'SpO2 + BPM Only (Low Res)'
            ],
            hint=_translate(
                "Choose 'Raw PPG' to save high-frequency waveform data (larger file). "
                "Choose 'Only' for simple 1Hz numeric logs."
            ),
            label=_translate("Data Mode"),
        )

        # Filename suffix appended to thisExp.dataFileName
        self.params["fileSuffix"] = Param(
            '"_spo2"',
            valType="code",
            allowedTypes=[],
            hint=_translate(
                "String suffix appended to thisExp.dataFileName to build "
                "the SpO2 output file base name."
            ),
            label=_translate("File suffix"),
        )

        # Enable live visualization (Tk window)
        self.params["enableVisualization"] = Param(
            False,
            valType="bool",
            allowedTypes=[],
            hint=_translate(
                "If checked, a separate window will show the live PPG waveform "
                "and current SpO2/BPM values."
            ),
            label=_translate("Enable visualization"),
        )

        # Extend the order so these appear near the top of the dialog
        self.order += ["model", "serialPort", "recordingMode", "fileSuffix", "enableVisualization"]

    # ------------------------------------------------------------------ #
    # Code generation helpers
    # ------------------------------------------------------------------ #

    def writeInitCode(self, buff):
        """
        Called by PsychoPy when generating the script, to write code which
        should run at the beginning of the experiment.
        """
        name = self.params["name"].val
        model = self.params["model"].val
        port = self.params["serialPort"].val
        rec_mode = self.params["recordingMode"].val
        suffix = self.params["fileSuffix"].val
        enable_viz = self.params["enableVisualization"].val

        # Generates code assuming a 'psychopy.hardware.spo2' module exists
        code = f"""
# Initialize SpO2 recording session for component '{name}'
# Note: Requires a fictional 'psychopy.hardware.spo2' driver.
from psychopy.hardware.spo2 import SpO2Session

# Resolve visualization flag
{name}_enableViz = bool({enable_viz})
{name}_fileBase = thisExp.dataFileName + {suffix}

# Map string selection to internal boolean or constant if needed
{name}_save_waveform = True if 'Raw PPG' in '{rec_mode}' else False

# Initialize the session
{name} = SpO2Session(
    model='{model}',
    port='{port}',
    save_ppg_waveform={name}_save_waveform,
    base_filename={name}_fileBase,
    enable_live_view={name}_enableViz
)

# Start connection to the oximeter
print(f"Connecting to SpO2 Device ({model}) on {port}...")
# Note: Some SpO2 devices take 3-5 seconds to stabilize signal
{name}.start()

# Log experiment_start event
{name}.log_event(
    time_sec=0.0,
    name="experiment_start",
    info={{"component": "{name}", "routine": "{self.parentName}", "model": "{model}"}}
)
"""
        buff.writeIndentedLines(code)

    def writeRoutineEachFrameCode(self, buff):
        """
        Called once per frame during the parent Routine.
        """
        name = self.params["name"].val
        code = f"""
# SpO2RecorderComponent '{name}': Visualization is handled in a separate thread/window.
# You could access {name}.get_current_spo2() here if you wanted to display it on screen.
"""
        buff.writeIndentedLines(code)

    def writeRoutineEndCode(self, buff):
        """
        Called when the parent Routine ends.
        """
        name = self.params["name"].val
        code = f"""
# Optional: Log routine boundary to SpO2 file
# {name}.log_event(time_sec=globalClock.getTime(), name="routine_end")
"""
        buff.writeIndentedLines(code)

    def writeExperimentEndCode(self, buff):
        """
        Called by PsychoPy when generating the script, to write code which
        should run at the end of the experiment.
        """
        name = self.params["name"].val

        code = f"""
# Stop SpO2 session for '{name}' and save final data
try:
    if '{name}' in locals():
        {name}.log_event(
            time_sec=globalClock.getTime(),
            name="experiment_end",
            info={{"component": "{name}"}}
        )
        print(f"Stopping SpO2 recording for {{name}}...")
        {name}.stop()
        {name}.close()
except Exception as _spo2_err:
    print("Error while stopping SpO2 session '{{}}': {{}}".format("{name}", _spo2_err))
"""
        buff.writeIndentedLines(code)
