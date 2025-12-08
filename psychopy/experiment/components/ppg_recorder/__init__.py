#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PsychoPy Builder component for PPG (Photoplethysmography) recording.

This component:
- Creates and starts a PPGSession at experiment start.
- Focuses on raw waveform acquisition for HRV (Heart Rate Variability) analysis.
- Supports configuration of Light Sources (Green/Red/IR) if hardware permits.
- Automatically derives the HDF5/CSV base filename from thisExp.dataFileName.
- Includes options for high-speed logging and live visualization.
"""
from pathlib import Path
from psychopy.experiment.components import BaseComponent, Param, _translate


class PPGRecorderComponent(BaseComponent):
    """Builder component which controls a raw PPG sensor session."""

    # Component categorization
    categories = ['Physiology']
    targets = ['PsychoPy']

    # Icons (Ensure these exist or comment them out)
    iconFile = Path(__file__).parent / 'ppg.png'
    iconSVG = Path(__file__).parent / 'ppg.svg'

    tooltip = _translate(
        'PPGRecorder: A component for recording raw Photoplethysmogram '
        'waveforms (BVP) for Heart Rate and HRV analysis.'
    )

    def __init__(self, exp, parentName, name="ppgRecorder"):
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
        self.type = "PPGRecorder"
        self.url = "https://example.com/psychopy-ppg-drivers"

        # ------------------------------------------------------------------ #
        # Custom Builder parameters (dialog fields)
        # ------------------------------------------------------------------ #

        # Device Model
        self.params["model"] = Param(
            "Shimmer3 Optical Pulse",
            valType="str",
            inputType="choice",
            allowedVals=[
                'Shimmer3 Optical Pulse',
                'Biopac MP160 PPG',
                'Polar Verity Sense (SDK)',
                'Vernier Go Direct',
                'Generic Serial Sensor'
            ],
            hint=_translate(
                "The hardware model of the PPG sensor."
            ),
            label=_translate("Device Model"),
        )

        # Serial Port / Bluetooth Address
        self.params["serialPort"] = Param(
            "COM3",
            valType="str",
            inputType="entry",
            allowedTypes=[],
            hint=_translate(
                "The Serial Port (e.g., 'COM3') or Bluetooth MAC Address (e.g., 'A0:B1:C2...')."
            ),
            label=_translate("Port / ID"),
        )

        # Sampling Rate
        # For HRV analysis, higher rates (>=250Hz) are preferred to detect peak timing accurately.
        self.params["sampleRateHz"] = Param(
            "256",
            valType="code",
            inputType="choice",
            allowedVals=['50', '64', '128', '256', '512', '1000'],
            hint=_translate(
                "Sampling rate in Hz. For precise HRV analysis, select 256Hz or higher."
            ),
            label=_translate("Sample Rate (Hz)"),
        )

        # Light Source Selection
        # Green is better for motion artifacts (wrist), Red/IR is better for depth/clinical.
        self.params["lightSource"] = Param(
            "Green (530nm)",
            valType="str",
            inputType="choice",
            allowedVals=[
                'Green (530nm)',
                'Red (660nm) + IR (940nm)',
                'Auto / Default'
            ],
            hint=_translate(
                "Select the LED wavelength. Green is standard for wearables; Red/IR is for clinical probes."
            ),
            label=_translate("Light Source"),
        )

        # Filename suffix
        self.params["fileSuffix"] = Param(
            '"_ppg"',
            valType="code",
            allowedTypes=[],
            hint=_translate(
                "String suffix appended to thisExp.dataFileName to build "
                "the PPG output file base name."
            ),
            label=_translate("File suffix"),
        )

        # Enable live visualization
        self.params["enableVisualization"] = Param(
            False,
            valType="bool",
            allowedTypes=[],
            hint=_translate(
                "If checked, a separate window will show the live blood volume pulse (BVP) signal."
            ),
            label=_translate("Enable visualization"),
        )

        # Extend the order
        self.order += ["model", "serialPort", "sampleRateHz", "lightSource", "fileSuffix", "enableVisualization"]

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
        light = self.params["lightSource"].val
        suffix = self.params["fileSuffix"].val
        enable_viz = self.params["enableVisualization"].val

        # Generates code assuming a 'psychopy.hardware.ppg' module exists
        code = f"""
# Initialize PPG recording session for component '{name}'
# Note: Requires a fictional 'psychopy.hardware.ppg' driver.
from psychopy.hardware.ppg import PPGSession

# Resolve visualization flag
{name}_enableViz = bool({enable_viz})
{name}_fileBase = thisExp.dataFileName + {suffix}

# Map light source string to code if necessary
{name}_light_config = '{light}'

# Initialize the session
{name} = PPGSession(
    model='{model}',
    port='{port}',
    sample_rate_hz={rate},
    light_source={name}_light_config,
    base_filename={name}_fileBase,
    enable_live_plot={name}_enableViz,
    buffer_size_seconds=10  # Keep 10s buffer for visualization
)

# Connect to sensor
print(f"Connecting to PPG Sensor ({model}) on {port}...")
{name}.start()

# Log experiment_start event
{name}.log_event(
    time_sec=0.0,
    name="experiment_start",
    info={{
        "component": "{name}", 
        "routine": "{self.parentName}", 
        "model": "{model}",
        "config_rate": {rate}
    }}
)
"""
        buff.writeIndentedLines(code)

    def writeRoutineEachFrameCode(self, buff):
        """
        Called once per frame during the parent Routine.
        """
        name = self.params["name"].val
        code = f"""
# PPGRecorderComponent '{name}': 
# Live visualization is running in a background thread if enabled.
"""
        buff.writeIndentedLines(code)

    def writeRoutineEndCode(self, buff):
        """
        Called when the parent Routine ends.
        """
        name = self.params["name"].val
        code = f"""
# Optional: Mark routine end in PPG data
# {name}.log_event(time_sec=globalClock.getTime(), name="routine_end")
"""
        buff.writeIndentedLines(code)

    def writeExperimentEndCode(self, buff):
        """
        Called by PsychoPy when generating the script.
        """
        name = self.params["name"].val

        code = f"""
# Stop PPG session for '{name}' and close file
try:
    if '{name}' in locals():
        {name}.log_event(
            time_sec=globalClock.getTime(),
            name="experiment_end",
            info={{"component": "{name}"}}
        )
        print(f"Stopping PPG recording for {{name}}...")
        {name}.stop()
        {name}.close()
except Exception as _ppg_err:
    print("Error while stopping PPG session '{{}}': {{}}".format("{name}", _ppg_err))
"""
        buff.writeIndentedLines(code)
