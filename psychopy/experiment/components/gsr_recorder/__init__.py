#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PsychoPy Builder component for GSR (EDA) recording.

This component:
- Creates and starts a GSRSession at experiment start.
- Supports common GSR hardware models (e.g., Shimmer, Biopac, Generic Serial).
- Automatically derives the GSR HDF5/CSV base filename from thisExp.dataFileName.
- Logs "experiment_start" and "experiment_end" events automatically.
- Optionally enables a live visualization.
"""
from pathlib import Path
from psychopy.experiment.components import BaseComponent, Param, _translate


class GSRRecorderComponent(BaseComponent):
    """Builder component which controls a GSR/EDA recording session."""

    # Component categorization in PsychoPy Builder interface
    categories = ['Physiology', 'GSR']
    targets = ['PsychoPy']

    # Icons (Pointing to generic paths, ensure these files exist or remove these lines)
    iconFile = Path(__file__).parent / 'gsr.png'
    iconSVG = Path(__file__).parent / 'gsr.svg'

    tooltip = _translate(
        'GSRRecorder: A component for recording Galvanic Skin Response '
        '(EDA) signals during your experiment.'
    )

    def __init__(self, exp, parentName, name="gsrRecorder"):
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
        self.type = "GSRRecorder"
        self.url = "https://example.com/psychopy-gsr-drivers"

        # ------------------------------------------------------------------ #
        # Custom Builder parameters (dialog fields)
        # ------------------------------------------------------------------ #

        # Device Model
        self.params["model"] = Param(
            "Shimmer3 GSR+",
            valType="str",
            inputType="choice",
            allowedVals=[
                'Shimmer3 GSR+',
                'Biopac MP160',
                'Muse 2 (Aux)',
                'Generic Serial'
            ],
            hint=_translate(
                "The hardware model of the GSR sensor."
            ),
            label=_translate("Device Model"),
        )

        # Serial Port / MAC Address
        self.params["serialPort"] = Param(
            "COM3",
            valType="str",
            inputType="entry",
            allowedTypes=[],
            hint=_translate(
                "The Serial Port (e.g., 'COM3', '/dev/ttyUSB0') or "
                "Bluetooth MAC address of the device."
            ),
            label=_translate("Port / Address"),
        )

        # Sampling Rate (Hz)
        # GSR is a slow signal, 10-128Hz is usually sufficient.
        self.params["sampleRateHz"] = Param(
            "50",
            valType="code",
            inputType="choice",
            allowedVals=['10', '50', '128', '256', '512'],
            hint=_translate(
                "Sampling rate in Hz. GSR is a slow signal; 50Hz is typically sufficient."
            ),
            label=_translate("Sample Rate (Hz)"),
        )

        # Filename suffix appended to thisExp.dataFileName
        self.params["fileSuffix"] = Param(
            '"_gsr"',
            valType="code",
            allowedTypes=[],
            hint=_translate(
                "String suffix appended to thisExp.dataFileName to build "
                "the GSR output file base name."
            ),
            label=_translate("File suffix"),
        )

        # Enable live visualization (Tk window)
        self.params["enableVisualization"] = Param(
            False,
            valType="bool",
            allowedTypes=[],
            hint=_translate(
                "If checked, a separate window will show live GSR (Conductance) "
                "waveforms to monitor signal quality."
            ),
            label=_translate("Enable visualization"),
        )

        # Extend the order so these appear near the top of the dialog
        self.order += ["model", "serialPort", "sampleRateHz", "fileSuffix", "enableVisualization"]

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
        rate = self.params["sampleRateHz"].val
        suffix = self.params["fileSuffix"].val
        enable_viz = self.params["enableVisualization"].val

        # Generates code assuming a 'psychopy.hardware.gsr' module exists
        # containing a generic 'GSRSession' class.
        code = f"""
# Initialize GSR recording session for component '{name}'
# Note: Requires a fictional 'psychopy.hardware.gsr' driver or similar wrapper.
from psychopy.hardware.gsr import GSRSession

# Resolve visualization flag
{name}_enableViz = bool({enable_viz})

{name}_fileBase = thisExp.dataFileName + {suffix}

# Initialize the session based on the selected model and port
{name} = GSRSession(
    model='{model}',
    port='{port}',
    sample_rate_hz={rate},
    base_filename={name}_fileBase,
    enable_live_buffer={name}_enableViz,
    buffer_window_size=500  # Window size for viz
)

# Verify connection and start stream
print(f"Connecting to GSR Device ({model}) on {port}...")
{name}.start()

# Log experiment_start event into GSR data stream
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
# GSRRecorderComponent '{name}': Live visualization is handled internally 
# by the session thread/window if enabled. No per-frame code needed here.
"""
        buff.writeIndentedLines(code)

    def writeRoutineEndCode(self, buff):
        """
        Called when the parent Routine ends.
        """
        name = self.params["name"].val
        # You might want to insert a marker for the end of a routine
        code = f"""
# Optional: Log routine end marker to GSR stream
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
# Log experiment_end event and stop GSR session for '{name}'
try:
    if '{name}' in locals():
        {name}.log_event(
            time_sec=globalClock.getTime(),
            name="experiment_end",
            info={{"component": "{name}"}}
        )
        print(f"Stopping GSR recording for {{name}}...")
        {name}.stop()
        {name}.close() # Close serial connections safely
except Exception as _gsr_err:
    print("Error while stopping GSR session '{{}}': {{}}".format("{name}", _gsr_err))
"""
        buff.writeIndentedLines(code)
