#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PsychoPy Builder component for WearLab EMG recording.

This component:

- Creates and starts a WearLabEMGSession at experiment start.
- Automatically derives the EMG HDF5 base filename from thisExp.dataFileName.
- Logs "experiment_start" and "experiment_end" events automatically.
- Optionally enables a live visualization using a separate Tk window
  implemented inside WearLabEMGSession.
"""
from pathlib import Path

from psychopy.experiment.components import BaseComponent, Param, _translate


class EMGRecorderComponent(BaseComponent):
    """Builder component which controls a WearLab EMG recording session."""

    targets = ['PsychoPy']

    categories = ['EMG']
    targets = ['PsychoPy']
    iconFile = Path(__file__).parent / 'wearlab_semg.png'
    iconSVG = Path(__file__).parent / 'wearlabSemg.svg'
    tooltip = _translate(
        'EMGRecorder: A component for recording Electromyography '
        '(EMG) signals during your experiment\n'
        '(requires WearLab™️ EMG hardware and drivers)'
    )

    def __init__(self, exp, parentName, name="emgRecorder"):
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
        self.type = "EMGRecorder"
        self.url = "https://example.com/wearlab-emg-psychopy"

        # ------------------------------------------------------------------ #
        # Custom Builder parameters (dialog fields)
        # ------------------------------------------------------------------ #

        # TS
        self.params["Brand"] = Param(
            "WearLab",
            valType="string",
            inputType="choice",
            allowedVals=['WearLab'],
            hint=_translate(
                "Brand or Protocol for the device"
            ),
            label=_translate("Brand"),
        )

        # Sampling period sent to the device firmware (ms)
        self.params["samplePeriodMs"] = Param(
            "1000",
            valType="code",
            inputType="choice",
            allowedVals=['250', '500', '1000', '2000', '4000'],
            hint=_translate(
                "Sampling period in milliseconds passed to the device "
                "(e.g. 1000 = 1000 ms, 1 = 1 ms, depending on firmware semantics)."
            ),
            label=_translate("Sample period (ms)"),
        )

        # Number of EMG channels
        self.params['nChannels'] = Param("16",
                                         valType="code", inputType="choice",
                                         allowedVals=['64', '16', '32', '8'],
                                         hint=_translate("Number of EMG channels."),
                                         label=_translate("Channels")
                                         )

        # PGA
        self.params['pga'] = Param("1",
                                   valType="code", inputType="choice",
                                   allowedVals=['1', '2', '4', '8', '12', '24'],
                                   hint=_translate("PGA"),
                                   label=_translate("PGA")
                                   )

        # Filename suffix appended to thisExp.dataFileName
        self.params["fileSuffix"] = Param(
            '"_emg"',
            valType="code",
            allowedTypes=[],
            hint=_translate(
                "String suffix appended to thisExp.dataFileName to build "
                "the EMG HDF5 file base name."
            ),
            label=_translate("File suffix"),
        )

        # Enable live visualization (Tk window)
        self.params["enableVisualization"] = Param(
            False,
            valType="bool",
            allowedTypes=[],
            hint=_translate(
                "If checked, a separate Tk window will show live EMG "
                "waveforms for all channels (handled by WearLabEMGSession)."
            ),
            label=_translate("Enable visualization"),
        )

        # Extend the order so these appear near the top of the dialog
        self.order += ["samplePeriodMs", "nChannels","pga", "fileSuffix", "enableVisualization"]

    # ------------------------------------------------------------------ #
    # Code generation helpers
    # ------------------------------------------------------------------ #

    def writeInitCode(self, buff):
        """
        Called by PsychoPy when generating the script, to write code which
        should run at the beginning of the experiment (after the Window is
        created, before any Routine starts).
        """
        name = self.params["name"].val
        period = self.params["samplePeriodMs"].val
        n_channels = self.params["nChannels"].val
        suffix = self.params["fileSuffix"].val
        enable_viz = self.params["enableVisualization"].val

        code = f"""
# Initialize WearLab EMG recording session for component '{name}'
from psychopy.hardware.wearlab_emg import WearLabEMGSession

# Resolve visualization flag directly from the Builder parameter
{name}_enableViz = bool({enable_viz})

{name}_fileBase = thisExp.dataFileName + {suffix}
{name} = WearLabEMGSession(
    base_filename={name}_fileBase,
    semg_cycle_ms={period},
    channels=int({n_channels}),
    enable_live_buffer={name}_enableViz,
    live_buffer_window_size=1000,
)
{name}.start()

# Log experiment_start event into EMG HDF5
# Note: globalClock may not be defined yet at this stage in the script,
# so we use 0.0 as the origin for the experiment_start event in the EMG file.
{name}.log_event(
    time_sec=0.0,
    name="experiment_start",
    info={{"component": "{name}", "routine": "{self.parentName}"}}
)
"""
        buff.writeIndentedLines(code)

    def writeRoutineEachFrameCode(self, buff):
        """
        Called once per frame during the parent Routine.

        In this version, there is no need for per-frame visualization code,
        because live plotting is handled entirely inside WearLabEMGSession
        using a separate Tk window.
        """
        name = self.params["name"].val

        code = f"""
# EMGRecorderComponent '{name}' does not require per-frame code.
# Live visualization (if enabled) is handled by WearLabEMGSession via a Tk window.
"""
        buff.writeIndentedLines(code)

    def writeRoutineEndCode(self, buff):
        """
        Called when the parent Routine ends.

        No additional cleanup is needed here because the EMG session
        continues across routines and is only stopped at experiment end.
        """
        name = self.params["name"].val

        code = f"""
# No per-Routine cleanup required for EMGRecorderComponent '{name}'.
"""
        buff.writeIndentedLines(code)

    def writeExperimentEndCode(self, buff):
        """
        Called by PsychoPy when generating the script, to write code which
        should run at the end of the experiment.
        """
        name = self.params["name"].val

        code = f"""
# Log experiment_end event and stop WearLab EMG session for '{name}'
try:
    {name}.log_event(
        time_sec=globalClock.getTime(),
        name="experiment_end",
        info={{"component": "{name}"}}
    )
    {name}.stop()
except Exception as _emg_err:
    # Swallow any EMG shutdown errors to avoid breaking the experiment cleanup
    print("Error while stopping EMG session '{{}}': {{}}".format("{name}", _emg_err))
"""
        buff.writeIndentedLines(code)
