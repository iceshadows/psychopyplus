#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PsychoPy Builder component for WearLab EMG recording.

This component:

- Creates and starts a WearLabEMGSession at experiment start.
- Automatically derives the EMG HDF5 base filename from thisExp.dataFileName.
- Logs "experiment_start" and "experiment_end" events automatically.
- Exposes the session instance as a variable with the component's name,
  so Code components can call:

    myEmgComponent.log_event(globalClock.getTime(), "trial_begin", {...})

To use this component, place this file in a folder which you add to:

    PsychoPy → Preferences → Builder → Components paths
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
    tooltip = _translate('EMGRecorder: A component for recording Electromyography '
                         '(EMG) signals during your experiment\n'
                         '(requires WearLab™️ EMG hardware and drivers)')

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

        # This component does not draw stimuli, so start/stop time are mostly
        # irrelevant, but we keep default timing params for consistency.

        # ------------------------------------------------------------------ #
        # Custom Builder parameters (dialog fields)
        # ------------------------------------------------------------------ #

        # Sampling period sent to the device firmware (ms)
        self.params["samplePeriodMs"] = Param(
            "1000",
            valType="code",
            allowedTypes=[],
            hint=_translate(
                "Sampling period in milliseconds passed to the device "
                "(e.g. 1000 = 1000 ms, 1 = 1 ms, depending on firmware semantics)."
            ),
            label=_translate("Sample period (ms)"),
        )

        # Number of EMG channels
        self.params["nChannels"] = Param(
            "16",
            valType="code",
            allowedTypes=[],
            hint=_translate("Number of EMG channels."),
            label=_translate("Channels"),
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

        # Extend the order so these appear near the top of the dialog
        self.order += ["samplePeriodMs", "nChannels", "fileSuffix"]

    # ------------------------------------------------------------------ #
    # Code generation helpers
    # ------------------------------------------------------------------ #

    def writeInitCode(self, buff):
        """
        Called by PsychoPy when generating the script, to write code which
        should run at the beginning of the experiment.
        """
        name = self.params["name"].val
        period = self.params["samplePeriodMs"].val
        n_channels = self.params["nChannels"].val
        suffix = self.params["fileSuffix"].val

        # Note: `thisExp` and `globalClock` are part of the standard
        # PsychoPy generated script environment.
        code = f"""
# Initialize WearLab EMG recording session for component '{name}'
from psychopy.hardware.wearlab_emg import WearLabEMGSession

{name}_fileBase = thisExp.dataFileName + {suffix}
{name} = WearLabEMGSession(
    base_filename={name}_fileBase,
    semg_cycle_ms={period},
    channels=int({n_channels})
)
{name}.start()

# Log experiment_start event into EMG HDF5
{name}.log_event(
    time_sec=0.0,
    name="experiment_start",
    info={{"component": "{name}", "routine": "{self.parentName}"}}
)
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
