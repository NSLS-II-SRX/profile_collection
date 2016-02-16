from ophyd.mca import (EpicsMCA, EpicsDXP)
from ophyd import (Component as Cpt, Device, EpicsSignal, EpicsSignalWithRBV,
                   DeviceStatus)
from ophyd.device import (BlueskyInterface, Staged)


class SaturnMCA(EpicsMCA):
    check_acquiring = Cpt(EpicsSignal, 'CheckACQG')
    client_wait = Cpt(EpicsSignal, 'ClientWait')
    collect_data = Cpt(EpicsSignal, 'CollectData')
    enable_wait = Cpt(EpicsSignal, 'EnableWait')
    erase = Cpt(EpicsSignal, 'Erase')
    erase_start = Cpt(EpicsSignal, 'EraseStart')
    read = Cpt(EpicsSignal, 'Read')
    read_callback = Cpt(EpicsSignal, 'ReadCallback')
    read_data_once = Cpt(EpicsSignal, 'ReadDataOnce')
    read_status_once = Cpt(EpicsSignal, 'ReadStatusOnce')
    set_client_wait = Cpt(EpicsSignal, 'SetClientWait')
    start = Cpt(EpicsSignal, 'Start')
    status = Cpt(EpicsSignal, 'Status')
    stop = Cpt(EpicsSignal, 'Stop')
    when_acq_stops = Cpt(EpicsSignal, 'WhenAcqStops')
    why1 = Cpt(EpicsSignal, 'Why1')
    why2 = Cpt(EpicsSignal, 'Why2')
    why3 = Cpt(EpicsSignal, 'Why3')
    why4 = Cpt(EpicsSignal, 'Why4')


class SaturnDXP(EpicsDXP):
    baseline_energy_array = Cpt(EpicsSignal, 'BaselineEnergyArray')
    baseline_histogram = Cpt(EpicsSignal, 'BaselineHistogram')
    calibration_energy = Cpt(EpicsSignal, 'CalibrationEnergy_RBV')
    current_pixel = Cpt(EpicsSignal, 'CurrentPixel')
    dynamic_range = Cpt(EpicsSignal, 'DynamicRange_RBV')
    elapsed_live_time = Cpt(EpicsSignal, 'ElapsedLiveTime')
    elapsed_real_time = Cpt(EpicsSignal, 'ElapsedRealTime')
    elapsed_trigger_live_time = Cpt(EpicsSignal, 'ElapsedTriggerLiveTime')
    energy_threshold = Cpt(EpicsSignalWithRBV, 'EnergyThreshold')
    gap_time = Cpt(EpicsSignalWithRBV, 'GapTime')
    max_width = Cpt(EpicsSignalWithRBV, 'MaxWidth')
    mca_bin_width = Cpt(EpicsSignal, 'MCABinWidth_RBV')
    num_ll_params = Cpt(EpicsSignal, 'NumLLParams')
    peaking_time = Cpt(EpicsSignalWithRBV, 'PeakingTime')
    preset_events = Cpt(EpicsSignalWithRBV, 'PresetEvents')
    preset_mode = Cpt(EpicsSignal, 'PresetMode')
    preset_mode = Cpt(EpicsSignal, 'PresetMode_RBV')
    preset_triggers = Cpt(EpicsSignalWithRBV, 'PresetTriggers')
    read_ll_params = Cpt(EpicsSignal, 'ReadLLParams')
    trace_data = Cpt(EpicsSignal, 'TraceData')
    trace_mode = Cpt(EpicsSignalWithRBV, 'TraceMode')
    trace_time_array = Cpt(EpicsSignal, 'TraceTimeArray')
    trace_time = Cpt(EpicsSignalWithRBV, 'TraceTime')
    trigger_gap_time = Cpt(EpicsSignalWithRBV, 'TriggerGapTime')
    trigger_peaking_time = Cpt(EpicsSignalWithRBV, 'TriggerPeakingTime')
    trigger_threshold = Cpt(EpicsSignalWithRBV, 'TriggerThreshold')


class Saturn(Device):
    dxp = Cpt(SaturnDXP, 'dxp1:')
    mca = Cpt(SaturnMCA, 'mca1')

    channel_advance = Cpt(EpicsSignal, 'ChannelAdvance')
    client_wait = Cpt(EpicsSignal, 'ClientWait')
    dwell = Cpt(EpicsSignal, 'Dwell')
    max_scas = Cpt(EpicsSignal, 'MaxSCAs')
    num_scas = Cpt(EpicsSignalWithRBV, 'NumSCAs')
    poll_time = Cpt(EpicsSignalWithRBV, 'PollTime')
    prescale = Cpt(EpicsSignal, 'Prescale')
    save_system = Cpt(EpicsSignalWithRBV, 'SaveSystem')
    save_system_file = Cpt(EpicsSignal, 'SaveSystemFile')
    set_client_wait = Cpt(EpicsSignal, 'SetClientWait')


class SaturnSoftTrigger(BlueskyInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = None
        self._acquisition_signal = self.mca.erase_start

        self.stage_sigs[self.dxp.preset_mode] = 'Real time'
        self._count_signal = self.mca.preset_real_time
        self._count_time = None

    def stage(self):
        self._acquisition_signal.subscribe(self._acquire_changed)
        if self._count_time is not None:
            self.stage_sigs[self._count_signal] = self._count_time

        super().stage()

    def unstage(self):
        try:
            super().unstage()
        finally:
            self._acquisition_signal.clear_sub(self._acquire_changed)

            if self._count_signal in self.stage_sigs:
                del self.stage_sigs[self._count_signal]
                self._count_time = None

    def trigger(self):
        "Trigger one acquisition."
        if self._staged != Staged.yes:
            raise RuntimeError("This detector is not ready to trigger."
                               "Call the stage() method before triggering.")

        self._status = DeviceStatus(self)
        self._acquisition_signal.put(1, wait=False)
        return self._status

    def _acquire_changed(self, value=None, old_value=None, **kwargs):
        "This is called when the 'acquire' signal changes."
        if self._status is None:
            return
        if (old_value == 1) and (value == 0):
            # Negative-going edge means an acquisition just finished.
            self._status._finished()
            self._status = None

    @property
    def count_time(self):
        '''Exposure time, as set by bluesky'''
        return self._count_time

    @count_time.setter
    def count_time(self, count_time):
        self._count_time = count_time


class SRXSaturn(SaturnSoftTrigger, Saturn):
    def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
                 **kwargs):
        if read_attrs is None:
            read_attrs = ['mca.spectrum']

        if configuration_attrs is None:
            configuration_attrs = ['mca.preset_real_time',
                                   'mca.live_real_time',
                                   'dxp.preset_mode',
                                   ]

        super().__init__(prefix, read_attrs=read_attrs,
                         configuration_attrs=configuration_attrs, **kwargs)


if __name__ == '__main__':
    from ophyd.commands import setup_ophyd
    setup_ophyd()

    saturn = SRXSaturn('dxpSaturn:', name='saturn')
