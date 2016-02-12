from ophyd.mca import (EpicsMCA, EpicsDXP)
from ophyd import (Component as Cpt, Device, EpicsSignal, EpicsSignalWithRBV)


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


class SRXSaturn(Saturn):
    pass


if __name__ == '__main__':
    from ophyd.commands import setup_ophyd
    setup_ophyd()

    saturn = SRXSaturn('dxpSaturn:', name='saturn')
