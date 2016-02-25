from ophyd import ProsilicaDetector, EpicsSignal, Device
from ophyd import Component as Cpt
from ophyd.ophydobj import StatusBase
from ophyd.status import wait

import time as ttime

hfm_cam = EpicsSignal('XF:05IDA-BI:1{FS:1-Cam:1}Acquire_RBV',
                        write_pv='XF:05IDA-BI:1{FS:1-Cam:1}Acquire',
                        name='hfm_cam_trigger')
hfm_tot1 = EpicsSignal('XF:05IDA-BI:1{FS:1-Cam:1}Stats1:Total_RBV',
                        name='hfm_tot1')
bpm1_cam = EpicsSignal('XF:05IDA-BI:1{BPM:1-Cam:1}Acquire_RBV',
                        write_pv='XF:05IDA-BI:1{Mir:1-Cam:1}Acquire',
                        name='hfm_cam_trigger')
bpm1_tot1 = EpicsSignal('XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:Total_RBV',
                         name='bpm1_tot1')

class BpmStats(Device):
    tot1 = Cpt(EpicsSignal, 'Stats1:Total_RBV')
    tot2 = Cpt(EpicsSignal, 'Stats2:Total_RBV')
    tot3 = Cpt(EpicsSignal, 'Stats3:Total_RBV')
    tot4 = Cpt(EpicsSignal, 'Stats4:Total_RBV')

bpm1_stats = BpmStats('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpm1_stats')

class BpmDiode(Device):
    "Beam Position Monitor Diode"
    diode0 = Cpt(EpicsSignalRO, '_Ch1')
    diode1 = Cpt(EpicsSignalRO, '_Ch2')
    diode2 = Cpt(EpicsSignalRO, '_Ch3')
    diode3 = Cpt(EpicsSignalRO, '_Ch4')
    # femto = EpicsSignal('XF:05IDA-BI:1{IM:1}Int-I')

    def trigger(self):
        # There is nothing to do. Just report that we are done.
        # Note: This really should not necessary to do --
        # future changes to PVPositioner may obviate this code.
        status = StatusBase()
        status._finished()
        return status


bpm1 = BpmDiode('xf05bpm03:DataRead', name='bpm1')
bpm2 = BpmDiode('xf05bpm04:DataRead', name='bpm2')


class SlitDrainCurrent(Device):
    t = Cpt(EpicsSignalRO, 'Current1:MeanValue_RBV')
    b = Cpt(EpicsSignalRO, 'Current2:MeanValue_RBV')
    i = Cpt(EpicsSignalRO, 'Current3:MeanValue_RBV')
    o = Cpt(EpicsSignalRO, 'Current4:MeanValue_RBV')

    def trigger(self):
        # There is nothing to do. Just report that we are done.
        # Note: This really should not necessary to do --
        # future changes to PVPositioner may obviate this code.
        status = StatusBase()
        status._finished()
        return status


wbs = SlitDrainCurrent('XF:05IDA-BI{BPM:01}AH501:', name='wbs')
pbs= SlitDrainCurrent('XF:05IDA-BI{BPM:02}AH501:', name='pbs')
ssa = SlitDrainCurrent('XF:05IDA-BI{BPM:05}AH501:', name='ssa')


class CurrentPreamp(Device):
    ch0 = Cpt(EpicsSignalRO, 'Cur:I0-I')
    ch1 = Cpt(EpicsSignalRO, 'Cur:I1-I')
    ch2 = Cpt(EpicsSignalRO, 'Cur:I2-I')
    ch3 = Cpt(EpicsSignalRO, 'Cur:I3-I')

    exp_time = Cpt(EpicsSignal, 'Per-SP')
    initi_trigger = Cpt(EpicsSignal, 'Cmd:Init')
    event_receiver = Cpt(EpicsSignal,
                         'XF:05IDD-ES:1{EVR:1-Out:FP3}Src:Scale-SP',
                         add_prefix=())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.event_receiver] = 'Force Low'
        #self.stage_sigs[self.initi_trigger] = 1 #this somewhat did not work

    def stage(self):

        # Customize what is done before every scan (and undone at the end)
        # self.stage_sigs[self.trans_diode] = 5
        # or just use pyepics directly if you need to
        ret = super().stage()
        self.initi_trigger.put(1, wait=True)
        wait(self.trigger())
        return ret

    def trigger(self):
        init_ts = self.ch0.timestamp
        self.event_receiver.put('Force Low', wait=True)
        self.event_receiver.put('Force High', wait=True)
        self.event_receiver.put('Force Low')
        ret = DeviceStatus(self)

        def done_cb(*args, obj=None, old_value=None, value=None,
                    timestamp=None, **kwargs):
            print('init ts: {!r}    cur ts : {!r}'.format(init_ts, timestamp))
            print('old value: {!r}    new value : {!r}'.format(init_ts,
                                                               timestamp))

            # if the timestamp or the value has changed, assume it is done
            if (timestamp != init_ts) or (value != old_value):
                ret._finished()
                obj.clear_sub(done_cb)

        self.ch0.subscribe(done_cb, event_type=self.ch0.SUB_VALUE, run=True)

        return ret

current_preamp = CurrentPreamp('XF:05IDA{IM:1}', name='current_preamp')
