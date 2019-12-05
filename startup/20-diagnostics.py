from ophyd import ProsilicaDetector, EpicsSignal, Device, EpicsScaler, TetrAMM, EpicsSignalRO
from ophyd import Component as Cpt
from ophyd.device import (Component as C, DynamicDeviceComponent as DDC)
from ophyd.ophydobj import StatusBase
from ophyd.status import wait
from hxntools.detectors.zebra import Zebra, EpicsSignalWithRBV
from collections import OrderedDict

import time as ttime
import threading


### BPM1 Statistics
class BpmStats(Device):
    tot1 = Cpt(EpicsSignal, 'Stats1:Total_RBV')
    tot2 = Cpt(EpicsSignal, 'Stats2:Total_RBV')
    tot3 = Cpt(EpicsSignal, 'Stats3:Total_RBV')
    tot4 = Cpt(EpicsSignal, 'Stats4:Total_RBV')

bpm1_stats = BpmStats('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpm1_stats')


### BPM Diode
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

# bpm1 = BpmDiode('xf05bpm03:DataRead', name='bpm1')
# bpm2 = BpmDiode('xf05bpm04:DataRead', name='bpm2')
# BPM IOC disabled 2019-04-15
# bpm1 = TetrAMM('XF:05IDA-BI{BPM:3}',name='bpm1')
# bpm2 = TetrAMM('XF:05IDA-BI{BPM:4}',name='bpm2')


### Diamond BPM
class DiamondBPM(Device):
    diode_top = Cpt(EpicsSignalRO, 'Current1:MeanValue_RBV')
    diode_inb = Cpt(EpicsSignalRO, 'Current2:MeanValue_RBV')
    diode_out = Cpt(EpicsSignalRO, 'Current3:MeanValue_RBV')
    diode_bot = Cpt(EpicsSignalRO, 'Current4:MeanValue_RBV')
    sigma_top = Cpt(EpicsSignalRO, 'Current1:Sigma_RBV')
    sigma_inb = Cpt(EpicsSignalRO, 'Current2:Sigma_RBV')
    sigma_out = Cpt(EpicsSignalRO, 'Current3:Sigma_RBV')
    sigma_bot = Cpt(EpicsSignalRO, 'Current4:Sigma_RBV')
    x_pos = Cpt(EpicsSignalRO, 'PosX:MeanValue_RBV')
    y_pos = Cpt(EpicsSignalRO, 'PosY:MeanValue_RBV')
    x_sigma = Cpt(EpicsSignalRO, 'PosX:Sigma_RBV')
    y_sigma = Cpt(EpicsSignalRO, 'PosY:Sigma_RBV')

dbpm = DiamondBPM('XF:05ID-BI:1{BPM:01}:', name='dbpm')


### Setup Slit Drain Current
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



### This device is no longer used
# Commenting out before deleting
# class CurrentPreamp(Device):
#     ch0 = Cpt(EpicsSignalRO, 'Cur:I0-I')
#     ch1 = Cpt(EpicsSignalRO, 'Cur:I1-I')
#     ch2 = Cpt(EpicsSignalRO, 'Cur:I2-I')
#     ch3 = Cpt(EpicsSignalRO, 'Cur:I3-I')
# 
#     exp_time = Cpt(EpicsSignal, 'Per-SP')
#     initi_trigger = Cpt(EpicsSignal, 'Cmd:Init')
#     event_receiver = Cpt(EpicsSignal,
#                          'XF:05IDD-ES:1{EVR:1-Out:FP3}Src:Scale-SP',
#                          add_prefix=())
# 
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stage_sigs[self.event_receiver] = 'Force Low'
#         #self.stage_sigs[self.initi_trigger] = 1 #this somewhat did not work
# 
#     def stage(self):
# 
#         # Customize what is done before every scan (and undone at the end)
#         # self.stage_sigs[self.trans_diode] = 5
#         # or just use pyepics directly if you need to
#         ret = super().stage()
#         self.initi_trigger.put(1, wait=True)
#         wait(self.trigger())
#         return ret
# 
#     def trigger(self):
#         init_ts = self.ch0.timestamp
#         self.event_receiver.put('Force Low', wait=True)
#         self.event_receiver.put('Force High', wait=True)
#         self.event_receiver.put('Force Low')
#         ret = DeviceStatus(self)
# 
#         def done_cb(*args, obj=None, old_value=None, value=None,
#                     timestamp=None, **kwargs):
#             #print('init ts: {!r}    cur ts : {!r}'.format(init_ts, timestamp))
#             #print('old value: {!r}    new value : {!r}'.format(init_ts,
#             #                                                   timestamp))
# 
#             # if the timestamp or the value has changed, assume it is done
#             if (timestamp != init_ts) or (value != old_value):
#                 ret._finished()
#                 obj.clear_sub(done_cb)
# 
#         self.ch0.subscribe(done_cb, event_type=self.ch0.SUB_VALUE, run=True)
# 
#         return ret

