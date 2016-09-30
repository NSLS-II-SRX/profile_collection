from ophyd import ProsilicaDetector, EpicsSignal, Device
from ophyd import Component as Cpt
from ophyd.ophydobj import StatusBase
from ophyd.status import wait
from hxntools.detectors.zebra import Zebra, EpicsSignalWithRBV

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
#hfvlm_cam = EpicsSignal('XF:05IDD-BI:1{Mscp:1-Cam:1}Acquire_RBV',
#                        write_pv='XF:05IDD-BI:1{Mscp:1-Cam:1}Acquire',
#                        name='hfvlm_cam_trigger')
#hfvlm_tot1 = EpicsSignal('XF:05IDD-BI:1{Mscp:1-Cam:1}Stats1:Total_RBV',
#                         name='hfvlm_tot1')
#hfvlm_tot3 = EpicsSignal('XF:05IDD-BI:1{Mscp:1-Cam:1}Stats3:Total_RBV',
#                         name='hfvlm_tot3')

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

dbpm = DiamondBPM('XF:05ID-BI:1{BPM:01}:')

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
            #print('init ts: {!r}    cur ts : {!r}'.format(init_ts, timestamp))
            #print('old value: {!r}    new value : {!r}'.format(init_ts,
            #                                                   timestamp))

            # if the timestamp or the value has changed, assume it is done
            if (timestamp != init_ts) or (value != old_value):
                ret._finished()
                obj.clear_sub(done_cb)

        self.ch0.subscribe(done_cb, event_type=self.ch0.SUB_VALUE, run=True)

        return ret

class CurrentPreampZebra(Device):
    ch0 = Cpt(EpicsSignalRO, 'Cur:I0-I')
    ch1 = Cpt(EpicsSignalRO, 'Cur:I1-I')
    ch2 = Cpt(EpicsSignalRO, 'Cur:I2-I')
    ch3 = Cpt(EpicsSignalRO, 'Cur:I3-I')

    exp_time = Cpt(EpicsSignal, 'Per-SP')
    initi_trigger = Cpt(EpicsSignal, 'Cmd:Init')
    zebra_trigger = Cpt(EpicsSignal, 'XF:05IDD-ES:1{Dev:Zebra1}:SOFT_IN:B0',
                        add_prefix=())
    zebra_pulse_3_source = Cpt(EpicsSignal, 
                            'XF:05IDD-ES:1{Dev:Zebra1}:PULSE3_INP',
                            add_prefix=())
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.zebra_trigger] = 0
#        self.stage_sigs[self.zebra_pulse_3_source] = 44
        self.stage_sigs[self.zebra_pulse_3_source] = 60

    def stage(self):

        # Customize what is done before every scan (and undone at the end)
        # self.stage_sigs[self.trans_diode] = 5
        # or just use pyepics directly if you need to
        ret = super().stage()
#        self.zebra_pulse_3_source.put(44,wait=True)
#        self.zebra_pulse_3_source.put(60,wait=True)
        self.initi_trigger.put(1, wait=True)
        wait(self.trigger())
#        wait(self.trigger())
        return ret

    def trigger(self):
        init_ts = self.ch0.timestamp
        self.zebra_trigger.put(0,wait = True)
        self.zebra_trigger.put(1,wait = True)
        ret = DeviceStatus(self)

        def done_cb(*args, obj=None, old_value=None, value=None,
                    timestamp=None, **kwargs):
#            print('init ts: {!r}    cur ts : {!r}'.format(init_ts, timestamp))
#            print('old value: {!r}    new value : {!r}'.format(old_value,
#                                                               value))

            # if the timestamp or the value has changed, assume it is done
#            if (timestamp != init_ts) or (value != old_value):
#            if (timestamp != init_ts):
            if (value != old_value):
                ret._finished()
                obj.clear_sub(done_cb)

        self.ch0.subscribe(done_cb, event_type=self.ch0.SUB_VALUE, run=True)

        return ret

current_preamp = CurrentPreampZebra('XF:05IDA{IM:1}', name='current_preamp')
#current_preamp = CurrentPreamp('XF:05IDA{IM:1}', name='current_preamp')

class ZebraPositionCaptureData(Device):
    '''Data arrays for the Zebra position capture function and their metadata.
    '''
    #data arrays
    div1 = Cpt(EpicsSignal, 'PC_DIV1')
    div2 = Cpt(EpicsSignal, 'PC_DIV2')
    div3 = Cpt(EpicsSignal, 'PC_DIV3')
    div4 = Cpt(EpicsSignal, 'PC_DIV4')
    enc1 = Cpt(EpicsSignal, 'PC_ENC1')
    enc2 = Cpt(EpicsSignal, 'PC_ENC2')
    enc3 = Cpt(EpicsSignal, 'PC_ENC3')
    enc4 = Cpt(EpicsSignal, 'PC_ENC4')
    filt1 = Cpt(EpicsSignal, 'PC_FILT1')
    filt2 = Cpt(EpicsSignal, 'PC_FILT2')
    filt3 = Cpt(EpicsSignal, 'PC_FILT3')
    filt4 = Cpt(EpicsSignal, 'PC_FILT4')
    time = Cpt(EpicsSignal, 'PC_TIME')
    #array sizes
    num_cap = Cpt(EpicsSignal, 'PC_NUM_CAP')
    num_down = Cpt(EpicsSignal, 'PC_NUM_DOWN')
    #BOOLs to denote arrays with data
    cap_enc1_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B0') 
    cap_enc2_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B1') 
    cap_enc3_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B2') 
    cap_enc4_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B3') 
    cap_filt1_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B4') 
    cap_filt2_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B5') 
    cap_div1_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B6') 
    cap_div2_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B7') 
    cap_div3_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B8') 
    cap_div4_bool = Cpt(EpicsSignal, 'PC_BIT_CAP:B9') 

class ZebraPositionCapture(Device):
    '''Signals for the position capture function of the Zebra
    '''
    #configuration settings and status PVs
    enc = Cpt(EpicsSignalWithRBV, 'PC_ENC')
    dir = Cpt(EpicsSignalWithRBV, 'PC_DIR')
    tspre = Cpt(EpicsSignalWithRBV, 'PC_TSPRE')
    trig_source = Cpt(EpicsSignalWithRBV, 'PC_ARM_SEL')
    arm = Cpt(EpicsSignal, 'PC_ARM')
    disarm = Cpt(EpicsSignal, 'PC_DISARM')
    armed = Cpt(EpicsSignalRO, 'PC_ARM_OUT')
    gate_source = Cpt(EpicsSignalWithRBV, 'PC_GATE_SEL')
    gate_start = Cpt(EpicsSignalWithRBV, 'PC_GATE_START')
    gate_width = Cpt(EpicsSignalWithRBV, 'PC_GATE_WID')
    gate_step = Cpt(EpicsSignalWithRBV, 'PC_GATE_STEP')
    gate_num = Cpt(EpicsSignalWithRBV, 'PC_GATE_NGATE')
    gated = Cpt(EpicsSignalRO, 'PC_GATE_OUT')
    pulse_source = Cpt(EpicsSignalWithRBV, 'PC_PULSE_SEL')
    pulse_start = Cpt(EpicsSignalWithRBV, 'PC_PULSE_START')
    pulse_width = Cpt(EpicsSignalWithRBV, 'PC_PULSE_WID')
    pulse_step = Cpt(EpicsSignalWithRBV, 'PC_PULSE_STEP')
    pulse_max = Cpt(EpicsSignalWithRBV, 'PC_PULSE_MAX')
    pulse = Cpt(EpicsSignalRO, 'PC_PULSE_OUT')
    
    data=Cpt(ZebraPositionCaptureData,'')


    def stage(self):
        self.arm.put(1)

        super().stage()

    def unstage(self):
        self.disarm.put(1)
        
        super().unstage()
    
class SRXZebra(Zebra):
    '''SRX Zebra device.
    '''

    pc=Cpt(ZebraPositionCapture,'')

    def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
                **kwargs):
        if read_attrs is None:
            read_attrs = []
        if configuration_attrs is None:
            configuration_attrs = [] 

        super().__init__(prefix, read_attrs=read_attrs,
                        configuration_attrs=configuration_attrs, **kwargs)
    
zebra = SRXZebra('XF:05IDD-ES:1{Dev:Zebra1}:', name='zebra')
zebra.read_attrs =['zebra_pc_data_enc1','zebra_pc_data_time']
