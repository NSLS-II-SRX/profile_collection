
class CurrentPreampZebra(Device):
    ch0 = Cpt(EpicsSignalRO, 'Cur:I0-I')
    ch1 = Cpt(EpicsSignalRO, 'Cur:I1-I')
    ch2 = Cpt(EpicsSignalRO, 'Cur:I2-I')
    ch3 = Cpt(EpicsSignalRO, 'Cur:I3-I')

    #exp_time = Cpt(EpicsSignal, 'Per-SP')
    exp_time = Cpt(EpicsSignal, 'XF:05IDD-ES:1{Dev:Zebra1}:PULSE3_WID',
                    add_prefix=())
    trigger_mode = Cpt(EpicsSignal, 'Cmd:TrigMode')
    initi_trigger = Cpt(EpicsSignal, 'Cmd:Init')
    zebra_trigger = Cpt(EpicsSignal, 'XF:05IDD-ES:1{Dev:Zebra1}:SOFT_IN:B0',
                        add_prefix=())
    zebra_pulse_3_source = Cpt(EpicsSignal,
                            'XF:05IDD-ES:1{Dev:Zebra1}:PULSE3_INP',
                            add_prefix=())

    current_scan_rate = Cpt(EpicsSignal, 'Cmd:RdCur.SCAN')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.zebra_trigger] = 0
#        self.stage_sigs[self.zebra_pulse_3_source] = 44
        self.stage_sigs[self.zebra_pulse_3_source] = 60

        self.current_scan_rate.put(9)
        #update
        # self.trigger_mode.put(5)
        self.stage_sigs[self.trigger_mode] = 5  # fix this
        self.initi_trigger.put(1, wait=True)

    def stage(self):

        # Customize what is done before every scan (and undone at the end)
        # self.stage_sigs[self.trans_diode] = 5
        # or just use pyepics directly if you need to
        ret = super().stage()
        self.initi_trigger.put(1, wait=True)
        return ret

    def trigger(self):
        init_ts = self.ch0.timestamp
        timeout = float(self.exp_time.get() + .8)

        def retrigger():
            print("[WW] Re-triggered ion chamber;  I0 for this point is suspect.")
            self.zebra_trigger.put(0,wait=True)
            self.zebra_trigger.put(1,wait=True)
        def done_cb(*args, obj=None, old_value=None, value=None,
                    timestamp=None, **kwargs):
            # if the value has changed, assume it is done
            if (value != old_value):
                tmr.cancel()
                ret._finished()
                obj.clear_sub(done_cb)

        tmr = threading.Timer(timeout,retrigger)
        tmr.start()
        ret = DeviceStatus(self)

        self.ch0.subscribe(done_cb, event_type=self.ch0.SUB_VALUE, run=False)
        self.zebra_trigger.put(0, wait=True)
        self.zebra_trigger.put(1, wait=True)

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
    enc_pos1_sync = Cpt(EpicsSignal, 'M1:SETPOS.PROC')
    enc_pos2_sync = Cpt(EpicsSignal, 'M2:SETPOS.PROC')
    enc_pos3_sync = Cpt(EpicsSignal, 'M3:SETPOS.PROC')
    enc_pos4_sync = Cpt(EpicsSignal, 'M4:SETPOS.PROC')
    enc_res1 = Cpt(EpicsSignal, 'M1:MRES')
    enc_res2 = Cpt(EpicsSignal, 'M2:MRES')
    enc_res3 = Cpt(EpicsSignal, 'M3:MRES')
    enc_res4 = Cpt(EpicsSignal, 'M4:MRES')
    data_in_progress = Cpt(EpicsSignalRO, 'ARRAY_ACQ')

    block_state_reset = Cpt(EpicsSignal, 'SYS_RESET.PROC')

    data=Cpt(ZebraPositionCaptureData,'')


    def stage(self):
        self.arm.put(1)

        super().stage()

    def unstage(self):
        self.disarm.put(1)
        self.block_state_reset.put(1)

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
zebra.read_attrs = ['pc.data.enc1', 'pc.data.time']
