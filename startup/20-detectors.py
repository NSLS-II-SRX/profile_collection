from ophyd.controls import ProsilicaDetector, EpicsSignal, EpicsScaler

# CSX-1 Scalar

#sclr = EpicsScaler('XF:23ID1-ES{Sclr:1}', name='sclr', numchan=32)
#sclr_trig = EpicsSignal('XF:23ID1-ES{Sclr:1}.CNT', rw=True,
#                        name='sclr_trig')
#sclr_ch1 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S1', rw=False,
#                       name='sclr_ch1')
#sclr_ch2 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S2', rw=False,
#                       name='sclr_ch2')
#sclr_ch3 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S3', rw=False,
#                       name='sclr_ch3')
#sclr_ch4 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S4', rw=False,
#                       name='sclr_ch4')
#sclr_ch5 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S5', rw=False,
#                       name='sclr_ch5')
#sclr_ch6 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S6', rw=False,
#                       name='sclr_ch6')
#temp_a = EpicsSignal('XF:23ID1-ES{TCtrl:1-Chan:A}T-I', rw=False,
#                     name='temp_a')
#temp_b = EpicsSignal('XF:23ID1-ES{TCtrl:1-Chan:B}T-I', rw=False,
#                     name='temp_b')

# AreaDetector Beam Instrumentation
# diag3_cam = ProsilicaDetector('XF:23ID1-BI{Diag:3-Cam:1}')
# For now, access as simple 'signals'
# diag3_cam = EpicsSignal('XF:23ID1-BI{Diag:3-Cam:1}cam1:Acquire_RBV',
#                         write_pv='XF:23ID1-BI{Diag:3-Cam:1}cam1:Acquire',
#                         rw=True, name='diag3_cam_trigger')

hfm_cam = EpicsSignal('XF:05IDA-BI:1{FS:1-Cam:1}Acquire_RBV',
                        write_pv='XF:05IDA-BI:1{FS:1-Cam:1}Acquire',
                        rw=True, name='hfm_cam_trigger')
hfm_tot1 = EpicsSignal('XF:05IDA-BI:1{FS:1-Cam:1}Stats1:Total_RBV',
                         rw=False, name='hfm_tot1')
bpm1_cam = EpicsSignal('XF:05IDA-BI:1{BPM:1-Cam:1}Acquire_RBV',
                        write_pv='XF:05IDA-BI:1{Mir:1-Cam:1}Acquire',
                        rw=True, name='hfm_cam_trigger')
bpm1_tot1 = EpicsSignal('XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:Total_RBV',
                         rw=False, name='bpm1_tot1')
bpm1_diode0 = EpicsSignal('xf05bpm03:DataRead_Ch1',rw=False,name='bpm1_diode0')
bpm1_diode1 = EpicsSignal('xf05bpm03:DataRead_Ch2',rw=False,name='bpm1_diode1')
bpm1_diode2 = EpicsSignal('xf05bpm03:DataRead_Ch3',rw=False,name='bpm1_diode2')
bpm1_diode3 = EpicsSignal('xf05bpm03:DataRead_Ch4',rw=False,name='bpm1_diode3')
bpm1_femto = EpicsSignal('XF:05IDA-BI:1{IM:1}Int-I',rw=False,name='bpm1_femto')
bpm2_diode0 = EpicsSignal('xf05bpm04:DataRead_Ch1',rw=False,name='bpm1_diode0')
bpm2_diode1 = EpicsSignal('xf05bpm04:DataRead_Ch2',rw=False,name='bpm1_diode1')
bpm2_diode2 = EpicsSignal('xf05bpm04:DataRead_Ch3',rw=False,name='bpm1_diode2')
bpm2_diode3 = EpicsSignal('xf05bpm04:DataRead_Ch4',rw=False,name='bpm1_diode3')
#bpm2_femto = EpicsSignal('',rw=False,name='bpm1_femto')
wbs_t=EpicsSignal('XF:05IDA-BI{BPM:01}AH501:Current1:MeanValue_RBV',rw=False,name='wbs_t')
wbs_b=EpicsSignal('XF:05IDA-BI{BPM:01}AH501:Current2:MeanValue_RBV',rw=False,name='wbs_b')
wbs_i=EpicsSignal('XF:05IDA-BI{BPM:01}AH501:Current3:MeanValue_RBV',rw=False,name='wbs_i')
wbs_o=EpicsSignal('XF:05IDA-BI{BPM:01}AH501:Current4:MeanValue_RBV',rw=False,name='wbs_o')

pbs_t=EpicsSignal('XF:05IDA-BI{BPM:02}AH501:Current1:MeanValue_RBV',rw=False,name='wbs_t')
pbs_b=EpicsSignal('XF:05IDA-BI{BPM:02}AH501:Current2:MeanValue_RBV',rw=False,name='wbs_b')
pbs_i=EpicsSignal('XF:05IDA-BI{BPM:02}AH501:Current3:MeanValue_RBV',rw=False,name='wbs_i')
pbs_o=EpicsSignal('XF:05IDA-BI{BPM:02}AH501:Current4:MeanValue_RBV',rw=False,name='wbs_o')

ssa_t=EpicsSignal('XF:05IDA-BI{BPM:05}AH501:Current1:MeanValue_RBV',rw=False,name='ssa_t')
ssa_b=EpicsSignal('XF:05IDA-BI{BPM:05}AH501:Current2:MeanValue_RBV',rw=False,name='ssa_b')
ssa_i=EpicsSignal('XF:05IDA-BI{BPM:05}AH501:Current3:MeanValue_RBV',rw=False,name='ssa_i')
ssa_o=EpicsSignal('XF:05IDA-BI{BPM:05}AH501:Current4:MeanValue_RBV',rw=False,name='ssa_o')

trans_diode = EpicsSignal('XF:05IDA{IM:1}Cur:I0-I', rw=False, name='f460')
