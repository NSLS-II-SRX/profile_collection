from ophyd.controls import ProsilicaDetector, EpicsSignal, EpicsScaler

# CSX-1 Scalar

sclr = EpicsScaler('XF:23ID1-ES{Sclr:1}', name='sclr', numchan=32)
sclr_trig = EpicsSignal('XF:23ID1-ES{Sclr:1}.CNT', rw=True,
                        name='sclr_trig')
sclr_ch1 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S1', rw=False,
                       name='sclr_ch1')
sclr_ch2 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S2', rw=False,
                       name='sclr_ch2')
sclr_ch3 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S3', rw=False,
                       name='sclr_ch3')
sclr_ch4 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S4', rw=False,
                       name='sclr_ch4')
sclr_ch5 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S5', rw=False,
                       name='sclr_ch5')
sclr_ch6 = EpicsSignal('XF:23ID1-ES{Sclr:1}.S6', rw=False,
                       name='sclr_ch6')
temp_a = EpicsSignal('XF:23ID1-ES{TCtrl:1-Chan:A}T-I', rw=False,
                     name='temp_a')
temp_b = EpicsSignal('XF:23ID1-ES{TCtrl:1-Chan:B}T-I', rw=False,
                     name='temp_b')

# AreaDetector Beam Instrumentation
diag3_cam = ProsilicaDetector('XF:23ID1-BI{Diag:3-Cam:1}')
# For now, access as simple 'signals'
# diag3_cam = EpicsSignal('XF:23ID1-BI{Diag:3-Cam:1}cam1:Acquire_RBV',
#                         write_pv='XF:23ID1-BI{Diag:3-Cam:1}cam1:Acquire',
#                         rw=True, name='diag3_cam_trigger')

diag5_cam = EpicsSignal('XF:23ID1-BI{Diag:5-Cam:1}cam1:Acquire_RBV',
                        write_pv='XF:23ID1-BI{Diag:5-Cam:1}cam1:Acquire',
                        rw=True, name='diag5_cam_trigger')

diag3_tot1 = EpicsSignal('XF:23ID1-BI{Diag:3-Cam:1}Stats1:Total_RBV',
                         rw=False, name='diag3_tot1')
diag3_tot5 = EpicsSignal('XF:23ID1-BI{Diag:3-Cam:1}Stats5:Total_RBV',
                         rw=False, name='diag3_tot5')

diag5_tot1 = EpicsSignal('XF:23ID1-BI{Diag:5-Cam:1}Stats1:Total_RBV',
                         rw=False, name='diag5_tot1')
diag5_tot5 = EpicsSignal('XF:23ID1-BI{Diag:5-Cam:1}Stats5:Total_RBV',
                         rw=False, name='diag5_tot5')

pimte_cam = EpicsSignal('XF:23ID1-ES{Dif-Cam:PIMTE}cam1:Acquire_RBV',
                        write_pv='XF:23ID1-ES{Dif-Cam:PIMTE}cam1:Acquire',
                        rw=True, name='pimte_cam_trigger')
pimte_tot1 = EpicsSignal('XF:23ID1-ES{Dif-Cam:PIMTE}Stats1:Total_RBV',
                         rw=False, name='pimte_tot1')
pimte_tot2 = EpicsSignal('XF:23ID1-ES{Dif-Cam:PIMTE}Stats2:Total_RBV',
                         rw=False, name='pimte_tot2')
pimte_tot3 = EpicsSignal('XF:23ID1-ES{Dif-Cam:PIMTE}Stats3:Total_RBV',
                         rw=False, name='pimte_tot3')
pimte_tot4 = EpicsSignal('XF:23ID1-ES{Dif-Cam:PIMTE}Stats4:Total_RBV',
                         rw=False, name='pimte_tot4')
pimte_tot5 = EpicsSignal('XF:23ID1-ES{Dif-Cam:PIMTE}Stats5:Total_RBV',
                         rw=False, name='pimte_tot5')
