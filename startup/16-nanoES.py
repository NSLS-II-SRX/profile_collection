print(f'Loading {__file__}...')


from ophyd import Device, EpicsMotor, EpicsSignal
from ophyd import Component as Cpt


### nano-KB mirrors
class SRXNANOKBFINE(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, 'SPOS') # XF:05IDD-ES:1{Mir:nKBv-Ax:PC}SPOS
    readback = Cpt(EpicsSignalRO, 'RPOS') # XF:05IDD-ES:1{Mir:nKBh-Ax:PC}RPOS

class SRXNANOKB(Device):
    v_y = Cpt(EpicsMotor, 'vert-Ax:Y}Mtr') # XF:05IDD-ES:1{nKB:vert-Ax:Y}Mtr.RBV
    v_pitch = SRXNANOKBFINE('XF:05IDD-ES:1{nKB:vert-Ax:PC}', name='nanoKB_v_pitch') # XF:05IDD-ES:1{nKB:vert-Ax:PC}RPOS
    v_pitch_um = Cpt(EpicsMotor, 'horz-Ax:PC}Mtr') # XF:05IDD-ES:1{nKB:horz-Ax:PC}Mtr.RBV
    h_x = Cpt(EpicsMotor, 'horz-Ax:X}Mtr') # XF:05IDD-ES:1{nKB:horz-Ax:X}Mtr.RBV
    h_pitch = SRXNANOKBFINE('XF:05IDD-ES:1{nKB:horz-Ax:PC}', name='nanoKB_h_pitch') # XF:05IDD-ES:1{nKB:horz-Ax:PC}RPOS
    h_pitch_um = Cpt(EpicsMotor, 'vert-Ax:PC}Mtr') # XF:05IDD-ES:1{nKB:vert-Ax:PC}Mtr.RBV

nanoKB = SRXNANOKB('XF:05IDD-ES:1{nKB:', name='nanoKB')


### High flux sample stages
class SRXNANOSTAGE(Device):
    x = Cpt(EpicsMotor, 'sx}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr.RBV
    y = Cpt(EpicsMotor, 'sy}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.RBV
    z = Cpt(EpicsMotor, 'sz}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:sz}Mtr.RBV
    # sx = Cpt(EpicsMotor, 'ssx}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr.RBV
    # sy = Cpt(EpicsMotor, 'ssy}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:ssy}Mtr.RBV
    # sz = Cpt(EpicsMotor, 'ssz}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:ssz}Mtr.RBV
    th = Cpt(EpicsMotor, 'th}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV
    topx = Cpt(EpicsMotor, 'xth}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.RBV
    topz = Cpt(EpicsMotor, 'zth}Mtr') # XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.RBV

nano_stage = SRXNANOSTAGE('XF:05IDD-ES:1{nKB:Smpl-Ax:', name='nano_stage')


### SDD motion
class SRXNANODET(Device):
    x = Cpt(EpicsMotor, 'X}Mtr') # XF:05IDD-ES:1{nKB:Det-Ax:X}Mtr.RBV
    y = Cpt(EpicsMotor, 'Y}Mtr') # XF:05IDD-ES:1{nKB:Det-Ax:Y}Mtr.RBV
    z = Cpt(EpicsMotor, 'Z}Mtr') # XF:05IDD-ES:1{nKB:Det-Ax:Z}Mtr.RBV

nano_det = SRXNANODET('XF:05IDD-ES:1{nKB:Det-Ax:', name='nano_det')

