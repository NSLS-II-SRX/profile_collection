print(f'Loading {__file__}...')


from ophyd import EpicsMotor, Device
from ophyd import Component as Cpt


### Downstream detector stages
class HFETomoStage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')

e_tomo = HFETomoStage('XF:05IDD-ES:1{Stg:Det2-Ax:', name = 'e_tomo')


### XFM stage setup for cryo/confocal experiments
class XFMstage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')

stage = XFMstage('XF:05IDD-ES:1{Mscp:1-Ax:', name='stage')

