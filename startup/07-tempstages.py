from ophyd import EpicsMotor, EpicsSignalRO, EpicsSignal, PVPositionerPC
from ophyd import Device
from ophyd import Component as Cpt

# XFM stage setup for cryo experiment
class XFMstage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')

stage = XFMstage('XF:05IDD-ES:1{Stg:XFM1-Ax:', name='stage')
relabel_motors(stage)
