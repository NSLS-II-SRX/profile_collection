print(f'Loading {__file__}...')


from ophyd import EpicsMotor, Device
from ophyd import Component as Cpt


# Downstream detector stages
class HFETomoStage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')


e_tomo = HFETomoStage('XF:05IDD-ES:1{Stg:Det2-Ax:', name='e_tomo')


# XFM stage setup for cryo/confocal experiments
class XFMstage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')

confocal_stage = XFMstage('XF:05IDD-ES:1{Stg:Cxrf-Ax:', name='confocal_stage')


# XFM slit setup for cryo/confocal experiments
class XFMslit(Device):
    h_trans = Cpt(EpicsMotor, 'HT}Mtr')
    h_gap = Cpt(EpicsMotor, 'HA}Mtr')
    v_trans = Cpt(EpicsMotor, 'VT}Mtr')
    v_gap = Cpt(EpicsMotor, 'VA}Mtr')

confocal_slit = XFMslit('XF:05IDD-OP:1{Slt:Cxrf-Ax:', name='confocal_slit')


# Stage setup for Dexela beamstop
class DexelaBeamstop(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')


dexela_bs = DexelaBeamstop('XF:05IDD-ES:1{Det:2-Ax:', name='dexela_bs')

