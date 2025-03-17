print(f'Loading {__file__}...')


import os
from ophyd import EpicsMotor, EpicsSignal
from ophyd import Device
from ophyd import Component as Cpt


# JJ Slits
class SRXJJSlits(Device):
    h_gap = Cpt(EpicsMotor, 'HA}Mtr')
    h_trans = Cpt(EpicsMotor, 'HT}Mtr')
    v_gap = Cpt(EpicsMotor, 'VA}Mtr')
    v_trans = Cpt(EpicsMotor, 'VT}Mtr')


jjslits = SRXJJSlits('XF:05IDD-OP:1{Slt:KB-Ax:', name='jjslits')


# Attenuator box
class SRXAttenuators(Device):
    Al_050um = Cpt(EpicsSignal, '8-Cmd')  # XF:05IDD-ES{IO:4}DO:8-Cmd
    Al_100um = Cpt(EpicsSignal, '7-Cmd')  # XF:05IDD-ES{IO:4}DO:7-Cmd
    Al_250um = Cpt(EpicsSignal, '6-Cmd')  # XF:05IDD-ES{IO:4}DO:6-Cmd
    Al_500um = Cpt(EpicsSignal, '5-Cmd')  # XF:05IDD-ES{IO:4}DO:5-Cmd
    Si_250um = Cpt(EpicsSignal, '2-Cmd')  # XF:05IDD-ES{IO:4}DO:2-Cmd
    Si_650um = Cpt(EpicsSignal, '1-Cmd')  # XF:05IDD-ES{IO:4}DO:1-Cmd


attenuators = SRXAttenuators('XF:05IDD-ES{IO:4}DO:', name='attenuators')


# High flux sample stages
class HFSampleStage(Device):
    x = Cpt(EpicsMotor, '{Stg:Smpl2-Ax:X}Mtr')
    y = Cpt(EpicsMotor, '{Stg:Smpl2-Ax:Y}Mtr')
    z = Cpt(EpicsMotor, '{Stg:Smpl1-Ax:Z}Mtr')
    th = Cpt(EpicsMotor, '{Smpl:1-Ax:Rot}Mtr')
    topx = Cpt(EpicsMotor, '{Smpl:1-Ax:XF}Mtr')
    topz = Cpt(EpicsMotor, '{Smpl:1-Ax:ZF}Mtr')

    RETRY_DEADBAND_X = Cpt(EpicsSignal,
                           'XF:05IDD-ES:1{Stg:Smpl2-Ax:X}Mtr.RDBD',
                           add_prefix=())
    RETRY_DEADBAND_Y = Cpt(EpicsSignal,
                           'XF:05IDD-ES:1{Stg:Smpl2-Ax:Y}Mtr.RDBD',
                           add_prefix=())
    _RETRY_DEADBAND_DEFAULT = 0.0001

    BACKLASH_SPEED_X = Cpt(EpicsSignal,
                           'XF:05IDD-ES:1{Stg:Smpl2-Ax:X}Mtr.BVEL',
                           add_prefix=())
    BACKLASH_SPEED_Y = Cpt(EpicsSignal,
                           'XF:05IDD-ES:1{Stg:Smpl2-Ax:Y}Mtr.BVEL',
                           add_prefix=())
    _BACKLASH_SPEED_DEFAULT = 0.1

    def reset_stage_defaults(self):
        yield from mv(self.RETRY_DEADBAND_X, self._RETRY_DEADBAND_DEFAULT,
                      self.RETRY_DEADBAND_Y, self._RETRY_DEADBAND_DEFAULT,
                      self.BACKLASH_SPEED_X, self._BACKLASH_SPEED_DEFAULT,
                      self.BACKLASH_SPEED_Y, self._BACKLASH_SPEED_DEFAULT)


hf_stage = HFSampleStage('XF:05IDD-ES:1', name='hf_stage')
if 'velocity' not in hf_stage.x.configuration_attrs:
    hf_stage.x.configuration_attrs.append('velocity')


# PCOEdge detector motion
class SRXDownStreamGantry(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')
    focus = Cpt(EpicsMotor, 'Foc}Mtr')


pcoedge_pos = SRXDownStreamGantry('XF:05IDD-ES:1{Det:3-Ax:',
                                  name='pcoedge_pos')
