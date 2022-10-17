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
    Fe_shutter = Cpt(EpicsSignal, '1-Cmd')
    Cu_shutter = Cpt(EpicsSignal, '2-Cmd')
    Si_shutter = Cpt(EpicsSignal, '3-Cmd')
    Mo_shutter = Cpt(EpicsSignal, '4-Cmd')

attenuators = SRXAttenuators('XF:05IDD-ES{IO:2}DO:', name='attenuators')
# shut_d = attenuators.Mo_shutter


# micro-KB mirrors from XFM
class SRXMicroKB(Device):
    KBv_y = Cpt(EpicsMotor, 'KBv-Ax:TY}Mtr')
    KBv_pitch = Cpt(EpicsMotor, 'KBv-Ax:Pitch}Mtr')
    KBv_USB = Cpt(EpicsMotor, 'KBv-Ax:UsB}Mtr')
    KBv_DSB = Cpt(EpicsMotor, 'KBv-Ax:DsB}Mtr')
    KBh_x = Cpt(EpicsMotor, 'KBv-Ax:TX}Mtr')
    KBh_pitch = Cpt(EpicsMotor, 'KBv-Ax:Pitch}Mtr')
    KBh_USB = Cpt(EpicsMotor, 'KBv-Ax:UsB}Mtr')
    KBh_DSB = Cpt(EpicsMotor, 'KBv-Ax:DsB}Mtr')


microKB = SRXMicroKB('XF:05IDD-OP:1{Mir:', name='microKB')


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


# SDD motion
class SRXUpStreamGantry(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')


sdd_pos = SRXUpStreamGantry('XF:05IDD-ES:1{Det:1-Ax:', name='sdd_pos')


# PCOEdge detector motion
class SRXDownStreamGantry(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')
    focus = Cpt(EpicsMotor, 'Foc}Mtr')


pcoedge_pos = SRXDownStreamGantry('XF:05IDD-ES:1{Det:3-Ax:',
                                  name='pcoedge_pos')
