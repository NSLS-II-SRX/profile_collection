from ophyd import EpicsMotor
from ophyd import Device
from ophyd import Component as Cpt

# Slits
class SRXSlits4(Device):
    top = Cpt(EpicsMotor, 'T}Mtr')
    bot = Cpt(EpicsMotor, 'B}Mtr')    
    inb = Cpt(EpicsMotor, 'I}Mtr')
    out = Cpt(EpicsMotor, 'O}Mtr')

class SRXSlits2(Device):
    inb = Cpt(EpicsMotor, 'I}Mtr')
    out = Cpt(EpicsMotor, 'O}Mtr')

# White Beam slits
slt_wb = SRXSlits4('XF:05IDA-OP:1{Slt:1-Ax:', name='slt_wb')
# Pink beam
slt_pb = SRXSlits2('XF:05IDA-OP:1{Slt:2-Ax:', name='slt_pb')
# Secondary source
slt_ssa = SRXSlits4('XF:05IDB-OP:1{Slt:SSA-Ax:', name='slt_ssa')

relabel_motors(slt_wb)
relabel_motors(slt_ssa)
relabel_motors(slt_pb)


# HFM Mirror
class SRXHFM(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    pitch = Cpt(EpicsMotor, 'P}Mtr')
    bend = Cpt(EpicsMotor, 'Bend}Mtr')

hfm = SRXHFM('XF:05IDA-OP:1{Mir:1-Ax:', name='hfm')
relabel_motors(hfm)


# HDCM
class SRXDCM(Device):
    bragg = Cpt(EpicsMotor, 'P}Mtr')
    c1_roll = Cpt(EpicsMotor, 'R1}Mtr')
    c2_x = Cpt(EpicsMotor, 'X2}Mtr')
    c2_ptich = Cpt(EpicsMotor, 'P2}Mtr')
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')

dcm = SRXDCM('XF:05IDA-OP:1{Mono:HDCM-Ax:' , name='dcm')
relabel_motors(dcm)


# BPMs
class SRXBPM(Device):
    y = Cpt(EpicsMotor, 'YFoil}Mtr')
    diode_x = Cpt(EpicsMotor, 'XDiode}Mtr')
    diode_y = Cpt(EpicsMotor, 'YDiode}Mtr')


bpm1_pos = SRXBPM('XF:05IDA-BI:1{BPM:1-Ax:', name='bpm1_pos')
bpm2_pos = SRXBPM('XF:05IDB-BI:1{BPM:2-Ax:', name='bpm2_pos')
relabel_motors(bpm1_pos)
relabel_motors(bpm2_pos)
