print(f'Loading {__file__}...')

from ophyd import EpicsMotor, EpicsSignal, EpicsSignalRO
from ophyd import Device
from ophyd import Component as Cpt
from ophyd import PVPositionerPC
from numpy import int16
from nslsii.devices import TwoButtonShutter


### Setup photon shutters
shut_fe = TwoButtonShutter('XF:05ID-PPS{Sh:WB}', name='shut_fe')
shut_a = TwoButtonShutter('XF:05IDA-PPS:1{PSh:2}', name='shut_a')
shut_b = TwoButtonShutter('XF:05IDB-PPS:1{PSh:4}', name='shut_b')


### Setup white/pink beam slits
class SRXSlitsWB(Device):
    # Real synthetic axes
    h_cen = Cpt(EpicsMotor, 'XCtr}Mtr')
    h_gap = Cpt(EpicsMotor, 'XGap}Mtr')
    v_cen = Cpt(EpicsMotor, 'YCtr}Mtr')
    v_gap = Cpt(EpicsMotor, 'YGap}Mtr')

    # Real motors
    top = Cpt(EpicsMotor, 'T}Mtr')
    bot = Cpt(EpicsMotor, 'B}Mtr')
    inb = Cpt(EpicsMotor, 'I}Mtr')
    out = Cpt(EpicsMotor, 'O}Mtr')

class SRXSlitsPB(Device):
    # Real synthetic axes
    h_cen = Cpt(EpicsMotor, 'XCtr}Mtr')
    h_gap = Cpt(EpicsMotor, 'XGap}Mtr')

    # Real motors
    inb = Cpt(EpicsMotor, 'I}Mtr')
    out = Cpt(EpicsMotor, 'O}Mtr')

slt_wb = SRXSlitsWB('XF:05IDA-OP:1{Slt:1-Ax:', name='slt_wb')
slt_pb = SRXSlitsPB('XF:05IDA-OP:1{Slt:2-Ax:', name='slt_pb')

# Is this used?
class SRXSlits2(Device):
    inb = Cpt(EpicsMotor, 'I}Mtr')
    out = Cpt(EpicsMotor, 'O}Mtr')


### Setup HFM Mirror
class SRXHFM(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')
    pitch = Cpt(EpicsMotor, 'P}Mtr')
    bend = Cpt(EpicsMotor, 'Bend}Mtr')

hfm = SRXHFM('XF:05IDA-OP:1{Mir:1-Ax:', name='hfm')


### Setup HDCM
class HDCMPIEZOROLL(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, '')
    readback = Cpt(EpicsSignalRO, '')
    pid_enabled = EpicsSignal('XF:05IDD-CT{FbPid:01}PID:on')
    pid_I = EpicsSignal('XF:05IDD-CT{FbPid:01}PID.I')

    def reset_pid(self):
        yield from bps.mov(self.pid_I, 0.0)

class HDCMPIEZOPITCH(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, '')
    readback = Cpt(EpicsSignalRO, '')
    pid_enabled = EpicsSignal('XF:05IDD-CT{FbPid:02}PID:on')
    pid_I = EpicsSignal('XF:05IDD-CT{FbPid:02}PID.I')

    def reset_pid(self):
        yield from bps.mov(self.pid_I, 0.0)

class SRXDCM(Device):
    bragg = Cpt(EpicsMotor, 'P}Mtr')
    c1_roll = Cpt(EpicsMotor, 'R1}Mtr')
    c1_fine = HDCMPIEZOROLL('XF:05ID-BI{EM:BPM1}DAC0', name='c1_roll')
    c2_x = Cpt(EpicsMotor, 'X2}Mtr')
    c2_pitch = Cpt(EpicsMotor, 'P2}Mtr')
    c2_fine = HDCMPIEZOPITCH('XF:05ID-BI{EM:BPM1}DAC1', name='c2_fine')
    c2_pitch_kill = Cpt(EpicsSignal, 'P2}Cmd:Kill-Cmd')
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')

    temp_pitch = Cpt(EpicsSignalRO, 'P}T-I')

dcm = SRXDCM('XF:05IDA-OP:1{Mono:HDCM-Ax:' , name='dcm')


### Setup BPM motors
class SRXBPM(Device):
    y = Cpt(EpicsMotor, 'YFoil}Mtr')
    diode_x = Cpt(EpicsMotor, 'XDiode}Mtr')
    diode_y = Cpt(EpicsMotor, 'YDiode}Mtr')

bpm1_pos = SRXBPM('XF:05IDA-BI:1{BPM:1-Ax:', name='bpm1_pos')
bpm2_pos = SRXBPM('XF:05IDB-BI:1{BPM:2-Ax:', name='bpm2_pos')


### Setup SSA
class SRXSSAHG(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, 'X}size')
    readback = Cpt(EpicsSignalRO, 'X}t2.C')

class SRXSSAHC(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, 'X}center')
    readback = Cpt(EpicsSignalRO, 'X}t2.D')

class SRXSSAVG(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, 'Y}size')
    readback = Cpt(EpicsSignalRO, 'Y}t2.C')

class SRXSSAVC(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, 'Y}center')
    readback = Cpt(EpicsSignalRO, 'Y}t2.D')

class SRXSSACalc(Device):
    h_cen = SRXSSAHC('XF:05IDB-OP:1{Slt:SSA-Ax:')
    h_gap = SRXSSAHG('XF:05IDB-OP:1{Slt:SSA-Ax:')
    v_cen = SRXSSAVC('XF:05IDB-OP:1{Slt:SSA-Ax:')
    v_gap = SRXSSAVG('XF:05IDB-OP:1{Slt:SSA-Ax:')

slt_ssa = SRXSSACalc('XF:05IDB-OP:1{Slt:SSA-Ax:',name='slt_ssa')


### Setup fast shutter
# This is not currently installed at SRX and is commented out
# class SRXSOFTINP(Device):
#     pulse = Cpt(EpicsSignal,'')
#     #these need to be put complete!!
#     def high_cmd(self):
#         self.pulse.put(1)
#     def low_cmd(self):
#         self.pulse.put(0)
#     def toggle_cmd(self):
#         if self.pulse.get() == 0:
#             self.pulse.put(1)
#         else:
#             self.pulse.put(0)
# shut_fast = SRXSOFTINP('XF:05IDD-ES:1{Sclr:1}UserLED',name='shut_fast')
# 
# class SRXFASTSHUT(SRXSOFTINP):
#     pulse = Cpt(EpicsSignal,':SOFT_IN:B1')
#     iobit = Cpt(EpicsSignalRO,':SYS_STAT1LO')
#     def status(self):
#         self.low_cmd()
#         shutopen = bool(int16(self.iobit.get()) & int16(2))
#         if shutopen is True:
#             return 'Open'
#         else:
#             return 'Closed'
#     def high_cmd(self):
#         self.pulse.put(1)
#     def low_cmd(self):
#         self.pulse.put(0)
#     def open_cmd(self):
#         print(self.status())
#         if self.status() is 'Closed':
#             print(self.status())
#         #    self.low_cmd()
#             self.high_cmd()
#     def close_cmd(self):
#         print(self.status())
#         if self.status() is 'Open':
#             print(self.status())
#          #   self.low_cmd()
#             self.high_cmd()
# 
# #shut_fast = SRXFASTSHUT('XF:05IDD-ES:1{Dev:Zebra1}',name='shut_fast')
