print(f"Loading {__file__}...")


from ophyd import EpicsMotor, EpicsSignal, EpicsSignalRO
from ophyd import Device
from ophyd import Component as Cpt
from ophyd import PVPositionerPC
from nslsii.devices import TwoButtonShutter

# print('imports done')

# Setup photon shutters
shut_fe = TwoButtonShutter("XF:05ID-PPS{Sh:WB}", name="shut_fe")
shut_a = TwoButtonShutter("XF:05IDA-PPS:1{PSh:2}", name="shut_a")
shut_b = TwoButtonShutter("XF:05IDB-PPS:1{PSh:4}", name="shut_b")

# print('photon shutters done')

# Check if shutters are open
def check_shutters(check, status):
    '''
    Check if the shutters are in the desired position. At the beginning of
    a scan, they will open. At the end of the scan, they will close.

    Inputs:
    check   <bool>      Move the shutters
    status  <string>    'Open' or 'Close' Should the function be openning
                        or closing the shutters

    Returns:
     -

    '''

    if check is False:
        print("\n**************************************************")
        print("WARNING: Shutters are not controlled in this scan.")
        print("**************************************************\n")
    else:
        if status == 'Open':
            if shut_b.status.get() == 'Not Open':
                print('Opening B-hutch shutter..')
                yield from mov(shut_b, "Open")
            print('Opening D-hutch shutter...')
            yield from mov(shut_d, 0)
        else:
            print('Closing D-hutch shutter...')
            yield from mov(shut_d, 1)


# Setup white/pink beam slits
class SRXSlitsWB(Device):
    # Real synthetic axes
    h_cen = Cpt(EpicsMotor, "XCtr}Mtr")
    h_gap = Cpt(EpicsMotor, "XGap}Mtr")
    v_cen = Cpt(EpicsMotor, "YCtr}Mtr")
    v_gap = Cpt(EpicsMotor, "YGap}Mtr")

    # Real motors
    top = Cpt(EpicsMotor, "T}Mtr")
    bot = Cpt(EpicsMotor, "B}Mtr")
    inb = Cpt(EpicsMotor, "I}Mtr")
    out = Cpt(EpicsMotor, "O}Mtr")


class SRXSlitsPB(Device):
    # Real synthetic axes
    h_cen = Cpt(EpicsMotor, "XCtr}Mtr")
    h_gap = Cpt(EpicsMotor, "XGap}Mtr")

    # Real motors
    inb = Cpt(EpicsMotor, "I}Mtr")
    out = Cpt(EpicsMotor, "O}Mtr")


slt_wb = SRXSlitsWB("XF:05IDA-OP:1{Slt:1-Ax:", name="slt_wb")
slt_pb = SRXSlitsPB("XF:05IDA-OP:1{Slt:2-Ax:", name="slt_pb")
# print('slits done')

# Setup HFM Mirror
class SRXHFM(Device):
    x = Cpt(EpicsMotor, "X}Mtr")
    y = Cpt(EpicsMotor, "Y}Mtr")
    pitch = Cpt(EpicsMotor, "P}Mtr")
    bend = Cpt(EpicsMotor, "Bend}Mtr")


hfm = SRXHFM("XF:05IDA-OP:1{Mir:1-Ax:", name="hfm")
# print('mirror done')

# Setup HDCM
class HDCMPiezoRoll(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "")
    readback = Cpt(EpicsSignalRO, "")
    pid_enabled = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:01}PID:on",
        name="pid_enabled",
        add_prefix=()
    )
    pid_I = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:01}PID.I",
        name="pid_I",
        add_prefix=()
    )

    def reset_pid(self):
        yield from bps.mov(self.pid_I, 0.0)


class HDCMPiezoPitch(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "")
    readback = Cpt(EpicsSignalRO, "")
    pid_enabled = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:02}PID:on",
        name="pid_enabled",
        add_prefix=()
    )
    pid_I = Cpt(
        EpicsSignal,
        "XF:05IDD-CT{FbPid:02}PID.I",
        name="pid_I",
        add_prefix=()
    )

    def reset_pid(self):
        yield from bps.mov(self.pid_I, 0.0)


class SRXDCM(Device):
    bragg = energy.bragg
    c1_roll = Cpt(EpicsMotor, "R1}Mtr")
    c1_fine = Cpt(
        HDCMPiezoRoll,
        "XF:05ID-BI{EM:BPM1}DAC0", name="c1_fine",
        add_prefix=()
    )
    c2_x = energy.c2_x
    c2_pitch = Cpt(EpicsMotor, "P2}Mtr")
    c2_fine = Cpt(
        HDCMPiezoPitch,
        "XF:05ID-BI{EM:BPM1}DAC1",
        name="c2_fine",
        add_prefix=()
    )
    c2_pitch_kill = Cpt(EpicsSignal, "P2}Cmd:Kill-Cmd")
    x = Cpt(EpicsMotor, "X}Mtr")
    y = Cpt(EpicsMotor, "Y}Mtr")

    temp_pitch = Cpt(EpicsSignalRO, "P}T-I")


dcm = SRXDCM("XF:05IDA-OP:1{Mono:HDCM-Ax:", name="dcm")
# print('dcm done')

# Setup BPM motors
class SRXBPM(Device):
    y = Cpt(EpicsMotor, "YFoil}Mtr")
    diode_x = Cpt(EpicsMotor, "XDiode}Mtr")
    diode_y = Cpt(EpicsMotor, "YDiode}Mtr")


bpm1_pos = SRXBPM("XF:05IDA-BI:1{BPM:1-Ax:", name="bpm1_pos")
bpm2_pos = SRXBPM("XF:05IDB-BI:1{BPM:2-Ax:", name="bpm2_pos")
# print('bpm done')

# Setup SSA
class SRXSSAHG(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "X}size")
    readback = Cpt(EpicsSignalRO, "X}t2.C")


class SRXSSAHC(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "X}center")
    readback = Cpt(EpicsSignalRO, "X}t2.D")


class SRXSSAVG(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "Y}size")
    readback = Cpt(EpicsSignalRO, "Y}t2.C")


class SRXSSAVC(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, "Y}center")
    readback = Cpt(EpicsSignalRO, "Y}t2.D")


class SRXSSACalc(Device):
    h_cen = Cpt(SRXSSAHC, "", name="h_cen")
    h_gap = Cpt(SRXSSAHG, "", name="h_gap")
    v_cen = Cpt(SRXSSAVC, "", name="v_cen")
    v_gap = Cpt(SRXSSAVG, "", name="v_gap")


slt_ssa = SRXSSACalc("XF:05IDB-OP:1{Slt:SSA-Ax:", name="slt_ssa")
# print('ssa done')

# Setup fast shutter
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
#         shutopen = bool(np.int16(self.iobit.get()) & np.int16(2))
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
