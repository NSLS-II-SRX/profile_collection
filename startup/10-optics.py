from ophyd.controls import EpicsMotor, PVPositioner

# M1A

# args = ('XF:23IDA-OP:1{Mir:1-Ax:Z}Mtr_POS_SP',
#        {'readback': 'XF:23IDA-OP:1{Mir:1-Ax:Z}Mtr_MON',
#         'act': 'XF:23IDA-OP:1{Mir:1}MOVE_CMD.PROC',
#         'act_val': 1,
#         'stop': 'XF:23IDA-OP:1{Mir:1}STOP_CMD.PROC',
#         'stop_val': 1,
#         'done': 'XF:23IDA-OP:1{Mir:1}BUSY_STS',
#         'done_val': 0,
#         'name': 'm1a_z',
#        })
# m1a_z = PVPositioner(args[0], **args[1])

# Slits

slt_wb_tb = EpicsMotor('XF:05IDA-OP:1{Slt:1-Ax:T}Mtr', name='slt_wb_tb')
slt_wb_bb = EpicsMotor('XF:05IDA-OP:1{Slt:1-Ax:B}Mtr', name='slt_wb_bb')
slt_wb_ib = EpicsMotor('XF:05IDA-OP:1{Slt:1-Ax:I}Mtr', name='slt_wb_ib')
slt_wb_ob = EpicsMotor('XF:05IDA-OP:1{Slt:1-Ax:O}Mtr', name='slt_wb_ob')

slt_pb_ib = EpicsMotor('XF:05IDA-OP:1{Slt:2-Ax:I}Mtr', name='slt_pb_ib')
slt_pb_ob = EpicsMotor('XF:05IDA-OP:1{Slt:2-Ax:O}Mtr', name='slt_pb_ob')


# HFM Mirror

hfm_x = EpicsMotor('XF:05IDA-OP:1{Mir:1-Ax:X}Mtr', name='hfm_x')
hfm_y = EpicsMotor('XF:05IDA-OP:1{Mir:1-Ax:Y}Mtr', name='hfm_y')
hfm_pit = EpicsMotor('XF:05IDA-OP:1{Mir:1-Ax:P}Mtr', name='hfm_pit')
hfm_bdr = EpicsMotor('XF:05IDA-OP:1{Mir:1-Ax:Bend}Mtr', name='hfm_bdr')


# HDCM

dcm_bragg = EpicsMotor('XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr', name='dcm_bragg')
dcm_c2_pitch = EpicsMotor('XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Mtr', name='dcm_c2_pitch')
dcm_c1_roll = EpicsMotor('XF:05IDA-OP:1{Mono:HDCM-Ax:R1}Mtr', name='dcm_c1_roll')
dcm_gap = EpicsMotor('XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr', name='c_gap')
dcm_x = EpicsMotor('XF:05IDA-OP:1{Mono:HDCM-Ax:X}Mtr', name='dcm_x')
dcm_y = EpicsMotor('XF:05IDA-OP:1{Mono:HDCM-Ax:Y}Mtr', name='dcm_y')


# BPMs

bpm1_y=EpicsMotor('XF:05IDA-BI:1{BPM:1-Ax:YFoil}Mtr', name='bpm1_y')
bpm1_diode_x=EpicsMotor('XF:05IDA-BI:1{BPM:1-Ax:XDiode}Mtr', name='bpm1_diode_x')
bpm1_diode_y=EpicsMotor('XF:05IDA-BI:1{BPM:1-Ax:YDiode}Mtr', name='bpm1_diode_y')

bpm2_y=EpicsMotor('XF:05IDB-BI:1{BPM:2-Ax:YFoil}Mtr', name='bpm2_y')
bpm2_diode_x=EpicsMotor('XF:05IDB-BI:1{BPM:2-Ax:XDiode}Mtr', name='bpm2_diode_x')
bpm2_diode_y=EpicsMotor('XF:05IDB-BI:1{BPM:2-Ax:YDiode}Mtr', name='bpm2_diode_y')


# Secondary source

slt_ssa_tb = EpicsMotor('XF:05IDB-OP:1{Slt:SSA-Ax:T}Mtr', name='slt_ssa_tb')
slt_ssa_bb = EpicsMotor('XF:05IDB-OP:1{Slt:SSA-Ax:B}Mtr', name='slt_ssa_bb')
slt_ssa_ib = EpicsMotor('XF:05IDB-OP:1{Slt:SSA-Ax:I}Mtr', name='slt_ssa_ib')
slt_ssa_ob = EpicsMotor('XF:05IDB-OP:1{Slt:SSA-Ax:O}Mtr', name='slt_ssa_ob')


# High Flux KB
hffm_vfm_pitch=EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:P}Mtr', name='hffm_vfm_pitch')
hffm_vfm_roll=EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:R}Mtr', name='hffm_vfm_roll')
hffm_vfm_y=EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:Y}Mtr', name='hffm_vfm_y')

