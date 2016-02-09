from ophyd import EpicsMotor

#motors for xrf, tomo, etc. go here

#High Flux KB mirrors

##pseudo motors (should these be in the 'endstation' or in the 'pseudomotor' configuration?)
m2_pitch = EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:P}Mtr', name='m2_pitch')
m2_roll = EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:R}Mtr', name='m2_roll')
m2_vertical = EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:Y}Mtr', name='m2_vertical')
#m2_x, m3_x, m3_pitch will need to be added here

##real motors
m2_xu = EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:XU}Mtr', name='m2_xu')
m2_xd = EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:XD}Mtr', name='m2_xd')
m2_z = EpicsMotor('XF:05IDD-OP:1{Mir:2-Ax:Z}Mtr', name='m2_z')
#m2_yui, m2_ydo, m2_ydi, m2_ydo are not included here yet; they should no be scanned independently

m3_xu = EpicsMotor('XF:05IDD-OP:1{Mir:3-Ax:XU}Mtr', name='m3_xu')
m3_xd = EpicsMotor('XF:05IDD-OP:1{Mir:3-Ax:XD}Mtr', name='m3_xd')
m3_y = EpicsMotor('XF:05IDD-OP:1{Mir:3-Ax:Y}Mtr', name='m3_y')

#High flux sample stages
#Aero_x = EpicsMotor('XF:05IDD-ES:1{Stg:XY-Ax:X}Mtr', name='Aero_x')
Aero_x = EpicsMotor('XF:05IDD-ES:1{Stg:Smpl1-Ax:X}Mtr', name='Aero_x')
Aero_y = EpicsMotor('XF:05IDD-ES:1{Stg:Smpl1-Ax:Y}Mtr', name='Aero_y')
#PI = EpicsMotor('XF:05IDD-OP:1{Stg:Smpl1-Ax:Z}Mtr', name='PI')
PI = EpicsMotor('XF:05IDD-ES:1{Stg:Smpl1-Ax:Z}Mtr', name='PI')

#SDD
Vortex_x = EpicsMotor('XF:05IDD-ES:1{Det:1-Ax:X}Mtr', name='Vortex_x')
Vortex_z = EpicsMotor('XF:05IDD-ES:1{Det:1-Ax:Z}Mtr', name='Vortex_z')
