from ophyd.controls import PVPositioner, EpicsMotor

# Diffo angles

delta = EpicsMotor('XF:23ID1-ES{Dif-Ax:Del}Mtr', name='delta')
gamma = EpicsMotor('XF:23ID1-ES{Dif-Ax:Gam}Mtr', name='gamma')
theta = EpicsMotor('XF:23ID1-ES{Dif-Ax:Th}Mtr', name='theta')

# Sample positions

sx = EpicsMotor('XF:23ID1-ES{Dif-Ax:X}Mtr', name='sx')
sy = PVPositioner('XF:23ID1-ES{Dif-Ax:SY}Pos-SP',
                  readback='XF:23ID1-ES{Dif-Ax:SY}Pos-RB',
                  stop='XF:23ID1-ES{Dif-Cryo}Cmd:Stop-Cmd',
                  stop_val=1, put_complete=True,
                  name='sy')

sz = PVPositioner('XF:23ID1-ES{Dif-Ax:SZ}Pos-SP',
                  readback='XF:23ID1-ES{Dif-Ax:SZ}Pos-SP',
                  stop='XF:23ID1-ES{Dif-Cryo}Cmd:Stop-Cmd',
                  stop_val=1, put_complete=True,
                  name='sz')

cryoangle = PVPositioner('XF:23ID1-ES{Dif-Cryo}Pos:Angle-SP',
                         readback='XF:23ID1-ES{Dif-Cryo}Pos:Angle-RB',
                         name='cryoangle')

# Nano-positioners

nptx = EpicsMotor('XF:23ID1-ES{Dif:Lens-Ax:TopX}Mtr', name='nptx')
npty = EpicsMotor('XF:23ID1-ES{Dif:Lens-Ax:TopY}Mtr', name='npty')
nptz = EpicsMotor('XF:23ID1-ES{Dif:Lens-Ax:TopZ}Mtr', name='nptz')
npbx = EpicsMotor('XF:23ID1-ES{Dif:Lens-Ax:BtmX}Mtr', name='npbx')
npby = EpicsMotor('XF:23ID1-ES{Dif:Lens-Ax:BtmY}Mtr', name='npby')
npbz = EpicsMotor('XF:23ID1-ES{Dif:Lens-Ax:BtmZ}Mtr', name='npbz')

# Lakeshore 336 Temp Controller

temp_sp = PVPositioner('XF:23ID1-ES{TCtrl:1-Out:1}T-SP',
                       readback='XF:23ID1-ES{TCtrl:1-Out:1}T-RB',
                       done='XF:23ID1-ES{TCtrl:1-Out:1}Sts:Ramp-Sts',
                       done_val=0, name='temp_sp')
