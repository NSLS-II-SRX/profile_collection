from ophyd.controls import PVPositioner

# Undulator

epu1_gap = PVPositioner('XF:23ID-ID{EPU:1-Ax:Gap}Pos-SP',
                        readback='XF:23ID-ID{EPU:1-Ax:Gap}Pos-I',
                        stop='SR:C23-ID:G1A{EPU:1-Ax:Gap}-Mtr.STOP',
                        stop_val=1,
                        put_complete=True,
                        name='epu2_gap')

epu2_gap = PVPositioner('XF:23ID-ID{EPU:2-Ax:Gap}Pos-SP',
                        readback='XF:23ID-ID{EPU:2-Ax:Gap}Pos-I',
                        stop='SR:C23-ID:G1A{EPU:2-Ax:Gap}-Mtr.STOP',
                        stop_val=1,
                        put_complete=True,
                        name='epu2_gap')

# Front End Slits (Primary Slits)

fe_xc = PVPositioner('FE:C23A-OP{Slt:12-Ax:X}center',
                     readback='FE:C23A-OP{Slt:12-Ax:X}t2.D',
                     stop='FE:C23A-CT{MC:1}allstop',
                     stop_val=1, put_complete=True,
                     name='fe_xc')

fe_yc = PVPositioner('FE:C23A-OP{Slt:12-Ax:Y}center',
                     readback='FE:C23A-OP{Slt:12-Ax:Y}t2.D',
                     stop='FE:C23A-CT{MC:1}allstop',
                     stop_val=1,
                     put_complete=True,
                     name='fe_yc')

fe_xg = PVPositioner('FE:C23A-OP{Slt:12-Ax:X}size',
                     readback='FE:C23A-OP{Slt:12-Ax:X}t2.C',
                     stop='FE:C23A-CT{MC:1}allstop',
                     stop_val=1, put_complete=True,
                     name='fe_xg')

fe_yg = PVPositioner('FE:C23A-OP{Slt:12-Ax:Y}size',
                     readback='FE:C23A-OP{Slt:12-Ax:Y}t2.C',
                     stop='FE:C23A-CT{MC:1}allstop',
                     stop_val=1,
                     put_complete=True,
                     name='fe_yg')
