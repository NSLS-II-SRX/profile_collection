from ophyd.controls import PVPositioner

# Undulator

ivu1_gap = PVPositioner('SR:C5-ID:G1{IVU21:1-Mtr:2}Inp:Pos',
                        readback='SR:C5-ID:G1{IVU21:1-LEnc}Gap',
                        stop='SR:C5-ID:G1{IVU21:1-Mtrc}Sw:Stp',
                        stop_val=1,
                        put_complete=True,
                        name='ivu1_gap')

# Front End Slits (Primary Slits)

fe_tb = PVPositioner('FE:C05A-OP{Slt:3-Ax:T}Mtr.VAL',
                     readback='FE:C05A-OP{Slt:3-Ax:T}Mtr.RBV',
                     stop='FE:C05A-OP{Slt:3-Ax:T}Mtr.STOP',
                     stop_val=1, put_complete=True,
                     name='fe_tb')

fe_bb = PVPositioner('FE:C05A-OP{Slt:4-Ax:B}Mtr.VAL',
                     readback='FE:C05A-OP{Slt:4-Ax:B}Mtr.RBV',
                     stop='FE:C05A-OP{Slt:4-Ax:B}Mtr.STOP',
                     stop_val=1, put_complete=True,
                     name='fe_bb')

fe_ib = PVPositioner('FE:C05A-OP{Slt:3-Ax:I}Mtr.VAL',
                     readback='FE:C05A-OP{Slt:3-Ax:I}Mtr.RBV',
                     stop='FE:C05A-OP{Slt:3-Ax:I}Mtr.STOP',
                     stop_val=1, put_complete=True,
                     name='fe_ib')

fe_ob = PVPositioner('FE:C05A-OP{Slt:4-Ax:O}Mtr.VAL',
                     readback='FE:C05A-OP{Slt:4-Ax:O}Mtr.RBV',
                     stop='FE:C05A-OP{Slt:4-Ax:O}Mtr.STOP',
                     stop_val=1, put_complete=True,
                     name='fe_ob')

