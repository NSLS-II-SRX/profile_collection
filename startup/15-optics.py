from ophyd import EpicsMotor
from ophyd import Device
from ophyd import Component as Cpt

# Slits
class SRXSlits4(MagicSetPseudoPositioner):
    # synthetic axes
    h_cen = Cpt(FixedPseudoSingle)
    h_gap = Cpt(FixedPseudoSingle)
    v_cen = Cpt(FixedPseudoSingle)
    v_gap = Cpt(FixedPseudoSingle)
    
    # real motors
    top = Cpt(EpicsMotor, 'T}Mtr')
    bot = Cpt(EpicsMotor, 'B}Mtr')    
    inb = Cpt(EpicsMotor, 'I}Mtr')
    out = Cpt(EpicsMotor, 'O}Mtr')

    # zero positions
    top_zero = Cpt(PermissiveGetSignal, None, add_prefix=(), value=None)
    bot_zero = Cpt(PermissiveGetSignal, None, add_prefix=(), value=None)
    inb_zero = Cpt(PermissiveGetSignal, None, add_prefix=(), value=None)
    out_zero = Cpt(PermissiveGetSignal, None, add_prefix=(), value=None)

    def forward(self, p_pos):
        h_cen, h_gap, v_cen, v_gap = p_pos
        
        zeros_pos = [getattr(self, k).get() for k in ['top_zero', 'bot_zero',
                                                      'inb_zero', 'out_zero']]
        if any([p is None for p in zeros_pos]):
            raise RuntimeError("You must configure the zero positions")
        top_zero, bot_zero, inb_zero, out_zero = zeros_pos

        top = (v_cen + top_zero) + (v_gap / 2)
        bot = (-v_cen + bot_zero) + (v_gap / 2)
        
        inb = (-h_cen + inb_zero) + (h_gap / 2)
        out = (h_cen + out_zero) + (h_gap / 2)

        return self.RealPosition(top=top, bot=bot, inb=inb, out=out)

    def inverse(self, r_pos):
        top, bot, inb, out = r_pos

        zeros_pos = [getattr(self, k).get() for k in ['top_zero', 'bot_zero',
                                                      'inb_zero', 'out_zero']]
        if any([p is None for p in zeros_pos]):
            raise RuntimeError("You must configure the zero positions")
        top_zero, bot_zero, inb_zero, out_zero = zeros_pos


        # have to flip one sign due to beamline slits coordinate system
        v_cen = ((top - top_zero) - (bot - bot_zero)) / 2
        v_gap = ((top - top_zero) + (bot - bot_zero))

        h_cen = ((out - out_zero) - (inb - inb_zero)) / 2
        h_gap = ((out - out_zero) + (inb - inb_zero))
        
        return self.PseudoPosition(v_cen=v_cen, v_gap=v_gap,
                                   h_cen=h_cen, h_gap=h_gap)

    def set(self, *args):
        v = self.PseudoPosition(*args)
        return super().set(v)


class SRXSlits2(Device):
    inb = Cpt(EpicsMotor, 'I}Mtr')
    out = Cpt(EpicsMotor, 'O}Mtr')

# White Beam slits
slt_wb = SRXSlits4('XF:05IDA-OP:1{Slt:1-Ax:', name='slt_wb')
slt_wb.top_zero.put(-5.775)
slt_wb.bot_zero.put(-4.905)
slt_wb.inb_zero.put(-6.705)
slt_wb.out_zero.put(-4.345)


# Pink beam
slt_pb = SRXSlits2('XF:05IDA-OP:1{Slt:2-Ax:', name='slt_pb')
# Secondary source
slt_ssa = SRXSlits4('XF:05IDB-OP:1{Slt:SSA-Ax:', name='slt_ssa')
slt_ssa.top_zero.put(0.2396)
slt_ssa.bot_zero.put(-2.2046)
slt_ssa.inb_zero.put(-0.4895)
slt_ssa.out_zero.put(1.3610)

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
