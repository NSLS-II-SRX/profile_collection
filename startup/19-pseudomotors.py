print(f'Loading {__file__}...')

import math
from ophyd import (Device, PVPositionerPC, EpicsMotor, Signal, EpicsSignal,
                   EpicsSignalRO, Component as Cpt, FormattedComponent as FCpt,
                   PseudoSingle, PseudoPositioner,
                   )
from ophyd.pseudopos import (real_position_argument,
                             pseudo_position_argument)
from hxntools.device import NamedDevice #qustion for Ken

class FineSampleLabX(PseudoPositioner, NamedDevice):
    '''Pseudo positioner definition for zoneplate fine sample positioner
    with angular correction
    (-1): is due to the reverse z direction
    '''
    # pseudo axes
    lab_x = Cpt(PseudoSingle)
    lab_z = Cpt(PseudoSingle)

    # real axes
    real_topx = Cpt(EpicsMotor, 'XF}Mtr', name='hf_stage_topx')
    real_topz = Cpt(EpicsMotor, 'ZF}Mtr', name='hf_stage_topz')
    real_theta = Cpt(EpicsMotor, 'Rot}Mtr', name='hf_stage_th')

    # configuration settings
    _theta0 = Cpt(Signal, value=-45.0, doc='theta offset')

    def __init__(self, prefix, **kwargs):
        super().__init__(prefix, **kwargs)

        # if theta changes, update the pseudo position
        self._theta0.subscribe(self.parameter_updated)

    def parameter_updated(self, value=None, **kwargs):
        self._update_position()

    @property
    def radian_theta(self):
        return math.radians(self.real_theta.position + self._theta0.get())

    @pseudo_position_argument
    def forward(self, position):
        theta = self.radian_theta
        c = math.cos(theta)
        s = math.sin(theta)

        x = c * position.lab_x + (-1) * s * position.lab_z
        z = 1 * s * position.lab_x + 1 * c * position.lab_z

        return self.RealPosition(real_topx=x, real_topz=z, real_theta = (math.degrees(theta)-self._theta0.get()))

    @real_position_argument
    def inverse(self, position):
        theta = self.radian_theta
        c = math.cos(theta)
        s = math.sin(theta)
        x = c * position.real_topx + s * (1) * position.real_topz
        z = (-1) * s * position.real_topx + c * (1) * position.real_topz
        return self.PseudoPosition(lab_x=x, lab_z=z)


lab_stage = FineSampleLabX('XF:05IDD-ES:1{Smpl:1-Ax:', name='lab_stage')
lab_stage.read_attrs = ['lab_x', 'lab_z', 'real_topx', 'real_topz', 'real_theta']

