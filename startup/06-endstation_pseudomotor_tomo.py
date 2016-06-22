# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:45:54 2016

@author: xf05id1
"""

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
    real_finex_top = Cpt(EpicsMotor, 'XFT}Mtr',  doc='Attocube ECS3030 x, sampe as tomo_stage.finex_top', settle_time= piezo_jena_settle_time)
    real_finez_top = Cpt(EpicsMotor, 'ZFT}Mtr',  doc='Attocube ECS3030 z, sampe as tomo_stage.finez_top', settle_time= piezo_jena_settle_time) 
    real_theta = Cpt(EpicsMotor, 'Theta}Mtr', doc='rotatry stage theta angle')

    # configuration settings
    theta0 = Cpt(Signal, value=0.0, doc='theta offset')

    def __init__(self, prefix, **kwargs):
        super().__init__(prefix, **kwargs)

        # if theta changes, update the pseudo position
        self.theta0.subscribe(self.parameter_updated)

    def parameter_updated(self, value=None, **kwargs):
        self._update_position()

    @property
    def radian_theta(self):
        return math.radians(self.real_theta.position + self.theta0.get())

    @pseudo_position_argument
    def forward(self, position):
        theta = self.radian_theta
        c = math.cos(theta)
        s = math.sin(theta)

        x = c * position.lab_x + s * position.lab_z
        z = (-1) * (-s * position.lab_x + c * position.lab_z)

        return self.RealPosition(real_finex_top=x, real_finez_top=z, real_theta = math.degrees(theta))

    @real_position_argument
    def inverse(self, position):
        theta = self.radian_theta
        c = math.cos(theta)
        s = math.sin(theta)
        x = c * position.real_finex_top - s * (-1) * position.real_finez_top
        z = s * position.real_finex_top + c * (-1) * position.real_finez_top
        return self.PseudoPosition(lab_x=x, lab_z=z)


tomo_lab = FineSampleLabX('XF:05IDD-ES:1{Stg:Tomo-Ax:', name='tomo_lab')
tomo_lab.read_attrs = ['lab_x', 'lab_z', 'real_finex_top', 'real_finez_top', 'real_theta']
#tomox_lab = zplab.tomox_lab
#tomoz_lab = zplab.tomoz_lab
relabel_motors(tomo_lab)