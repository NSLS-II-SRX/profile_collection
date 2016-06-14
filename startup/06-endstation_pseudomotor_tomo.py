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
    zpssx_lab = Cpt(PseudoSingle)
    zpssz_lab = Cpt(PseudoSingle)

    # real axes
    #zpssx = Cpt(EpicsMotor, '{Ppmac:1-zpssx}Mtr')
    #zpssz = Cpt(EpicsMotor, '{Ppmac:1-zpssz}Mtr')
    #zpsth = Cpt(EpicsMotor, '{SC210:1-Ax:1}Mtr', doc='theta angle')

    zpssx = Cpt(EpicsMotor, 'XFT}Mtr')
    zpssz = Cpt(EpicsMotor, 'ZFT}Mtr') 
    zpsth = Cpt(EpicsMotor, 'Theta}Mtr', doc='theta angle')

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
        return math.radians(self.zpsth.position + self.theta0.get())

    @pseudo_position_argument
    def forward(self, position):
        theta = self.radian_theta
        c = math.cos(theta)
        s = math.sin(theta)

        x = c * position.zpssx_lab + s * position.zpssz_lab
        z = (-1) * (-s * position.zpssx_lab + c * position.zpssz_lab)

#        return self.RealPosition(zpssx=x, zpssz=z)
        return self.RealPosition(zpssx=x, zpssz=z, zpsth = math.degrees(theta))

    @real_position_argument
    def inverse(self, position):
        theta = self.radian_theta
        c = math.cos(theta)
        s = math.sin(theta)
        x = c * position.zpssx - s * (-1) * position.zpssz
        z = s * position.zpssx + c * (-1) * position.zpssz
        return self.PseudoPosition(zpssx_lab=x, zpssz_lab=z)


tomo_lab = FineSampleLabX('XF:05IDD-ES:1{Stg:Tomo-Ax:', name='tomo_lab')
#zplab = FineSampleLabX('XF:03IDC-ES', name='zplab')
#zpssx_lab = zplab.zpssx_lab
#zpssz_lab = zplab.zpssz_lab
relabel_motors(tomo_lab)