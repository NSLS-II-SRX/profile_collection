print(f'Loading {__file__}...')

import os
import numpy as np
from ophyd import (EpicsSignal, EpicsSignalRO, EpicsMotor,
                   Device, Signal, PseudoPositioner, PseudoSingle)
from ophyd.utils.epics_pvs import set_and_wait
from ophyd.pseudopos import (pseudo_position_argument, real_position_argument)
from ophyd.positioner import PositionerBase
from ophyd import Component as Cpt

from scipy.interpolate import InterpolatedUnivariateSpline
import functools
import math
from pathlib import Path

'''
For organization, this file will define objects for the machine. This will
include the undulator (and energy axis) and front end slits.
'''


### Signals
ring_current = EpicsSignalRO('SR:C03-BI{DCCT:1}I:Real-I', name='ring_current')


### Setup undulator
class InsertionDevice(Device, PositionerBase):
    gap = Cpt(EpicsMotor, '-Ax:Gap}-Mtr',
              kind='hinted', name='')
    brake = Cpt(EpicsSignal, '}BrakesDisengaged-Sts',
                write_pv='}BrakesDisengaged-SP',
                kind='omitted', add_prefix=('read_pv', 'write_pv', 'suffix'))

    # These are debugging values, not even connected to by default
    elev = Cpt(EpicsSignalRO, '-Ax:Elev}-Mtr.RBV',
               kind='omitted')
    taper = Cpt(EpicsSignalRO, '-Ax:Taper}-Mtr.RBV',
                kind='omitted')
    tilt = Cpt(EpicsSignalRO, '-Ax:Tilt}-Mtr.RBV',
               kind='omitted')
    elev_u = Cpt(EpicsSignalRO, '-Ax:E}-Mtr.RBV',
                 kind='omitted')

    def set(self, *args, **kwargs):
        set_and_wait(self.brake, 1)
        return self.gap.set(*args, **kwargs)

    def stop(self, *, success=False):
        return self.gap.stop(success=success)

    @property
    def settle_time(self):
        return self.gap.settle_time

    @settle_time.setter
    def settle_time(self, val):
        self.gap.settle_time = val

    @property
    def timeout(self):
        return self.gap.timeout

    @timeout.setter
    def timeout(self, val):
        self.gap.timeout = val

    @property
    def egu(self):
        return self.gap.egu

    @property
    def limits(self):
        return self.gap.limits

    @property
    def low_limit(self):
        return self.gap.low_limit

    @property
    def high_limit(self):
        return self.gap.high_limit

    def move(self, *args, moved_cb=None, **kwargs):
        if moved_cb is not None:
            @functools.wraps(moved_cb)
            def inner(obj=None):
                if obj is not None:
                    obj = self
                return moved_cb(obj=obj)
        else:
            inner = None
        return self.set(*args, moved_cb=inner, **kwargs)

    @property
    def position(self):
        return self.gap.position

    @property
    def moving(self):
        return self.gap.moving

    def subscribe(self, callback, *args, **kwargs):
        @functools.wraps(callback)
        def inner(obj, **kwargs):
            return callback(obj=self, **kwargs)

        return self.gap.subscribe(inner, *args, **kwargs)



### Setup front end slits (primary slits)
class SRXSlitsFE(Device):
    top = Cpt(EpicsMotor, '3-Ax:T}Mtr')
    bot = Cpt(EpicsMotor, '4-Ax:B}Mtr')
    inb = Cpt(EpicsMotor, '3-Ax:I}Mtr')
    out = Cpt(EpicsMotor, '4-Ax:O}Mtr')


fe = SRXSlitsFE('FE:C05A-OP{Slt:', name='fe')
