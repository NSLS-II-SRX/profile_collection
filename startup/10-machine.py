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

'''
For organization, this file will define objects for the machine. This will
include the undulator (and energy axis) and front end slits.
'''


### Constants
ANG_OVER_EV = 12.3984


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


def retune_undulator():
    energy.detune.put(0.)
    energy.move(energy.energy.get()[0])


### Setup energy axis
class Energy(PseudoPositioner):
    # Synthetic axis
    energy = Cpt(PseudoSingle)
    # Real motors
    u_gap = Cpt(InsertionDevice, 'SR:C5-ID:G1{IVU21:1')
    bragg = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr', add_prefix=(),
                read_attrs=['user_readback'])
    c2_x = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr', add_prefix=(),
               read_attrs=['user_readback'])
    epics_d_spacing = EpicsSignal('XF:05IDA-CT{IOC:Status01}DCMDspacing.VAL')
    epics_bragg_offset = EpicsSignal('XF:05IDA-CT{IOC:Status01}BraggOffset.VAL')

    # Motor enable flags
    move_u_gap = Cpt(Signal, None, add_prefix=(), value=True)
    move_c2_x = Cpt(Signal, None, add_prefix=(), value=True)
    harmonic = Cpt(Signal, None, add_prefix=(), value=0, kind='config')
    selected_harmonic = Cpt(Signal, None, add_prefix=(), value=0)

    # Experimental
    detune = Cpt(Signal, None, add_prefix=(), value=0)

    def energy_to_positions(self, target_energy, undulator_harmonic, u_detune):
        """Compute undulator and mono positions given a target energy

        Paramaters
        ----------
        target_energy : float
            Target energy in keV

        undulator_harmonic : int, optional
            The harmonic in the undulator to use

        uv_mistune : float, optional
            Amount to 'mistune' the undulator in keV.  Will settings
            such that the peak of the undulator spectrum will be at
            `target_energy + uv_mistune`.

        Returns
        -------
        bragg : float
             The angle to set the monocromotor

        """
        # Set up constants
        Xoffset = self._xoffset
        d_111 = self._d_111
        delta_bragg = self._delta_bragg
        C2Xcal = self._c2xcal
        T2cal = self._t2cal
        etoulookup = self.etoulookup

        # Calculate Bragg RBV
        BraggRBV = (
            np.arcsin((ANG_OVER_EV / target_energy) / (2 * d_111)) /
            np.pi * 180 -
            delta_bragg)

        # Calculate C2X
        Bragg = BraggRBV + delta_bragg
        T2 = (Xoffset *
              np.sin(Bragg * np.pi / 180) /
              np.sin(2 * Bragg * np.pi / 180))
        dT2 = T2 - T2cal
        C2X = C2Xcal - dT2

        # Calculate undulator gap

        #  TODO make this more sohpisticated to stay a fixed distance
        #  off the peak of the undulator energy
        ugap = float(
            etoulookup((target_energy + u_detune) /
                       undulator_harmonic))  # in mm
        ugap *= 1000  # convert to um

        return BraggRBV, C2X, ugap

    def undulator_energy(self, harmonic=3):
        """Return the current energy peak of the undulator at the given harmonic

        Paramaters
        ----------
        harmonic : int, optional
            The harmonic to use, defaults to 3
        """
        p = self.u_gap.get().readback
        utoelookup = self.utoelookup

        fundemental = float(utoelookup(ugap))

        energy = fundemental * harmonic

        return energy

    def __init__(self, *args,
                 xoffset=None, d_111=None, delta_bragg=None, C2Xcal=None, T2cal=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._xoffset = xoffset
        self._d_111 = d_111
        self._delta_bragg = delta_bragg
        self._c2xcal = C2Xcal
        self._t2cal = T2cal

        calib_path = '/nfs/xf05id1/UndulatorCalibration/'
        calib_file = 'SRXUgapCalibration20170612.txt'

        with open(os.path.join(calib_path, calib_file), 'r') as f:
            next(f)
            uposlistIn=[]
            elistIn=[]
            for line in f:
                num = [float(x) for x in line.split()]
                uposlistIn.append(num[0])
                elistIn.append(num[1])

        self.etoulookup = InterpolatedUnivariateSpline(elistIn, uposlistIn)
        self.utoelookup = InterpolatedUnivariateSpline(uposlistIn, elistIn)

        self.u_gap.gap.user_readback.name = self.u_gap.name


    def crystal_gap(self):
        """
        Return the current physical gap between first and second crystals
        """
        C2X = self.c2_x.get().user_readback
        bragg = self.bragg.get().user_readback

        T2cal = self._t2cal
        delta_bragg = self._delta_bragg
        d_111 = self._d_111
        c2x_cal = self._c2xcal

        Bragg = np.pi/180 * (bragg + delta_bragg)

        dT2 = c2x_cal - C2X
        T2 = dT2 + T2cal

        XoffsetVal = T2/(np.sin(Bragg)/np.sin(2*Bragg))

        return XoffsetVal

    @pseudo_position_argument
    def forward(self, p_pos):
        energy = p_pos.energy
        harmonic = int(self.harmonic.get())
        if harmonic < 0 or ((harmonic % 2) == 0 and harmonic != 0):
            raise RuntimeError(f"The harmonic must be 0 or odd and positive, you set {harmonic}.  "
                               "Set `energy.harmonic` to a positive odd integer or 0.")
        detune = self.detune.get()
        if energy <= 4.4:
            raise ValueError("The energy you entered is too low ({} keV). "
                             "Minimum energy = 4.4 keV".format(energy))
        if (energy > 25.):
            if (energy < 4400.) or (energy > 25000.):
            # Energy is invalid
                raise ValueError('The requested photon energy is invalid ({} keV). '
                             'Values must be in the range of 4.4 - 25 keV'.format(energy))
            else:
            # Energy is in eV
                energy = energy / 1000.

        # harmonic cannot be None, it is an undesired datatype
        # Previously, we were finding the harmonic with the highest flux, this
        # was always done during energy change since harmonic was returned to
        # None
        # Here, we are programming it in
        # if harmonic is None:
        if (harmonic < 3):
            harmonic = 3
            # Choose the right harmonic
            braggcal, c2xcal, ugapcal = self.energy_to_positions(energy, harmonic, detune)
            # Try higher harmonics until the required gap is too small
            while True:
                braggcal, c2xcal, ugapcal = self.energy_to_positions(energy, harmonic + 2, detune)
                if ugapcal < self.u_gap.low_limit:
                    break
                harmonic += 2
        self.selected_harmonic.put(harmonic)
        # Compute where we would move everything to in a perfect world
        bragg, c2_x, u_gap = self.energy_to_positions(energy, harmonic, detune)

        # Sometimes move the crystal gap
        if not self.move_c2_x.get():
            c2_x = self.c2_x.position

        # Sometimes move the undulator
        if not self.move_u_gap.get():
            u_gap = self.u_gap.position

        return self.RealPosition(bragg=bragg, c2_x=c2_x, u_gap=u_gap)

    @real_position_argument
    def inverse(self, r_pos):
        bragg = r_pos.bragg
        e = ANG_OVER_EV / (2 * self._d_111 * math.sin(math.radians(bragg + self._delta_bragg)))
        return self.PseudoPosition(energy=float(e))

    @pseudo_position_argument
    def set(self, position):
        return super().set([float(_) for _ in position])

    def synch_with_epics(self):
        self.epics_d_spacing.put(self._d_111)
        self.epics_bragg_offset.put(self._delta_bragg)


# Let's try to only keep one "old" value in for ease of use
# Calibrated 2019-08-16
# Use scans 29496 - 29503
# cal_data_2019cycle3 = {
#  'd_111': 3.1287603762011367,
#  'delta_bragg': 0.21004620069236488,
#  'C2Xcal': 3.6,
#  'T2cal': 15.0347755916,
#  'xoffset': 24.770
# }

# Calibrated 2019-09-12
# Use scans 30612 - 30617
cal_data_2019cycle3 = {
 'd_111': 3.1294298470798565,
 'delta_bragg': 0.20569524708214598,
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.770
}

energy = Energy(prefix='', name='energy', **cal_data_2019cycle3)
energy.synch_with_epics()
energy.value = 1.0


### Setup front end slits (primary slits)
class SRXSlitsFE(Device):
    top = Cpt(EpicsMotor, '3-Ax:T}Mtr')
    bot = Cpt(EpicsMotor, '4-Ax:B}Mtr')
    inb = Cpt(EpicsMotor, '3-Ax:I}Mtr')
    out = Cpt(EpicsMotor, '4-Ax:O}Mtr')

fe = SRXSlitsFE('FE:C05A-OP{Slt:', name='fe')

