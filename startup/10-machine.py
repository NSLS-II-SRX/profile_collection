import os
import numpy as np
from ophyd import (PVPositioner, EpicsSignal, EpicsSignalRO, EpicsMotor,
                   Device, Signal, PseudoPositioner, PseudoSingle)
from ophyd.utils.epics_pvs import set_and_wait
from ophyd.ophydobj import MoveStatus
from ophyd.pseudopos import (pseudo_position_argument, real_position_argument)
from ophyd.positioner import PositionerBase
from ophyd import Component as Cpt
from nslsii.devices import TwoButtonShutter
from pathlib import Path

from scipy.interpolate import InterpolatedUnivariateSpline
import functools
import math
import uuid


ring_current = EpicsSignalRO('SR:C03-BI{DCCT:1}I:Real-I', name='ring_current')
cryo_v19 = EpicsSignal('XF:05IDA-UT{Cryo:1-IV:19}Sts-Sts', name='cryo_v19')


_undulator_kwargs = dict(name='ivu1_gap', read_attrs=['readback'],
                         calib_path='/nfs/xf05id1/UndulatorCalibration/',
                         # calib_file='SRXUgapCalibration20150411_final.text',
                         # calib_file='SRXUgapCalibration20160608_final.text',
                         # calib_file='SRXUgapCalibration20170131.txt',
                         calib_file='SRXUgapCalibration20170612.txt',
                         configuration_attrs=['corrfunc_sta', 'pos', 'girder',
                                              'real_pos', 'elevation'])


ANG_OVER_EV = 12.3984


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


class Energy(PseudoPositioner):
    # synthetic axis
    energy = Cpt(PseudoSingle)
    # real motors
    u_gap = Cpt(InsertionDevice, 'SR:C5-ID:G1{IVU21:1')
    bragg = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr', add_prefix=(),
                read_attrs=['user_readback'])
    c2_x = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr', add_prefix=(),
               read_attrs=['user_readback'])
    epics_d_spacing = EpicsSignal('XF:05IDA-CT{IOC:Status01}DCMDspacing.VAL')
    epics_bragg_offset = EpicsSignal('XF:05IDA-CT{IOC:Status01}BraggOffset.VAL')

    # motor enable flags
    move_u_gap = Cpt(Signal, None, add_prefix=(), value=True)
    move_c2_x = Cpt(Signal, None, add_prefix=(), value=True)
    harmonic = Cpt(Signal, None, add_prefix=(), value=0, kind='config')
    selected_harmonic = Cpt(Signal, None, add_prefix=(), value=0)

    # experimental
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
        # set up constants
        Xoffset = self._xoffset
        d_111 = self._d_111
        delta_bragg = self._delta_bragg
        C2Xcal = self._c2xcal
        T2cal = self._t2cal
        etoulookup = self.etoulookup

        # calculate Bragg RBV
        BraggRBV = (
            np.arcsin((ANG_OVER_EV / target_energy) / (2 * d_111)) /
            np.pi * 180 -
            delta_bragg)

        # calculate C2X
        Bragg = BraggRBV + delta_bragg
        T2 = (Xoffset *
              np.sin(Bragg * np.pi / 180) /
              np.sin(2 * Bragg * np.pi / 180))
        dT2 = T2 - T2cal
        C2X = C2Xcal - dT2

        # calculate undulator gap

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
#        uga    scanlogDic = {'Fe': 11256, 'Cu':11254, 'Cr': 11258, 'Ti': 11260, 'Se':11251}
        # scanlogDic = {'Fe':11369, 'Cu':11367, 'Ti':11371, 'Se':11364}
        p = self.u_gap.get().readback
        utoelookup = self.utoelookup

        fundemental = float(utoelookup(ugap))

        energy = fundemental * harmonic

        return energy

    def __init__(self, *args,
                 xoffset=None, d_111=None, delta_bragg=None, C2Xcal=None, T2cal=None,
                 **kwargs):
        self._xoffset = xoffset
        self._d_111 = d_111
        self._delta_bragg = delta_bragg
        self._c2xcal = C2Xcal
        self._t2cal = T2cal
        super().__init__(*args, **kwargs)

        calib_path = Path(__file__).parent
        calib_file = 'SRXUgapCalibration20170612.txt'

        with open(calib_path / calib_file, 'r') as f:
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
        """Return the current physical gap between first and second crystals
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
#        if energy >= 25:
#            raise ValueError('The energy you entered is too high ({} keV). '
#                             'Maximum energy = 25.0 keV'.format(energy))
        if (energy > 25.):
            if (energy < 4400.) or (energy > 25000.):
            #energy is invalid
                raise ValueError('The requested photon energy is invalid ({} keV). '
                             'Values must be in the range of 4.4 - 25 keV'.format(energy))
            else:
            #energy is in eV
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
        # compute where we would move everything to in a perfect world
        bragg, c2_x, u_gap = self.energy_to_positions(energy, harmonic, detune)

        # sometimes move the crystal gap
        if not self.move_c2_x.get():
            c2_x = self.c2_x.position

        # sometimes move the undulator
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


# change it to a better way to pass the calibration
cal_data_2016cycle1 = {'d_111': 3.12961447804,
                       'delta_bragg': 0.322545952931,
                       'C2Xcal': 3.6,
                       'T2cal': 13.463294326,
                       'xoffset': 25.2521}

cal_data_2016cycle1_2 = {'d_111': 3.12924894907,  # 2016/1/27 (Se, Cu, Fe, Ti)
                       'delta_bragg': 0.315532509387,  # 2016/1/27 (Se, Cu, Fe, Ti)
                       'delta_bragg': 0.317124613301,  # ? before 8/18/2016
                       #'delta_bragg': 0.357124613301,  # 2016/8/16 (Cu)
                       # not in energy axis but for the record
                       # 'C1Rcal' :  -4.88949983261, # 2016/1/29
                       'C2Xcal': 3.6,  # 2016/1/29
                       'T2cal': 13.7187120636,
                       #'xoffset': 25.01567531283996 #2016 Jan
                       #'xoffset': 24.756374595028607 #2016/2/25 on Rh stripe, y=10 mm
                       #'xoffset': 24.908823293008666 #2016/2/26 on Si stripe, y=5 mm
                       #'xoffset': 24.621311485825125 #2016/3/10 Fe edge
                       #'xoffset': 24.661899615539824 #2016/3/13 Pt edge
                       #'xoffset': 24.761023845083287 #2016/3/17 Zr edge, 18.2 keV
                       #'xoffset': 24.741174927854445 #2016/3/23 15 keV
                       #'xoffset': 24.593840056028178 #2016/3/26 10.5 keV W
                       #'xoffset': 24.773110163531658 #2016/3/26 18.0 keV U
                       #'xoffset':  24.615016005680289 #2016/4/06 12.8 keV Se
                       #'xoffset': 24.672213516710034
                       #'xoffset': 24.809807128906538
                       #mono warmed up at 4/12/16 0.494304059171
                       #'xoffset': 24.809838976060604 #17keV
                       #'xoffset': 24.887490886653893 #8.2 keV
                       #'xoffset': 24.770168843970197 #12.5 keV
                       }  # 2016/1/29}

                        #2016-2
cal_data_2016cycle2  ={ #'d_111': 3.13130245128, #2016/6/9 (Ti, Cr, Fe, Cu, Se)
                        'd_111': 3.12929567478, #2016/8/1 (Ti, Fe, Cu, Se)
                        #'delta_bragg' : 0.309366522013,
                        #'delta_bragg' : 0.32936652201300004,
                        #'delta_bragg': 0.337124613301,
                        'delta_bragg': 0.317124613301, #2016/8/16 (Ti, Cr, Fe, Cu, Se)
                        #'xoffset': 24.864494684263519,
                        'C2Xcal': 3.6,  # 2016/1/29
                        'T2cal': 14.2470486188,
                        #'xoffset': 25.941277803299684
                        #'xoffset': 25.921698318063775
                        #'xoffset': 25.802588306223701 #2016/7/5 17.5 keV
                        #'xoffset': 25.542954465467645 #2016/7/6 17.5 keV fullfield
                        #'xoffset': 25.39464922124886 #2016/7/20 13.5/6 keV microscopy
                        #'xoffset': 25.354968816872358 #2016/7/28 12-14 keV microscopy
                        #'xoffset': 25.414219669872864 #2016/8/2 14 keV
                        #'xoffset': 25.175826062860775 #2016/8/2 18 keV
                        'xoffset': 25.527059255709876 #2016/8/16 9 keV
                        #'xoffset': 25.487723997622723 #2016/8/18 11 keV
                        #'xoffset': 25.488305806234468, #2016/8/21 9.2 keV, aligned by Garth on 2016/8/20
                        #'C1Rcal':-4.7089492561 for the record
                      }

cal_data_2016cycle3  ={'d_111': 3.12941028109, #2016/10/3 (Ti, Fe, Cu, Se)
                       'delta_bragg': 0.317209816326, #2016/10/3 (Ti, Fe, Cu, Se)
                        'C2Xcal': 3.6,  # 2016/1/29
                        'T2cal': 14.2470486188,
#                        'xoffset': 25.056582386746765, #2016/10/3 9 keV
#                        'xoffset': 25.028130552150312, #2016/10/12 8 keV
#                        'xoffset': 25.182303347383915, #2016/10/24 7.4 keV
#                        'xoffset': 25.531497575767418, #2016/10/24 7.4 keV
                        'xoffset': 25.491819462118674, #2016/11/4 12.8 keV
                        #'C1Rcal': -5.03023390228  #for the record, 2016/10/3
                      }


cal_data_2017cycle1  ={#'d_111': 3.12988412345, #2017/1/17 (Ti, Cr, Fe, Cu, Se)
#                        'd_111': 3.13246886211,
#                        'delta_bragg': 0.298805934621, # Cu, Se 3/17/2017
#                        'd_111': 3.11630891423,
#                        'delta_bragg': 0.357259819067, # {'Cu':3941, 'Se':3940, 'V':3937, 'Ti':3946, 'Mn':3951} 3/23/2017
                        'd_111': 3.12969964541,
                        'delta_bragg': 0.306854237528, # {'Ni':4156, 'Co':4154, 'Mn':4152} 4/10/2017
#                        'd_111': 3.14434282312,
#                        'delta_bragg': 0.265291543541, # {'Se':4002, 'Ti':4004, 'Mn':4000} 4/05/2017 bad
#                       'delta_bragg': 0.314906135851, #2017/1/17 (Ti, Cr, Fe, Cu, Se)
#                       'delta_bragg': 0.33490613585100004, #2017/3/9 (peakup spectrum)
#                       'delta_bragg': 0.309906135851, #2017/3/17 (Cu)
                        'C2Xcal': 3.6,  # 2017/1/17
                        'T2cal': 15.0347755916,  # 2017/1/17
                        'xoffset': 25.4253705456081, #2017/1/17 12.6 keV
                        #'C1Rcal': -4.98854110244  #for the record, #2017/1/17
                      }
cal_data_2017cycle2 = {
 'd_111': 3.12894833524,
 'delta_bragg': 0.309484915727, #{'Ti':4843,'Mn':4845,'Cu':4847,'Se':4848} 20170525
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.581644618999363, #value for Ti worked best using the C2 pitch to correct at higher E
}
cal_data_2017cycle3 = {
 'd_111': 3.10752771302,
 'delta_bragg': 0.352619283445, #{'Ti':7959,'V':7961,'Mn':7963,'Cu':7964,'Se':7965} 201700928
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.581644618999363, #value for Ti worked best using the C2 pitch to correct at higher E
}
"""
cal_data_2018cycle1 = {
 'd_111': 3.03796349212,
 'delta_bragg': 0.494304059171, #{'Ti':11371,'Fe':11369,'Cu':11367,'Se':11364} 201800203
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.581644618999363, #value for Ti worked best using the C2 pitch to correct at higher E
}
"""
cal_data_2018cycle1 = {
 'd_111': 3.12949171228,
 'delta_bragg': 0.309950475093, #{'Ti':11371,'Fe':11369,'Cu':11367,'Se':11364} 201800203
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.581644618999363, #value for Ti worked best using the C2 pitch to correct at higher E
}
cal_data_2018cycle1a = {
 'd_111': 3.1263, #changed by hand after DCM calibration changed; assume that heating drives this and not angle offset
 'delta_bragg': 0.309950475093, #{'Ti':11371,'Fe':11369,'Cu':11367,'Se':11364} 201800203
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.75, #best value for 12 and 5 keV
}
cal_data_2018cycle1b = {
 'd_111': 3.1291350735, #changed by hand after DCM calibration changed; assume that heating drives this and not angle offset
 'delta_bragg': 0.309886328328, #{'Ti':12195,'Fe':12194,'Se':12187} 20180224
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.75, #best value for 12 and 5 keV
}

cal_data_2018cycle2 = {
 'd_111': 3.128549107739033,
 'delta_bragg': 0.3139487894740349, # {'Fe':14476, 'V':14477, 'Cr':14478, 'Cu':14480, 'Se':14481, 'Zr':14482}
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.75, #best value for 12 and 5 keV
}

cal_data_2018cycle3 = {
 'd_111': 3.1292294240934786,
 'delta_bragg': 0.3113245678165956, # {'V':18037, 'Cr':18040, 'Fe':18043, 'Cu':18046, 'Se':18049, 'Zr':18052}
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.75, #best value for 12 and 5 keV
}

cal_data_2019cycle1 = {
 'd_111': 3.129024799425239,
 'delta_bragg': 0.3092257938019577, # {'V':21828, 'Cr':21830, 'Fe':21833, 'Cu':21835, 'Se':21838, 'Zr':21843}
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.465, #best value for 12 and 5 keV
}

cal_data_2019cycle1 = {
 'd_111': 3.143196587210581,
 'delta_bragg': 0.2011442467795649,
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.646398991691104 #best value for 12 and 5 keV
}

cal_data_2019cycle1 = {
 'd_111': 3.128774072188798,
 'delta_bragg': 0.22324196449806297,
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 # 'xoffset': 24.646398991691104 #best value for 12 and 5 keV
 'xoffset': 24.770  # not sure why value changed midcycle, reoptimized at 5 and 12 keV
}

# Calibrated 2019-08-16
# Use scans 29496 - 29503
cal_data_2019cycle3 = {
 'd_111': 3.1287603762011367,
 'delta_bragg': 0.21004620069236488,
 'C2Xcal': 3.6,
 'T2cal': 15.0347755916,
 'xoffset': 24.770
}

energy = Energy(prefix='', name='energy', **cal_data_2019cycle3)
energy.synch_with_epics()
energy.value = 1.0


# Front End Slits (Primary Slits)
class SRXSlitsFE(Device):
    top = Cpt(EpicsMotor, '3-Ax:T}Mtr')
    bot = Cpt(EpicsMotor, '4-Ax:B}Mtr')
    inb = Cpt(EpicsMotor, '3-Ax:I}Mtr')
    out = Cpt(EpicsMotor, '4-Ax:O}Mtr')


fe = SRXSlitsFE('FE:C05A-OP{Slt:', name='fe')

_time_fmtstr = '%Y-%m-%d %H:%M:%S'


# class TwoButtonShutter(Device):
#     # TODO this needs to be fixed in EPICS as these names make no sense
#     # the vlaue comingout of the PV do not match what is shown in CSS
#     open_cmd = Cpt(EpicsSignal, 'Cmd:Opn-Cmd', string=True)
#     open_val = 'Open'
# 
#     close_cmd = Cpt(EpicsSignal, 'Cmd:Cls-Cmd', string=True)
#     close_val = 'Not Open'
# 
#     status = Cpt(EpicsSignalRO, 'Pos-Sts', string=True)
# 
#     close_status = Cpt(EpicsSignalRO, 'Sts:Cls-Sts')
#     fail_to_close = Cpt(EpicsSignalRO, 'Sts:FailCls-Sts', string=True)
#     fail_to_open = Cpt(EpicsSignalRO, 'Sts:FailOpn-Sts', string=True)
#     # user facing commands
#     open_str = 'Open'
#     close_str = 'Close'
# 
#     def set(self, val):
#         if self._set_st is not None:
#             raise RuntimeError('trying to set while a set is in progress')
# 
#         cmd_map = {self.open_str: self.open_cmd,
#                    self.close_str: self.close_cmd}
#         target_map = {self.open_str: self.open_val,
#                       self.close_str: self.close_val}
# 
#         cmd_sig = cmd_map[val]
#         target_val = target_map[val]
# 
#         st = self._set_st = DeviceStatus(self)
#         enums = self.status.enum_strs
# 
#         def shutter_cb(value, timestamp, **kwargs):
#             value = enums[int(value)]
#             if value == target_val:
#                 self._set_st._finished()
#                 self._set_st = None
#                 self.status.clear_sub(shutter_cb)
#         uid = str(uuid.uuid4())
#         cmd_enums = cmd_sig.enum_strs
#         count = 0
# 
#         def cmd_retry_cb(value, timestamp, **kwargs):
#             nonlocal count
#             value = cmd_enums[int(value)]
#             # ts = datetime.datetime.fromtimestamp(timestamp).strftime(_time_fmtstr)
#             # print('sh', ts, val, st)
#             if count > 5:
#                 cmd_sig.clear_sub(cmd_retry_cb)
#                 st._finished(success=False)
#             if value == 'None':
#                 if not st.done:
#                     import time
#                     time.sleep(1)
#                     count += 1
#                     cmd_sig.set(1)
#                 else:
#                     cmd_sig.clear_sub(cmd_retry_cb)
# 
#         cmd_sig.subscribe(cmd_retry_cb, run=False)
#         cmd_sig.set(1)
#         self.status.subscribe(shutter_cb)
#         return st
# 
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._set_st = None
#         self.read_attrs = ['status']


shut_fe = TwoButtonShutter('XF:05ID-PPS{Sh:WB}', name='shut_fe')
shut_a = TwoButtonShutter('XF:05IDA-PPS:1{PSh:2}', name='shut_a')
shut_b = TwoButtonShutter('XF:05IDB-PPS:1{PSh:4}', name='shut_b')
