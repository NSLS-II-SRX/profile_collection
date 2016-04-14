import time as ttime

from ophyd import (PVPositioner, EpicsSignal, EpicsSignalRO, EpicsMotor,
                   Device, Signal, PseudoPositioner, PseudoSingle)
from ophyd.utils.epics_pvs import set_and_wait
from ophyd.ophydobj import StatusBase, MoveStatus
from ophyd import Component as Cpt
from scipy.interpolate import InterpolatedUnivariateSpline

ring_current = EpicsSignalRO('SR:C03-BI{DCCT:1}I:Real-I', name='ring_current')

class UVDone(PermissiveGetSignal):
    def __init__(self, parent, brake, readback, **kwargs):
        super().__init__(parent=parent, value=1, **kwargs)
        self._rbv = readback
        self._brake = brake
        self._started = False

    def put(self, *arg, **kwargs):
        raise TypeError("You con not tell an undulator it is done")

    def _put(self, *args, **kwargs):
        return super().put(*args, **kwargs)

    def _watcher(self, obj=None, value=None, **kwargs):
        target = self.target
        rbv = getattr(self.parent, self._rbv)
        cur_value = rbv.get()
        brake = getattr(self.parent, self._brake)
        brake_on = brake.get()
        if not self._started:
            self._started = not brake_on
        # come back and check this threshold value
        #if brake_on and abs(target - cur_value) < 0.002:
        if abs(target - cur_value) < 0.002:
            self._put(1)
            rbv.clear_sub(self._watcher)
            brake.clear_sub(self._watcher)
            self._started = False

        elif brake_on and self._started:
            print(self.parent.name, ": reactuated due to not reaching target")
            self.parent.actuate.put(self.parent.actuate_value)

    def reset(self, target):
        self._put(0)
        self.target = target
        self._started = False


class URealPos(Device):
    #undulator real position, gap and taper
    ds_low = Cpt(EpicsSignalRO, '}REAL_POSITION_DS_LOWER')
    ds_upp = Cpt(EpicsSignalRO, '}REAL_POSITION_DS_UPPER')
    us_low = Cpt(EpicsSignalRO, '}REAL_POSITION_US_LOWER')
    us_upp = Cpt(EpicsSignalRO, '}REAL_POSITION_US_UPPER')


class UPos(Device):
    #undulator positions, gap and taper
    ds_low = Cpt(EpicsSignalRO, '}POSITION_DS_LOWER')
    ds_upp = Cpt(EpicsSignalRO, '}POSITION_DS_UPPER')
    us_low = Cpt(EpicsSignalRO, '}POSITION_US_LOWER')
    us_upp = Cpt(EpicsSignalRO, '}POSITION_US_UPPER')

class GapPos(Device):
    gap_avg = Cpt(EpicsSignalRO, '}GAP_AVG')
    gap_ds = Cpt(EpicsSignalRO, '}GAP_DS')
    gap_us = Cpt(EpicsSignalRO, '}GAP_US')
    gap_taper = Cpt(EpicsSignalRO, '}GAP_TAPER')


class Girder(Device):
    lower_tilt = Cpt(EpicsSignalRO, '}GIRDER_LOWER_TILT')
    upper_tile = Cpt(EpicsSignalRO, '}GIRDER_UPPER_TILT')
    tilt_error = Cpt(EpicsSignalRO, '}GIRDER_TILT_ERROR')
    tilt_limit = Cpt(EpicsSignalRO, '}GIRDER_TILT_LIMIT')


class Elev(Device):
    ct_us =     Cpt(EpicsSignalRO, '-LEnc:1}Pos')
    offset_us = Cpt(EpicsSignalRO, '-LEnc:1}Offset:RB')
    ct_ds =     Cpt(EpicsSignalRO, '-LEnc:6}Pos')
    offset_ds = Cpt(EpicsSignalRO, '-LEnc:6}Offset:RB')


class FixedPVPositioner(PVPositioner):
    """This subclass ensures that the setpoint is really set before
    """
    def _move_async(self, position, **kwargs):
        '''Move and do not wait until motion is complete (asynchronous)'''
        if self.actuate is not None:
            set_and_wait(self.setpoint, position)
            self.actuate.put(self.actuate_value, wait=False)
        else:
            self.setpoint.put(position, wait=False)
    
    def move(self, v, *args, **kwargs):
        kwargs['timeout'] = None
        self.done.reset(v)
        ret = super().move(v, *args, **kwargs)
        self.brake_on.subscribe(self.done._watcher,
                                event_type=self.brake_on.SUB_VALUE)
        self.readback.subscribe(self.done._watcher,
                                event_type=self.readback.SUB_VALUE)
        return ret


class Undulator(FixedPVPositioner):
    # positioner signals
    setpoint = Cpt(EpicsSignal, '-Mtr:2}Inp:Pos')
    readback = Cpt(EpicsSignalRO, '-LEnc}Gap')
    stop_signal = Cpt(EpicsSignal, '-Mtrc}Sw:Stp')
    actuate = Cpt(EpicsSignal, '-Mtr:2}Sw:Go')
    actuate_value = 1
    done = Cpt(UVDone, None, brake='brake_on',
               readback='readback', add_prefix=())

    # correction function signals, need to be merged into single object
    corrfunc_en = Cpt(EpicsSignal, '-MtrC}EnaAdj:out')
    corrfunc_dis = Cpt(EpicsSignal, '-MtrC}DisAdj:out')
    corrfunc_sta = Cpt(EpicsSignal, '-MtrC}AdjSta:RB')

    # brake status
    brake_on = Cpt(EpicsSignalRO, '-Mtr:2}Rb:Brk')

    # low-level positional information about undulator
    real_pos = Cpt(URealPos, '')
    pos = Cpt(UPos, '')
    girder = Cpt(Girder, '')
    elevation = Cpt(Elev, '')

    def move(self, v, *args, moved_cb=None, **kwargs):
        kwargs['timeout'] = None
        if np.abs(v - self.position) < .001:
            self._started_moving = True
            self._moving = False
            self._done_moving()
            st = MoveStatus(self, v)
            if moved_cb:
                moved_cb(obj=self)
            st._finished()
            return st
        return super().move(v, *args, moved_cb=moved_cb, **kwargs)

    def __init__(self, *args, calib_path=None, calib_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        # todo make these error messages look more like standard exceptions
        if calib_path is None:
            raise TypeError("must provide calib_dir")
        if calib_file is None:
            raise TypeError("must provide calib_file")

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


_undulator_kwargs = dict(name='ivu1_gap', read_attrs=['readback'],
                         calib_path='/nfs/xf05id1/UndulatorCalibration/',
                         calib_file='SRXUgapCalibration20150411_final.text',
                         configuration_attrs=['corrfunc_sta', 'pos', 'girder',
                                              'real_pos', 'elevation'])

ANG_OVER_EV = 12.3984

class Energy(PseudoPositioner):
    # synthetic axis
    energy = Cpt(FixedPseudoSingle)
    # real motors
    u_gap = Cpt(Undulator, 'SR:C5-ID:G1{IVU21:1', add_prefix=(), **_undulator_kwargs)
    bragg = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr', add_prefix=(),
                read_attrs=['user_readback'])
    c2_x = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr', add_prefix=(),
                read_attrs=['user_readback'])
    # motor enable flags
    move_u_gap = Cpt(PermissiveGetSignal, None, add_prefix=(), value=True)
    move_c2_x = Cpt(PermissiveGetSignal, None, add_prefix=(), value=True)
    harmonic = Cpt(PermissiveGetSignal, None, add_prefix=(), value=None)

    # experimental
    detune = Cpt(PermissiveGetSignal, None, add_prefix=(), value=0)

    def energy_to_positions(self, target_energy, undulator_harmonic, u_detune):
        """Compute undulator and mono positions given a target energy

        Paramaters
        ----------
        target_energy : float
            Target energy in keV

        undulator_harmonic : int, optional
            The harmonic in the undulator to use

        uv_mistune : float, optional
            Amount to 'mistune' the undulator in keV.  Will settings such that the
            peak of the undulator spectrum will be at `target_energy + uv_mistune`.

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
        etoulookup = self.u_gap.etoulookup


        #calculate Bragg RBV
        BraggRBV = np.arcsin((ANG_OVER_EV / target_energy)/(2 * d_111))/np.pi*180 - delta_bragg

        #calculate C2X
        Bragg = BraggRBV + delta_bragg
        T2 = Xoffset * np.sin(Bragg * np.pi / 180)/np.sin(2 * Bragg * np.pi / 180)
        dT2 = T2 - T2cal
        C2X = C2Xcal - dT2

        #calculate undulator gap
        # TODO make this more sohpisticated to stay a fixed distance off the
        # peak of the undulator energy
        ugap = float(etoulookup((target_energy + u_detune)/undulator_harmonic))

        return BraggRBV, C2X, ugap

    def undulator_energy(self, harmonic=3):
        """Return the current enegry peak of the undulator at the given harmonic

        Paramaters
        ----------
        harmanic : int, optional
            The harmonic to use, defaults to 3
        """
        ugap = self.u_gap.get().readback
        utoelookup = self.u_gap.utoelookup

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

    def forward(self, p_pos):
        energy = p_pos.energy
        harmonic = self.harmonic.get()
        detune = self.detune.get()
        if energy <= 4.4:
            raise ValueError("The energy you entered is too low ({} keV). "
                             "Minimum energy = 4.4 keV".format(energy))
        if energy >= 25:
            raise ValueError('The energy you entered is too high ({} keV). '
                             'Maximum energy = 25.0 keV'.format(energy))

        if harmonic is None:
            harmonic = 3
            #choose the right harmonic
            braggcal, c2xcal, ugapcal = self.energy_to_positions(energy, harmonic, detune)
            # try higher harmonics until the required gap is too small
            while True:
                braggcal, c2xcal, ugapcal = self.energy_to_positions(energy, harmonic + 2, detune)
                if ugapcal < 6.4:
                    break
                harmonic += 2

        # compute where we would move everything to in a perfect world
        bragg, c2_x, u_gap = self.energy_to_positions(energy, harmonic, detune)

        # sometimes move the crystal gap
        if not self.move_c2_x.get():
            c2_x = self.c2_x.position

        # sometimes move the undulator
        if not self.move_u_gap.get():
            u_gap = self.u_gap.position

        return self.RealPosition(bragg=bragg, c2_x=c2_x, u_gap=u_gap)

    def inverse(self, r_pos):
        bragg = r_pos.bragg
        e = ANG_OVER_EV / (2 * self._d_111 * np.sin(np.deg2rad(bragg + self._delta_bragg)))
        return self.PseudoPosition(energy=e)

# change it to a better way to pass the calibration
cal_data_2016cycle1 = {'d_111': 3.12961447804,
                       'delta_bragg': 0.322545952931,
                       'C2Xcal': 3.6,
                       'T2cal': 13.463294326,
                       'xoffset': 25.2521}

cal_data_2016cycle2 = {'d_111': 3.12924894907,  # 2016/1/27 (Se, Cu, Fe, Ti)
                       'delta_bragg': 0.315532509387,  # 2016/1/27 (Se, Cu, Fe, Ti)
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
                       #mono warmed up at 4/12/16
                       #'xoffset': 24.809838976060604 #17keV
                       #'xoffset': 24.887490886653893 #8.2 keV
                       'xoffset': 24.770168843970197 #12.5 keV
                       }  # 2016/1/29}

energy = Energy(prefix='', name='energy', **cal_data_2016cycle2)


# Front End Slits (Primary Slits)
class SRXSlitsFE(Device):
    top = Cpt(EpicsMotor, '3-Ax:T}Mtr')
    bot = Cpt(EpicsMotor, '4-Ax:B}Mtr')
    inb = Cpt(EpicsMotor, '3-Ax:I}Mtr')
    out = Cpt(EpicsMotor, '4-Ax:O}Mtr')

fe = SRXSlitsFE('FE:C05A-OP{Slt:', name='fe')

class SRXShutter(Device):
    close_cmd = Cpt(EpicsSignal, 'Cmd:Cls-Cmd')
    open_cmd = Cpt(EpicsSignal, 'Cmd:Opn-Cmd')
    close_status = Cpt(EpicsSignalRO, 'Sts:Cls-Sts')

shut_fe = SRXShutter('XF:05IDB-PPS{Sh:WB}', name='shut_fe')
shut_a = SRXShutter('XF:05IDB-PPS:1{PSh:2}', name='shut_a')
shut_b = SRXShutter('XF:05IDB-PPS:1{PSh:4}', name='shut_b')
