import time as ttime

from ophyd import PVPositioner, EpicsSignal, EpicsSignalRO, EpicsMotor, Device, Signal
from ophyd.ophydobj import StatusBase
from ophyd import Component as Cpt
from scipy.interpolate import InterpolatedUnivariateSpline

class UVDone(Signal):
    def __init__(self, parent, brake, readback, **kwargs):
        super().__init__(parent=parent, value=0, **kwargs)
        self._rbv = readback
        self._brake = brake

    def put(self, *arg, **kwargs):
        raise TypeError("You con not tell an undulator it is done")

    def _put(self, *args, **kwargs):
        return super().put(*args, **kwargs)
    
    def _watcher(self, obj=None, **kwargs):
        target = self.target
        rbv = getattr(self.parent, self._rbv)
        cur_value = rbv.get()
        brake = getattr(self.parent, self._brake)
        brake_on = brake.get()
        if brake_on and abs(target - cur_value) < 0.002:
            self._put(1)
            rbv.clear_sub(self._watcher)
            brake.clear_sub(self._watcher)

    def reset(self, target):
        self._put(0)
        self.target = target


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
    offset_us = Cpt(EpicsSignalRO, '-LEnc:1}Offset')
    ct_ds =     Cpt(EpicsSignalRO, '-LEnc:6}Pos')
    offset_ds = Cpt(EpicsSignalRO, '-LEnc:6}Offset')

class Undulator(PVPositioner):
    setpoint = Cpt(EpicsSignal, '-Mtr:2}Inp:Pos')
    readback = Cpt(EpicsSignalRO, '-LEnc}Gap')
    stop_signal = Cpt(EpicsSignal, '-Mtrc}Sw:Stp')
    actuate = Cpt(EpicsSignal, '-Mtr:2}Sw:Go')
    actuate_value = 1
    done = Cpt(UVDone, None, brake='brake_on', 
               readback='readback', add_prefix=())

    corrfunc_en = Cpt(EpicsSignal, '-MtrC}EnaAdj:out')
    corrfunc_dis = Cpt(EpicsSignal, '-MtrC}DisAdj:out')
    corrfunc_sta = Cpt(EpicsSignal, '-MtrC}AdjSta:RB')
    brake_on = Cpt(EpicsSignalRO, '-Mtr:2}Rb:Brk')

    real_pos = Cpt(URealPos, '')
    pos = Cpt(UPos, '')
    girder = Cpt(Girder, '')
    elevation = Cpt(Elev, '')

    def _trigger(self):
        self.go_signal.put(1)
        status = StatusBase()
        target = self.setpoint.get()

        rb = self.readback

        def done_cb(**kwargs):
            cur_value = self.readback.get()
            if abs(target - cur_value) < 0.002:
                status._finished()
                rb.clear_sub(done_cb)

        rb.subscribe(done_cb, event_type=rb.SUB_VALUE)
        return status

    def set(self, v, *args, **kwargs):
        self.done.reset(v)
        ret = super().set(v, *args, **kwargs)
        self.brake_on.subscribe(self.done._watcher, 
                                event_type=self.brake_on.SUB_VALUE)
        self.readback.subscribe(self.done._watcher, 
                                event_type=self.readback.SUB_VALUE) 
        return ret

    def configure(self, d):
        raise NotImplemented("coor fuc status is broken")
    
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

class Energy(Device):     
    uv = Cpt(Undulator, 'SR:C5-ID:G1{IVU21:1', add_prefix=(), **_undulator_kwargs)
    bragg = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr', add_prefix=(),
                read_attrs=['user_readback'])
    c2_x = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr', add_prefix=(),
                read_attrs=['user_readback'])
    move_uv = Cpt(Signal, None, add_prefix=(), value=True)
    move_c2_x = Cpt(Signal, None, add_prefix=(), value=True)

    def energy_to_positions(self, target_energy, undulator_harmonic=3, uv_mistune=0):
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
        etoulookup = self.uv.etoulookup


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
        ugap = float(etoulookup((target_energy + uv_mistune)/undulator_harmonic))
        
        return BraggRBV, C2X, ugap         

    def mono_energy(self):
        brag = self.bragg.get().user_readback
        return ANG_OVER_EV/(2*self._d_111*np.sin((brag+self._delta_bragg)*np.pi/180))

    
    def undulator_energy(self, harmonic=3):
        """Return the current enegry peak of the undulator at the given harmonic

        Paramaters
        ----------
        harmanic : int, optional
            The harmonic to use, defaults to 3
        """
        ugap = self.uv.get().readback
        utoelookup = self.uv.utoelookup

        fundemental = float(utoelookup(ugap))

        energy = fundemental * harmonic
                
        return energy
        
    def read(self, *args, **kwargs):
        ret = super().read()
        ret['energy'] = {'value':self.mono_energy(), 'timestamp': ttime.time()}
        return ret

    def describe(self):
        ret = super().describe()
        ret['energy'] =  {'dtype': 'number',
                          'shape': [],
                          'source': 'computed',
                          'units': 'keV'}
        return ret
    
    def __init__(self, *args, 
                 xoffset=None, d_111=None, delta_bragg=None, C2Xcal=None, T2cal=None, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._xoffset = xoffset
        self._d_111 = d_111
        self._delta_bragg = delta_bragg
        self._c2xcal = C2Xcal
        self._t2cal = T2cal

    def cyrstal_gap(self):
        """Return the current physical gap between first and second crystals
        """
        C2X = self.c2_x.get().user_readback
        bragg = self.bragg.get().user_readback

        T2cal = self._t2cal
        delta_bragg = self._delta_bragg
        d_111 = self._d_111
        c2x_cal = self._c2xcal

        Bragg = np.pi/180 * (bragg + delta_bragg)

        dT2 = c2xcal - C2X
        T2 = dT2 + T2cal

        XoffsetVal = T2/(np.sin(Bragg)/np.sin(2*Bragg))
    
        return XoffsetVal

    def set(self, energy, *, harmonic=None):
        if energy <= 4.4:
            raise ValueError("The energy you entered is too low ({} keV). "
                             "Minimum energy = 4.4 keV".format(energy))
        if energy >= 25:
            raise ValueError('The energy you entered is too high ({} keV). '
                             'Maximum energy = 25.0 keV'.format(energy))

        if harmonic is None:
            harmonic = 3
            #choose the right harmonic
            braggcal, c2xcal, ugapcal = self.energy_to_positions(energy, harmonic)
            # try higher harmonics until the required gap is too small
            while True:
                braggcal, c2xcal, ugapcal = self.energy_to_positions(energy, harmonic + 2)
                if ugapcal < 6.4:
                    break
                harmonic += 2


        bragg, c2_x, ugag = self.energy_to_positions(energy, harmonic)
        st = []
        # always rotate the crystal
        st1 = self.bragg.set(bragg)
        st.append(st1)
        # sometimes move the crystal gap
        if self.move_c2_x.get():
            st2 = self.c2_x.set(c2_x)
            st.append(st2)
        # sometimes move the undulator
        if self.move_uv.get():
            st3 = self.uv.set(ugag)
            st.append(st3)
        return MergedStatus(*st)

    def stop(self):
        self.uv.stop()
        self.bragg.stop()
        self.c2_x.stop()


class MergedStatus(StatusBase):
    def __init__(self, *stats):
        super().__init__()
        self.done_count = 0
        total = len(stats)

        def inner():
            with self._lock:
                self.done_count += 1
                if self.done_count == total:
                    self._finished()
                
        for st in stats:
            st.finished_cb = inner
        self._stats = stats
    
    def __repr__(self):
        return repr(self._stats)
            
        

energy = Energy(prefix='', name='energy', 
                d_111=3.12961447804, 
                delta_bragg=0.322545952931,
                C2Xcal=3.6, 
                T2cal=13.463294326, 
                xoffset=25.2521, 
                configuration_attrs=['uv', 'move_uv', 'move_c2_x'])

# Front End Slits (Primary Slits)

fe_tb = EpicsMotor('FE:C05A-OP{Slt:3-Ax:T}Mtr', name='fe_tb')
fe_bb = EpicsMotor('FE:C05A-OP{Slt:4-Ax:B}Mtr', name='fe_bb')
fe_ib = EpicsMotor('FE:C05A-OP{Slt:3-Ax:I}Mtr', name='fe_ib')
fe_ob = EpicsMotor('FE:C05A-OP{Slt:4-Ax:O}Mtr', name='fe_ob')

