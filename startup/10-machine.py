import time as ttime

from ophyd import PVPositionerPC, EpicsSignal, EpicsSignalRO, EpicsMotor, Device
from ophyd.ophydobj import StatusBase
from ophyd import Component as Cpt
from scipy.interpolate import InterpolatedUnivariateSpline


class Undulator(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, '-Mtr:2}Inp:Pos')
    readback = Cpt(EpicsSignalRO, '-LEnc}Gap')
    stop_signal = Cpt(EpicsSignal, '-Mtrc}Sw:Stp')
    go_signal = Cpt(EpicsSignal, '-Mtr:2}Sw:Go')
    corrfunc_en = Cpt(EpicsSignal, '-MtrC}EnaAdj:out')
    corrfunc_dis = Cpt(EpicsSignal, '-MtrC}DisAdj:out')
    corrfunc_sta = Cpt(EpicsSignal, '-MtrC}AdjSta:RB')

    def trigger(self):
        self.go_signal.put(1)
        status = StatusBase()
        target = self.setpoint.get()

        rb = self.readback

        def done_cb(**kwargs):
            cur_value = self.readback.get()
            if abs(target - cur_value) < 0.002:
                status._finished()
                rb._reset_sub(rb.SUB_VALUE)

        rb.subscribe(done_cb, event_type=rb.SUB_VALUE)
        return status

    def set(self, setpoint):
        # TODO make this not maybe lie
        self.setpoint.put(setpoint)
        status = StatusBase()
        status._finished()
        return status

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
                     configuration_attrs=['corrfunc_sta'], 
                     calib_path='/nfs/xf05id1/UndulatorCalibration/', 
                     calib_file='SRXUgapCalibration20150411_final.text')\

ivu1_gap = Undulator('SR:C5-ID:G1{IVU21:1', **_undulator_kwargs)
ANG_OVER_EV = 12.3984

class Energy(Device):
    uv = Cpt(Undulator, 'SR:C5-ID:G1{IVU21:1', add_prefix=(), **_undulator_kwargs)
    bragg = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr', add_prefix=(),
                read_attrs=['user_readback'])
    c_gap = Cpt(EpicsMotor, 'XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr', add_prefix=(),
                read_attrs=['user_readback'])

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
        C2X = self.c_gap.get().user_readback
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

    def set(self, energy, *, harmonic=3):
        bragg, c_gap, ugag = self.energy_to_positions(energy, harmonic)
        st1 = self.bragg.set(bragg)
        st2 = self.c_gap.set(c_gap)
        st3 = self.uv.set(ugag)
        return MergedStatus(st1, st2, st3)

    def stage(self):
        self.stage_sigs.update([(self.uv, 5),])
        super().stage()

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
            
        

engergy = Energy(prefix='', name='energy', 
                 d_111=3.12961447804, 
                 delta_bragg=0.322545952931,
                 C2Xcal=3.6, 
                 T2cal=13.463294326, 
                 xoffset=25.2521)

# Front End Slits (Primary Slits)

fe_tb = EpicsMotor('FE:C05A-OP{Slt:3-Ax:T}Mtr', name='fe_tb')
fe_bb = EpicsMotor('FE:C05A-OP{Slt:4-Ax:B}Mtr', name='fe_bb')
fe_ib = EpicsMotor('FE:C05A-OP{Slt:3-Ax:I}Mtr', name='fe_ib')
fe_ob = EpicsMotor('FE:C05A-OP{Slt:4-Ax:O}Mtr', name='fe_ob')

