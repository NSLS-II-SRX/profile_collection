#####
# Pseudocode for fly scanning
# User supplies:
# X start       Xstart is used for gating, Xo is an X stage value where the scan
#   will start
# Y start       Ystart is used for gating, Yo is an Y stage value where the scan
#   will start
# range X       scan X size
# dwell         time spent per atom, range X and dwell give stage speed
# range Y       scan Y size
#####

#Transverse flyer for SRX: fly X, step Y
#Aerotech X stage, DC motor need to set desired velocity
#Zebra staged, data read and written to file, file registered
#Options:  open and write to Xspress3 HDF5, write to HDF5 and link in 
#   Xspress3 data, write separately in some format and register later

class SRXFlyer1Axis:

#    def __init__(self, encoder, xspress3, motor, start, incr, dwell, Npts=1000):
    def __init__(self, encoder, motor, start, incr, dwell, Npts=1000):
        self._encoder = encoder
#        self._xspress3 = xspress3
        self.start = float(start)
        self.incr = float(incr)
        self.dwell = float(dwell)
        self._motor = motor
        self.npts = Npts
        self.speed = float(self.incr / self.dwell)

    def stage(self):
        #in principle, one could change these and restage...
        self.stage_sigs[self._motor.velocity] = self.speed
        self.stage_sigs[self._encoder.arm] = 1
        self.stage_sigs[self._encoder.gate_num = self.npts

        super().stage()

    def unstage(self):
        super().unstage()
        

    def kickoff(self):
        return_values = []
#        # set stage speed
#        self._motor.velocity = self._speed
#        #####need to reset this!!!####
#        # set number of gates
#        self._encoder.gate_num = self.npts
#        # arm motion capture
#        self._encoder.arm = 1
#        # set hdf parameters where applicable
#        for device in devices:
#            if hasattr(device, 'hdf5'):
#                device.hdf5.capture = self.npts
        #trigger detectors
#        ret = yield Msg('trigger', self._xspress3, group=fly-group)
#        return_values.append(ret)
        # command continuous motion
        ret = yield Msg('set',self.motor, self.start+(self.npts*self.incr))
        return_values.append(ret)
        return return_values

    def describe_collect(self):
        return OrderedDict() 
    
    def read_configuration(self):
        return OrderedDict()
    
    def describe_configuration(self):
        return OrderedDict()
    
    def complete(self):
        return NullStatus()
    
    def collect(self):
        # fetch data from Zebra
        data = self._encoder.pc.data.get()
        np.array(data.enc1)
        np.array(data.time)
        # now what?
    
    def stop(self):
        pass


def SRXFly(xstart=None,xstepsize=None,xpts=None,dwell=None,ystart=None,ystepsize=None,ypts=None,xs=xs):
    #xspress3 scan-specific set up
    xs.hdf5.capture = xpts

    rows = np.linspace(ystart,ystart+(ypts-1)*ystepsize,ypts)

    md = ?
    try:
        xs.external_trig = True
        yield from open_run(md)
        for n in rows:
            #flyer = SRXFlyer(encoder, detectors, motor, start, incr, dwell, Npts=1000)
            flyer = SRXFlyer(zebra, hf_stage.x, xstart, xstepsize, dwell, xpts)
            flyer.stage()
            yield Msg('checkpoint')
            yield Msg('stage', xs)
            yield Msg('stage',(flyer)
            yield from set_abs(hf_stage.y,n,wait=True)
            yield Msg('trigger',xs)
            yield from kickoff(flyer, wait=True)
            yield from complete(flyer, wait=True)
            yield from collect(flyer)
            yield Msg('unstage',xs)
            yield Msg('unstage',flyer)
        yield from close_run()
    finally:
        xs.external_trig = False
