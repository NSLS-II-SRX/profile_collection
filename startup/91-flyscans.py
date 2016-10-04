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

    def __init__(self, encoder, detectors, motor, start, incr, dwell, Npts=1000):
        self._encoder = encoder
        self._detectors = detectors
        self._start = float(start)
        self._incr = float(incr)
        self._speed = float(incr / dwell)
        self._motor = motor
        self._npts = Npts

        return_values = []
        
    def kickoff(self):
        # Arm Zebra, trigger any other detectors
        devices = self._detectors
        return_values = []
        # set stage speed
        ret = yield from abs_set(motor.velocity,self._speed)
        return_values.append(ret)
        # set hdf parameters??
        for device in devices:
            if hasattr(device, 'hdf5'):
                ret = yield from abs_set(device.hdf5.capture, self._npts)

        for device in devices:
            if hasattr(device, 'trigger'):
                ret = yield Msg('trigger', device, group=fly-group)
                return_values.append(ret)
        # arm zebra
        ret = yield Msg('stage', self._encoder, group=fly-group)
        return_values.append(ret)
        # command continuous motion
        ret = yield Msg('set',self._motor, self._start+(self._npts*self._incr))
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
        # fetch data from Zebra and do something with it
        # halt xspress3
        yield Msg('read',self._encoder)
    
    def stop(self):
        pass

flyer = SRXFlyer(...)

def SRXFly(...):
    #conduct fly scan

    #parse user information for Zebra and stage settings

    #stage Zebra, Xspress3, F460

    yield from open_run(md)
    for n in num_rows:
        yield Msg('checkpoint')
        yield from flyer.kickoff(flyer, wait=True)
        yield from flyer.complete(flyer, wait=True)
        yield from flyer.collect(flyer)
    
    
    yield from close_run()
