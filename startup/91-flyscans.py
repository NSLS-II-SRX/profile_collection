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
    def __init__(self, encoder, motor, start, incr, dwell, Npts=1000, delta):
        self._encoder = encoder
#        self._xspress3 = xspress3
        self.start = float(start)
        self.extent = float(start) + float(Ntps * incr)
        self.incr = float(incr)
        self.dwell = float(dwell)
        self._motor = motor
        self.npts = Npts
        self.speed = float(self.incr / self.dwell)
        self.delta = delta
        
        self._encoder.enc_pos1_sync = 1 

    def stage(self):
        # X speed
        self.stage_sigs[self._motor.velocity] = self.speed
        # gating info for encoder capture
        self.stage_sigs[self._encoder.gate_start] = self.start
        self.stage_sigs[self._encoder.gate_width] = self.extent
        self.stage_sigs[self._encoder.gate_step] = self.incr
        self.stage_sigs[self._encoder.gate_num] = self.npts
        self.stage_sigs[self._encoder.arm] = 1
        #pc gate output is 30 for zebra.  use it to trigger xspress3 and I0
        self.stage_sigs[self._encoder.output1] = 30
        self.stage_sigs[self._encoder.output3] = 30

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
        ret = yield Msg('set',self.motor, self.start+(self.npts*self.incr)+self.delta)
        return_values.append(ret)
        return return_values

    def describe_collect(self):
        return OrderedDict() 
    
    def read_configuration(self):
        return OrderedDict()
    
    def describe_configuration(self):
        return OrderedDict()
    
    def complete(self):
        #depends on motor position and self._encoder.data_in_progress bool
        #first, motor completes, then data download begins.
        return NullStatus()
    
    def collect(self):
        # fetch data from Zebra
        data = self._encoder.pc.data.get()
        output = np.zeros((2, len(data[0])))
        output[0] = np.array(data.time)
        output[1] = np.array(data.enc1)
        # will cache these to xf05id1 user nfs directory and copy them to the 
        # data volume
        userdir = '/nfs/xf05id1/.fs_cache'
        data_path = os.path.join(userdir, str(datetime.date.today())
        uid = save_ndarray(output,data_path)
        resource_document = fsapi.insert_resource('npy', data_path)
        datum_document = fsapi.insert_datum(resource_document, uid, {})
        # need to register this with metadatastore as x values...
        # how?
    
    def stop(self):
        pass


def SRXFly(xstart=None,xstepsize=None,xpts=None,dwell=None,ystart=None,ystepsize=None,ypts=None,xs=xs,ion=current_preamp):
    delta = 0.01
    rows = np.linspace(ystart,ystart+(ypts-1)*ystepsize,ypts)

    md = ChainMap(md, {
        'detectors': [zebra,xs],
        'x_range' : xstepsize*xpts,
        'dwell' : dwell,
        'y_range' : ystepzie*ypts,
        }
    )
    try:
        xs.external_trig = True
        #set ion chamber to windowing mode
        ion.trigger_mode = 5 
        yield from set_abs(hf_stage.x, xstart-delta, wait=True)
        yield from open_run(md)
        for n in rows:
            #xspress3 scan-specific set up
            xs.hdf5.capture = xpts
            #flyer = SRXFlyer(encoder, detectors, motor, start, incr, dwell, Npts=1000)
            flyer = SRXFlyer(zebra, hf_stage.x, xstart, xstepsize, dwell, xpts, delta)
            flyer.stage()
            yield Msg('checkpoint')
            yield Msg('stage', xs)
            yield Msg('stage', ion)
            yield Msg('trigger',xs)
            #might be better to send I0 to file than database.  how?
            yield Msg('monitor', ion)
            yield Msg('stage', flyer)
            yield from set_abs(hf_stage.y, n, wait=True)
            yield from kickoff(flyer, wait=True)
            yield from complete(flyer, wait=True)
            yield from collect(flyer)
            yield Msg('unstage', flyer)
            yield Msg('unmonitor', ion)
            yield Msg('unstage', ion)
            yield Msg('unstage', xs)
            #run-specific metadata?
        yield from close_run()
    finally:
        xs.external_trig = False
        ion.trigger_mode = 2
