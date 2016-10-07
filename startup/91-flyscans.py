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
#
from filestore import filestore.commands as fs
import uuid

class SRXFlyer1Axis:
    LARGE_FILE_DIRECTORY_READ_PATH = '/tmp/test_data'
    LARGE_FILE_DIRECTORY_WRITE_PATH = '/tmp/test_data'
    "This is the Zebra."
#    def __init__(self, encoder, xspress3, motor, start, incr, dwell, Npts=1000):
    def __init__(self, encoder, motor, start, incr, dwell, delta, Npts=1000, *, fs=fs):
        self._mode = 'idle'
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

        # X speed
        self.stage_sigs[self._motor.velocity] = self.speed
        # gating info for encoder capture
        self.stage_sigs[self._encoder.gate_start] = self.start
        self.stage_sigs[self._encoder.gate_width] = self.extent
        self.stage_sigs[self._encoder.gate_step] = self.extent 
        self.stage_sigs[self._encoder.gate_num] = 1
        self.stage_sigs[self._encoder.pulse_start] = 0
        self.stage_sigs[self._encoder.pulse_width] = self.dwell-0.005
        self.stage_sigs[self._encoder.pulse_step] = self.dwell
        self.stage_sigs[self._encoder.pulse_num] = self.npts

        self.stage_sigs[self._encoder.arm] = 1
        #pc gate output is 30 for zebra.  use it to trigger xspress3 and I0
        self.stage_sigs[self._encoder.output1] = 31
        self.stage_sigs[self._encoder.output3] = 31
        
        self._encoder.enc_pos1_sync = 1 

    def stage(self):
        super().stage()

    def kickoff(self, xstart, xstop, xnum, dwell):
        self._mode = 'kicked off'
        self.gate_start.put(xstart)
        self.gate_stop.put(xstop)
        self.pulse_max.put(xnum)
        self.gate_width.put(dwell)
        self.arm.put(1)
        return NullStatus()  # TODO Return a status object *first* and do the above asynchronously.

        # NOTHING BELOW IS EVER RUN.
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

    def complete(self):
        """
        Call this when all needed data has been collected. This has no idea
        whether that is true, so it will obligingly stop immediately. It is
        up to the caller to ensure that the motion is actually complete.
        """
        self._mode = 'complete'
        self.arm.put(0)  # sanity check; this should happen automatically
        return NullStatus()
    
    def collect(self):
        # Create records in the FileStore database.
        filename = '{}.h5'.format(uuid.uuid4())
        read_filepath = os.path.join(self.LARGE_FILE_DIRECTORY_READ_PATH, filename)
        write_filepath = os.path.join(self.LARGE_FILE_DIRECTORY_WRITE_PATH, filename)
        resource = fs.insert_resource('ZEBRA_HDF51', read_filepath)
        time_datum_id = uuid.uuid4()
        enc1_datum_id = uuid.uuid4()
        fs.insert_datum(resource, time_datum_id, {'column': 'time'})
        fs.insert_datum(resource, enc1_datum_id, {'column': 'enc1'})

        # Write the file.
        export_zebra_data(self._encoder, write_filepath)

        # Yield a (partial) Event document. The RunEngine will put this
        # into metadatastore, as it does all readings.
        yield {'time': time.time(), 'seq_num': 1,
               'data': {'time': time_datum_id,
                        'enc1': enc1_datum_id},
               'timestamps': {'time': time_datum_id,  # not a typo
                              'enc1': time_datum_id}}
        self._mode = 'idle'
    
    def stop(self):
        pass

    def pause(self):
        "Pausing in the middle of a kickoff nukes the partial dataset."
        self.arm.put(0)
        self._mode = 'idle'
        self.unstage()

    def resume(self):
        self.stage()


def export_zebra_data(zebra, filepath):
    data = zebra.pc.data.get()
    output = np.zeros((2, len(data[0])))
    output[0] = np.array(data.time)
    output[1] = np.array(data.enc1)
    with h5py.File(filepath, 'w') as f:
        r.write(output)


def SRXFly(xstart=None,xstepsize=None,xpts=None,dwell=None,
           ystart=None,ystepsize=None,ypts=None,xs=xs,ion=current_preamp, md=None):
    """

    Monitor IO.
    Zebra buffers x(t) points as a flyer.
    Xpress3 is our detector.
    The aerotech has the x and y positioners.
    """
    delta = 0.01
    rows = np.linspace(ystart,ystart+(ypts-1)*ystepsize,ypts)

    if md is None:
        md = {}
    md = ChainMap(md, {
        'detectors': [zebra,xs],
        'x_range' : xstepsize*xpts,
        'dwell' : dwell,
        'y_range' : ystepzie*ypts,
        }
    )
    try:
        xs.external_trig.put(True)
        #set ion chamber to windowing mode
        yield from set_abs(hf_stage.x, xstart-delta, wait=True)
        yield from open_run(md)
        for n in rows:
            #xspress3 scan-specific set up
            xs.hdf5.capture = xpts
            #flyer = SRXFlyer(encoder, detectors, motor, start, incr, dwell, Npts=1000)
            flyer = SRXFlyer(zebra, hf_stage.x, xstart, xstepsize, dwell, delta, xpts)
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
        xs.external_trig.put(False)


from bluesky.plans import (one_1d_step, kickoff, collect, complete, scan
                           monitor_during_wrapper, abs_set)


def scan_and_fly(xstart, xstop, xnum, ystart, ystop, ynum, dwell, *,
                 delta=None,
                 xmotor=hf_stage.x, ymotor=hf_stage.y,
                 xs=xs, ion=current_preamp,
                 flying_zebra=flying_zebra, md=None):
    """

    Monitor IO.
    Zebra buffers x(t) points as a flyer.
    Xpress3 is our detector.
    The aerotech has the x and y positioners.
    """
    if md is None:
        md = {}
    if delta is None:
        delta=0.01
    md = ChainMap(md, {
        'detectors': [zebra,xs],
        'x_range' : xstepsize*xpts,
        'dwell' : dwell,
        'y_range' : ystepzie*ypts,
        }
    )

    def fly_each_step(detectors, motor, step):
        # First, let 'scan' handle the normal y step, including a checkpoint.
        yield from one_1d_step(detectors, motor, step)

        # Now do the x steps.
        yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move
        v = (xstop - xstart) / xnum / dwell  # compute "stage speed"
        yield from abs_set(xmotor.velocity, v)  # set the "stage speed"
        # arm the Zebra (start caching x positions)
        yield from kickoff(flying_zebra, xstart, xstop, xnum)
        yield from abs_set(xmotor, stop, wait=True)  # move in x
        yield from complete(flying_zebra)  # tell the Zebra we are done
        yield from collect(flying_zebra)  # extract data from Zebra

    @monitor_during_decorator([ion, xs])  # monitor values from ion and acquisitions from xs
    @stage_decorator([xmotor, flying_zebra, xs, ion])  # Below, 'scan' stage ymotor.
    def plan():
        return (yield from scan([], ymotor, ystart, ystop, ynum, per_step=fly_each_step, md=md))
