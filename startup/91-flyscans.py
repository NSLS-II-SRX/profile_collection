
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
import numpy as np
from bluesky.plans import (scan, )
from bluesky.plan_stubs import (one_1d_step, kickoff, collect, complete,
                                abs_set, mv)

import bluesky.plan_stubs as bps


from bluesky.preprocessors import (stage_decorator,
                                   run_decorator, subs_decorator,
                                   monitor_during_decorator)
from ophyd.sim import NullStatus
from bluesky.callbacks import CallbackBase, LiveGrid
from ophyd import Device
import uuid
import h5py
from collections import ChainMap
from ophyd.areadetector.filestore_mixins import resource_factory

from hxntools.handlers import register
register(db)


class SRXFlyer1Axis(Device):
    LARGE_FILE_DIRECTORY_WRITE_PATH = '/nsls2/xf05id1/XF05ID1/data/2018-1/fly_scan_ancillary/'
#    LARGE_FILE_DIRECTORY_READ_PATH = '/tmp/test_data'
#    LARGE_FILE_DIRECTORY_WRITE_PATH = '/tmp/fly_scan_ancillary'
    LARGE_FILE_DIRECTORY_READ_PATH = '/nsls2/xf05id1/XF05ID1/data/2018-1/fly_scan_ancillary/'
    "This is the Zebra."
    def __init__(self, encoder, xs, sclr1, fast_axis, *, reg=db.reg, **kwargs):
        super().__init__('', parent=None, **kwargs)
        self._mode = 'idle'
        self._encoder = encoder
        self._det = xs
        self._sis = sclr1
        self._filestore_resource = None

        self._fast_axis = fast_axis

        if self._fast_axis == 'HOR':
            self.stage_sigs[self._encoder.pc.enc] = 'Enc2'
            self.stage_sigs[self._encoder.pc.dir] = 'Positive'
            self.stage_sigs[self._encoder.pc.enc_res2] = 5E-6
        elif self._fast_axis == 'VER':
            self.stage_sigs[self._encoder.pc.enc] = 'Enc1'
            self.stage_sigs[self._encoder.pc.dir] = 'Positive'
            self.stage_sigs[self._encoder.pc.enc_res1] = 5E-6

        # gating info for encoder capture
        self.stage_sigs[self._encoder.pc.gate_num] = 1
        self.stage_sigs[self._encoder.pc.pulse_start] = 0

        #pc gate output is 31 for zebra.  use it to trigger xspress3 and I0
        self.stage_sigs[self._encoder.output1.ttl.addr] = 31
        self.stage_sigs[self._encoder.output3.ttl.addr] = 31

        self.stage_sigs[self._encoder.pc.enc_pos1_sync] = 1
        self.stage_sigs[self._encoder.pc.enc_pos2_sync] = 1

        #put SIS3820 into single count (not autocount) mode
        self.stage_sigs[self._sis.count_mode] = 0

        #stop the SIS3820
        self._sis.stop_all.put(1)

        self._encoder.pc.block_state_reset.put(1)
        self.reg = reg
        self._document_cache = []
        self._last_bulk = None

    def stage(self):
        super().stage()

    def describe_collect(self):

        ext_spec = 'FileStore:'

        spec = {'external': ext_spec,
            'dtype' : 'array',
            'shape' : [self._npts],
            'source': ''  # make this the PV of the array the det is writing
        }

        desc = OrderedDict()
        for chan in ('time', 'enc1'):
            desc[chan] = spec
            desc[chan]['source'] = getattr(self._encoder.pc.data, chan).pvname
        desc['fluor'] = spec
        desc['fluor']['source'] = self._det.prefix
        desc['i0'] = spec
        desc['i0']['source'] = self._sis.mca2.pvname
        desc['i0_time'] = spec
        desc['i0_time']['source'] = self._sis.mca1.pvname

        return {'stream0': desc}


    def kickoff(self, *, xstart, xstop, xnum, dwell):
        self._encoder.pc.arm.put(0)
        self._mode = 'kicked off'
        self._npts = int(xnum)
        extent = xstop - xstart
        pxsize = extent / (xnum-1)
        #1 ms delay between pulses
        decrement = ((pxsize / dwell) * 0.002)
        self._encoder.pc.gate_start.put(xstart)
        #self._encoder.pc.gate_step.put(extent+0.01)
        #self._encoder.pc.gate_width.put(extent+0.005)
        self._encoder.pc.gate_step.put(extent+0.0005)
        self._encoder.pc.gate_width.put(extent+0.0001)
        self._encoder.pc.pulse_max.put(xnum)
#        self._encoder.pc.pulse_step.put(dwell)
#        self._encoder.pc.pulse_width.put(dwell - 0.005)
#        self._encoder.pc.pulse_step.put(extent/xnum)
#        self._encoder.pc.pulse_width.put(extent/xnum-decrement)
        self._encoder.pc.pulse_step.put(pxsize)
        self._encoder.pc.pulse_width.put(pxsize-decrement)
        self._encoder.pc.pulse_start.put(0.0)
        #self._encoder.pc.pulse_step.put(dwell)
        #self._encoder.pc.pulse_width.put(dwell-0.001)
        self._encoder.pc.arm.put(1)
        #THIS MUST CHANGE!!!!
        if self._fast_axis == 'VER':
            self._encoder.pc.enc_pos1_sync.put(1)
        elif self._fast_axis == 'HOR':
            self._encoder.pc.enc_pos2_sync.put(1)

        st = NullStatus()  # TODO Return a status object *first* and do the above asynchronously.
        return st

    def complete(self):
        """
        Call this when all needed data has been collected. This has no idea
        whether that is true, so it will obligingly stop immediately. It is
        up to the caller to ensure that the motion is actually complete.
        """
        # Our acquisition complete PV is : XF:05IDD-ES:1{Dev:Zebra1}:ARRAY_ACQ
        while self._encoder.pc.data_in_progress.get() == 1:
            ttime.sleep(.1)
            #poll()
        ttime.sleep(.1)
        self._mode = 'complete'
        # self._encoder.pc.arm.put(0)  # sanity check; this should happen automatically
        # this does the same as the above, but also aborts data collection
        self._encoder.pc.block_state_reset.put(1)
        #see triggering errors of the xspress3 on suspension.  This is to test the reset of the xspress3 after a line.
        self._det.settings.acquire.put(0)

        self.__filename = '{}.h5'.format(uuid.uuid4())
        self.__filename_sis = '{}.h5'.format(uuid.uuid4())
        self.__read_filepath = os.path.join(self.LARGE_FILE_DIRECTORY_READ_PATH,
                                            self.__filename)
        self.__read_filepath_sis = os.path.join(self.LARGE_FILE_DIRECTORY_READ_PATH,
                                                self.__filename_sis)
        self.__write_filepath = os.path.join(self.LARGE_FILE_DIRECTORY_WRITE_PATH,
                                             self.__filename)
        self.__write_filepath_sis = os.path.join(self.LARGE_FILE_DIRECTORY_WRITE_PATH,
                                                 self.__filename_sis)

        self.__filestore_resource, datum_factory_z = resource_factory(
            'ZEBRA_HDF51', root='/',
            resource_path=self.__read_filepath,
            resource_kwargs={}, path_semantics='posix')
        self.__filestore_resource_sis, datum_factory_sis = resource_factory(
            'SIS_HDF51', root='/',
            resource_path=self.__read_filepath_sis,
            resource_kwargs={},
            path_semantics='posix')

        time_datum = datum_factory_z({'column': 'time'})
        enc1_datum = datum_factory_z({'column': 'enc1'})
        sis_datum =  datum_factory_sis({'column': 'i0'})
        sis_time =  datum_factory_sis({'column': 'time'})

        self._document_cache.extend(('resource', d) for d in (self.__filestore_resource,
                                                             self.__filestore_resource_sis))
        self._document_cache.extend(('datum', d) for d in (time_datum, enc1_datum,
                                                          sis_datum, sis_time))
        self._document_cache.extend(self._det.collect_asset_docs())

        # TODO call 'read' on the detector instead
        # xs_datum_id = self.reg.register_datum(self._det.hdf5._filestore_res, {})
        xs_reading = self._det.read()
        # Write the file.
        export_zebra_data(self._encoder, self.__write_filepath,self._fast_axis)
        export_sis_data(self._sis, self.__write_filepath_sis)

        # Yield a (partial) Event document. The RunEngine will put this
        # into metadatastore, as it does all readings.
        self._last_bulk =  {
            'time': time.time(), 'seq_num': 1,
            'data': {'time': time_datum['datum_id'],
                     'enc1': enc1_datum['datum_id'],
                     'fluor': xs_reading['fluor']['value'],
                     'i0': sis_datum['datum_id'],
                     'i0_time': sis_time['datum_id']},
            'timestamps': {'time': time_datum['datum_id'],  # not a typo#
                           'enc1': time_datum['datum_id'],
                           'fluor': xs_reading['fluor']['timestamp'],
                           'i0': sis_time['datum_id'],
                           'i0_time': sis_time['datum_id']}
        }
        return NullStatus()

    def collect(self):
        # Create records in the FileStore database.
        # move this to stage because I thinkt hat describe_collect needs the
        # resource id
        # TODO use ophyd.areadectector.filestoer_mixins.resllource_factory here
        if self._last_bulk is None:
            raise Exception("the order of complete and collect is brittle and out "
                            "of sync. This device relies on in-order and 1:1 calls "
                            "between complete and collect to correctly create and stash "
                            "the asset registry documents")
        yield self._last_bulk
        self._last_bulk = None
        self._mode = 'idle'

    def collect_asset_docs(self):
        yield from iter(list(self._document_cache))
        self._document_cache.clear()

    def stop(self):
        self._encoder.pc.block_state_reset.put(1)
        pass

    def pause(self):
        "Pausing in the middle of a kickoff nukes the partial dataset."
        #self._encoder.pc.arm.put(0)
        self._encoder.pc.block_state_reset.put(1)
        self._sis.stop_all.put(1)
        self._det.settings.acquire.put(0)
        self._mode = 'idle'
        self.unstage()

    def resume(self):
        self.stage()


flying_zebra = SRXFlyer1Axis(zebra, xs, sclr1, 'HOR', name='flying_zebra')
flying_zebra_y = SRXFlyer1Axis(zebra, xs, sclr1, 'VER', name='flying_zebra')
#flying_zebra = SRXFlyer1Axis(zebra)


def export_zebra_data(zebra, filepath, fast_axis):
    j = 0
    while zebra.pc.data_in_progress.get()==1:
        print('waiting zebra')
        ttime.sleep(0.1)
        j += 1
        if j > 10:
            print('THE ZEBRA IS BEHAVING BADLY CARRYING ON')
            break

    #ttime.sleep(.5)
    time_d = zebra.pc.data.time.get()
    if fast_axis == 'HOR':
        enc1_d = zebra.pc.data.enc2.get()
    else:
        enc1_d = zebra.pc.data.enc1.get()

    while len(time_d) == 0 or len(time_d) != len(enc1_d):
        time_d = zebra.pc.data.time.get()
        #enc1_d = zebra.pc.data.enc2.get()
        if fast_axis == 'HOR':
            enc1_d = zebra.pc.data.enc2.get()
        else:
            enc1_d = zebra.pc.data.enc1.get()

    size = (len(time_d),)
    with h5py.File(filepath, 'w') as f:
        dset0 = f.create_dataset("time",size,dtype='f')
        dset0[...] = np.array(time_d)
        dset1 = f.create_dataset("enc1",size,dtype='f')
        dset1[...] = np.array(enc1_d)
        f.close()

def export_sis_data(ion, filepath):
    t = ion.mca1.get(timeout=5.)
    i = ion.mca2.get(timeout=5.)
    while len(t) == 0 and len(t) != len(i):
        t = ion.mca1.get(timeout=5.)
        i = ion.mca2.get(timeout=5.)
    correct_length = zebra.pc.data.num_down.get()
    size = (len(t),)
    size2 = (len(i),)
    with h5py.File(filepath, 'w') as f:
        #dset0 = f.create_dataset("time",size,dtype='f')
        #dset0[...] = np.array(t)
        #dset1 = f.create_dataset("i0",size2,dtype='f')
        #dset1[...] = np.array(i)
        if len(t) != correct_length:
            correction_factor = (correct_length-len(t))
            #print(correction_factor,len(t),correct_length)
            new_t = [k for k in t] + [ 1e10 for _ in range(0,int(correction_factor)) ]
            new_i = [k for k in i] + [ 1e10 for _ in range(0,int(correction_factor)) ]
        else:
            correction_factor = 0
            #print(correction_factor,len(t),correct_length)
            new_t = t
            new_i = i
        dset0 = f.create_dataset("time",(correct_length,),dtype='f')
        dset0[...] = np.array(new_t)
        dset1 = f.create_dataset("i0",(correct_length,),dtype='f')
        dset1[...] = np.array(new_i)
        f.close()

class ZebraHDF5Handler(HandlerBase):
    HANDLER_NAME = 'ZEBRA_HDF51'
    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self, *, column):
        return self._handle[column][:]

class SISHDF5Handler(HandlerBase):
    HANDLER_NAME = 'SIS_HDF51'
    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self, *, column):
        return self._handle[column][:]


db.reg.register_handler('SIS_HDF51', SISHDF5Handler, overwrite=True)
db.reg.register_handler('ZEBRA_HDF51', ZebraHDF5Handler, overwrite=True)


class LiveZebraPlot(CallbackBase):
    """
    This is a really dumb approach but it gets the job done. To fix later.
    """

    def __init__(self, ax=None):
        self._uid = None
        self._desc_uid = None
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.legend_title = 'sequence #'

    def start(self, doc):
        self._uid = doc['uid']

    def descriptor(self, doc):
        if doc['name'] == 'stream0':
            self._desc_uid = doc['uid']

    def bulk_events(self, docs):
        # Don't actually use the docs, but use the fact that they have been
        # emitted as a signal to go grab the data from the databroker now.
        event_uids = [doc['uid'] for doc in docs[self._desc_uid]]
        events = db.get_events(db[self._uid], stream_name='stream0', fields=['enc1', 'time'], fill=True)
        for event in events:
            if event['uid'] in event_uids:
                self.ax.plot(event['data']['time'], event['data']['enc1'], label=event['seq_num'])
        self.ax.legend(loc=0, title=self.legend_title)

    def stop(self, doc):
        self._uid = None
        self._desc_uid = None


# changed the flyer device to be aware of fast vs slow axis in a 2D scan
# should abstract this method to use fast and slow axes, rather than x and y
def scan_and_fly(xstart, xstop, xnum, ystart, ystop, ynum, dwell, *,
                 delta=0.002, shutter=True,
                 xmotor=hf_stage.x, ymotor=hf_stage.y,
                 xs=xs, ion=sclr1, align=False,
                 flying_zebra=flying_zebra, md=None):
    """

    Read IO from SIS3820.
    Zebra buffers x(t) points as a flyer.
    Xpress3 is our detector.
    The aerotech has the x and y positioners.
    delta should be chosen so that it takes about 0.5 sec to reach the gate??
    ymotor  slow axis
    xmotor  fast axis
    """
    c2pitch_kill = EpicsSignal("XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Cmd:Kill-Cmd")
    if md is None:
        md = {}
    if delta is None:
        delta = 0.002
    yield from abs_set(ymotor, ystart, wait=True) # ready to move
    yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move

    if shutter is True:
        yield from mv(shut_b, 'Open')

    if align:
        fly_ps = PeakStats(dcm.c2_pitch.name, i0.name)
        align_scan = scan([sclr1], dcm.c2_pitch, -19.320, -19.360, 41)
        align_scan = bp.subs_wrapper(align_scan, fly_ps)
        yield from align_scan
        yield from abs_set(dcm.c2_pitch, fly_ps.max[0], wait=True)
        #ttime.sleep(10)
        #yield from abs_set(c2pitch_kill, 1)

    md = ChainMap(md, {
        'plan_name': 'scan_and_fly',
        'detectors': [zebra.name, xs.name, ion.name],
        'dwell': dwell,
        'shape': (xnum, ynum),
        'scaninfo': {'type': 'XRF_fly',
                     'raster': False,
                     'fast_axis': flying_zebra._fast_axis},
        'scan_params': [xstart, xstop, xnum, ystart, ystop, ynum, dwell]
        }
    )

    @stage_decorator([xs])
    def fly_each_step(detectors, motor, step, firststep):
        "See http://nsls-ii.github.io/bluesky/plans.html#the-per-step-hook"
        # First, let 'scan' handle the normal y step, including a checkpoint.
        yield from one_1d_step(detectors, motor, step)

        # Now do the x steps.
        v = (xstop - xstart) / (xnum-1) / dwell  # compute "stage speed"
        yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move
        yield from abs_set(xmotor.velocity, v, wait=True)  # set the "stage speed"

        yield from abs_set(xs.hdf5.num_capture, xnum, wait=True)
        yield from abs_set(xs.settings.num_images, xnum, wait=True)
        yield from abs_set(ion.nuse_all,xnum)
        # arm the Zebra (start caching x positions)


        yield from kickoff(flying_zebra, xstart=xstart, xstop=xstop, xnum=xnum, dwell=dwell, wait=True)
        yield from abs_set(ion.erase_start, 1) # arm SIS3820, note that there is a 1 sec delay in setting X into motion
                                               # so the first point *in each row* won't normalize...
        yield from bps.trigger(xs, group='row')
        #if firststep == True:
        #    ttime.sleep(0.)
        yield from bps.sleep(1.5)
        yield from abs_set(xmotor, xstop+1*delta, group='row')  # move in x
        yield from bps.wait(group='row')
        # yield from abs_set(xs.settings.acquire, 0)  # stop acquiring images
        yield from abs_set(ion.stop_all, 1)  # stop acquiring scaler
        yield from complete(flying_zebra)  # tell the Zebra we are done
        yield from collect(flying_zebra)  # extract data from Zebra
        yield from abs_set(xmotor.velocity, 1.0, wait=True)  # set the "stage speed"

    def at_scan(name, doc):
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        scanrecord.current_type.put(md['scaninfo']['type'])
        scanrecord.scanning.put(True)
        scanrecord.time_remaining.put((dwell*xnum + 3.8)/3600)

    def finalize_scan(name, doc):
        logscan_detailed('xrf_fly')
        scanrecord.scanning.put(False)
        scanrecord.time_remaining.put(0)

    #@subs_decorator([LiveTable([ymotor]), RowBasedLiveGrid((ynum, xnum), ion.name, row_key=ymotor.name), LiveZebraPlot()])
    #@subs_decorator([LiveTable([ymotor]), LiveGrid((ynum, xnum), sclr1.mca1.name)])
    @subs_decorator([LiveGrid((ynum, xnum+1),
                              xs.channel1.rois.roi01.value.name,
                              extent=(xstart, xstop, ystop, ystart))])
    @subs_decorator({'start': at_scan})
    @subs_decorator({'stop': finalize_scan})
    # monitor values from xs
    @monitor_during_decorator([xs.channel1.rois.roi01.value])
    #@monitor_during_decorator([xs], run=False)  # monitor values from xs
    @stage_decorator([flying_zebra])  # Below, 'scan' stage ymotor.
    @run_decorator(md=md)
    def plan():
        yield from bps.mov(xs.total_points, xnum)
        # added to "prime" the detector
        #yield from abs_set(xs.settings.trigger_mode, 'TTL Veto Only')

        yield from bs.mov(xs.external_trig, True)
        ystep = 0

        for step in np.linspace(ystart, ystop, ynum):
            yield from abs_set(scanrecord.time_remaining,
                               (ynum - ystep) * ( dwell * xnum + 3.8 ) / 3600.)
            ystep = ystep + 1
            # 'arm' the xs for outputting fly data
            yield from bps.mov(xs.fly_next, True)
#            print('h5 armed\t',time.time())
            if step == ystart:
                firststep = True
            else:
                firststep = False
            yield from fly_each_step([], ymotor, step, firststep)
#            print('return from step\t',time.time())
        yield from bps.mov(xs.external_trig, False,
                          ion.count_mode, 1)
        if shutter is True:
            yield from mv(shut_b, 'Close')

    return (yield from plan())


class RowBasedLiveGrid(LiveGrid):
    """
    Synthesize info from two event stream here.

    Use the event with 'row_key' in it to figure out when we have moved to a new row.
    Figure out if the seq_num has the right value, given the expected raster_shape.
    If seq_num is low, we have missed some values. Pad the seq_num (effectively leaving
    empty tiles at the end of the row) for future events.
    """
    def __init__(self, *args, row_key, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_key = row_key
        self._last_row = None
        self._column_counter = None  # count tiles we have seen in current row
        self._pad = None
        self._desired_columns = self.raster_shape[1]  # number of tiles row should have

    def start(self, doc):
        super().start(doc)
        self._column_counter = 0
        self._last_row = None
        self._pad = 0

    def event(self, doc):
        # If this is an event that tells us what row we are in:
        if self._row_key in doc['data']:
            this_row = doc['data'][self._row_key]
            if self._last_row is None:
                # initialize with the first y value we see
                self._last_row = this_row
                return
            if this_row != self._last_row:
                # new row -- pad future sequence numbers if row fell short
                missing = self._desired_columns - self._column_counter
                self._pad += missing
                self._last_row = this_row
                self._column_counter = 0
        # If this is an event with the data we want to plot:
        if self.I in doc['data']:
            self._column_counter += 1
            doc = doc.copy()
            doc['seq_num'] += self._pad
            super().event(doc)

    def stop(self, doc):
        self._last_row = None
        self._column_counter = None
        self._pad = None


class SrxXSP3Handler:
    XRF_DATA_KEY = 'entry/instrument/detector/data'

    def __init__(self, filepath, **kwargs):
        self._filepath = filepath

    def __call__(self, **kwargs):
        with h5py.File(self._filepath, 'r') as f:
            return np.asarray(f[self.XRF_DATA_KEY])


def batch_fly(paramlist, kwlist=None, zlist=None):
    '''
    paramlist   list    list of positional and dwell time arguments to scan_and_fly
    kwlist      list    list of dicts containing keywords to pass to scan_and_fly
    '''

    if kwlist == None:
        kwlist=list()
        for _ in paramlist:
            kwlist.append({})

    for i in range(0,len(paramlist)):
        #this should be made more general
        if zlist is not None:
            yield from abs_set(hf_stage.z,zlist[i],wait=True)
        yield from scan_and_fly(*paramlist[i],**kwlist[i])


def batch_fly_arb(paramlist, kwlist=None, motlist=None):
    '''
    paramlist   list            list of positional and dwell time arguments to scan_and_fly
    kwlist      list            list of dicts containing keywords to pass to scan_and_fly
    motlist     list of lists   a list of motor,value pairs that define the prestart condition
    '''

    if kwlist is None:
        kwlist = list()
        for _ in paramlist:
            kwlist.append({})

#    for i in range(0,len(paramlist)):
#        if motlist is not None:
#            yield from abs_set(motlist[i][0],motlist[i][1],wait=True)
#        yield from scan_and_fly(*paramlist[i],**kwlist[i])
    for i in range(0, len(paramlist)):
        if motlist is not None:
            for pospair in motlist[i]:
                yield from abs_set(pospair[0], pospair[1], wait=True)
        yield from scan_and_fly(*paramlist[i], **kwlist[i])


def y_scan_and_fly(*args, **kwargs):
    '''
    convenience wrapper for scanning Y as the fast axis.
    call scan_and_fly, forcing slow and fast axes to be X and Y.
    in this function, the first three scan parameters are for the *fast axis*,
    i.e., the vertical, and the second three for the *slow axis*, horizontal.
    '''
    if 'delta' in kwargs.keys():
        if kwargs['delta'] is not None:
            kwargs['delta'] = 0.004
    yield from scan_and_fly(*args, **kwargs,
                            xmotor=hf_stage.y,
                            ymotor=hf_stage.x,
                            flying_zebra=flying_zebra_y)
