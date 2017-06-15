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
import numpy
import copy
from bluesky.plans import (one_1d_step, kickoff, collect, complete, scan, wait,
                           monitor_during_wrapper, stage_decorator, abs_set,
                           run_decorator)
from bluesky.examples import NullStatus
import filestore.commands as fs
from bluesky.callbacks import LiveTable, LivePlot, CallbackBase, LiveRaster
from ophyd import Device
import uuid
import h5py
from collections import ChainMap

from hxntools.handlers import register
register()
from hxntools.detectors.xspress3 import Xspress3FileStore

class SRXFlyer1Axis(Device):
    LARGE_FILE_DIRECTORY_READ_PATH = '/tmp/test_data'
    LARGE_FILE_DIRECTORY_WRITE_PATH = '/tmp/test_data'
    "This is the Zebra."
    def __init__(self, encoder, xs, sclr1, *, fs=fs):
        super().__init__('', parent=None)
        self._mode = 'idle'
        self._encoder = encoder
        self._det = xs
        self._sis = sclr1
        self._filestore_resource = None

        # gating info for encoder capture
        self.stage_sigs[self._encoder.pc.gate_num] = 1
        self.stage_sigs[self._encoder.pc.pulse_start] = 0

        #pc gate output is 31 for zebra.  use it to trigger xspress3 and I0
        self.stage_sigs[self._encoder.output1.ttl.addr] = 31
        self.stage_sigs[self._encoder.output3.ttl.addr] = 31

        self.stage_sigs[self._encoder.pc.enc_pos1_sync] = 1

        #put SIS3820 into single count (not autocount) mode
        self.stage_sigs[self._sis.count_mode] = 0

        #stop the SIS3820
        self._sis.stop_all.put(1)

        self._encoder.pc.block_state_reset.put(1)


    def stage(self):
        super().stage()

    def describe_collect(self):

        if self._filestore_resource is not None:
            ext_spec = 'FileStore::{!s}'.format(self._filestore_resource['id'])
        else:
            ext_spec = 'FileStore:'

        spec = {'external': ext_spec,
            'dtype' : 'array',
            'shape' : [self._npts],
            'source': ''  # make this the PV of the array the det is writing
        }

        desc = OrderedDict()
        for chan in ('time','enc1'):
            desc[chan] = spec
            desc[chan]['source'] = getattr(self._encoder.pc.data, chan).pvname
        desc['fluor'] = spec
        desc['fluor']['source'] = 'FileStore::{!s}'.format(self._det.hdf5._filestore_res['id'])
        desc['i0'] = spec
        desc['i0']['source'] = self._sis.mca1.pvname

        return {'stream0':desc}

    def kickoff(self, *, xstart, xstop, xnum, dwell):
        self._mode = 'kicked off'
        self._npts = int(xnum)
        extent = xstop - xstart
        self._encoder.pc.gate_start.put(xstart)
        self._encoder.pc.gate_step.put(extent)
        self._encoder.pc.gate_width.put(extent)
        self._encoder.pc.pulse_max.put(xnum)
        self._encoder.pc.pulse_step.put(dwell)
        self._encoder.pc.pulse_width.put(dwell - 0.005)
        self._encoder.pc.arm.put(1)
        st = NullStatus()  # TODO Return a status object *first* and do the above asynchronously.
        return st

    def complete(self):
        """
        Call this when all needed data has been collected. This has no idea
        whether that is true, so it will obligingly stop immediately. It is
        up to the caller to ensure that the motion is actually complete.
        """
        # Our acquisition complete PV is : XF:05IDD-ES:1{Dev:Zebra1}:ARRAY_ACQ
        while self._encoder.pc.data_in_progress is 1:
            poll()
        self._mode = 'complete'
        # self._encoder.pc.arm.put(0)  # sanity check; this should happen automatically
        # this does the same as the above, but also aborts data collection
        self._encoder.pc.block_state_reset.put(1)
        return NullStatus()

    def collect(self):
        # Create records in the FileStore database.
        # move this to stage because I thinkt hat describe_collect needs the
        # resource id
        self.__filename = '{}.h5'.format(uuid.uuid4())
        self.__filename_sis = '{}.h5'.format(uuid.uuid4())
        self.__read_filepath = os.path.join(self.LARGE_FILE_DIRECTORY_READ_PATH, self.__filename)
        self.__read_filepath_sis = os.path.join(self.LARGE_FILE_DIRECTORY_READ_PATH, self.__filename_sis)
        self.__write_filepath = os.path.join(self.LARGE_FILE_DIRECTORY_WRITE_PATH, self.__filename)
        self.__write_filepath_sis = os.path.join(self.LARGE_FILE_DIRECTORY_WRITE_PATH, self.__filename_sis)
        
        self.__filestore_resource = fs.insert_resource('ZEBRA_HDF51', self.__read_filepath, root='/')
        self.__filestore_resource_sis = fs.insert_resource('SIS_HDF51', self.__read_filepath_sis, root='/')
        
        time_datum_id = str(uuid.uuid4())
        enc1_datum_id = str(uuid.uuid4())
        xs_datum_id = str(uuid.uuid4())
        sis_datum_id = str(uuid.uuid4())
        fs.insert_datum(self.__filestore_resource, time_datum_id, {'column': 'time'})
        fs.insert_datum(self.__filestore_resource, enc1_datum_id, {'column': 'enc1'})
        fs.insert_datum(self.__filestore_resource_sis, sis_datum_id, {'column': 'i0'})
        
        fs.insert_datum(self._det.hdf5._filestore_res, xs_datum_id, {})

        # Write the file.
        export_zebra_data(self._encoder, self.__write_filepath)
        export_sis_data(self._sis, self.__write_filepath_sis)

        # Yield a (partial) Event document. The RunEngine will put this
        # into metadatastore, as it does all readings.
        yield {'time': time.time(), 'seq_num': 1,
               'data': {'time': time_datum_id,
#                        'enc1': enc1_datum_id},
                        'enc1': enc1_datum_id,
                        'fluor' : xs_datum_id,
                        'i0': sis_datum_id},
               'timestamps': {'time': time_datum_id,  # not a typo
#                              'enc1': time_datum_id}}
                              'enc1': time_datum_id,
                              'fluor' : time_datum_id,
                              'i0': time_datum_id}}
        self._mode = 'idle'

    def stop(self):
        pass

    def pause(self):
        "Pausing in the middle of a kickoff nukes the partial dataset."
        self._encoder.pc.arm.put(0)
        self._sis.stop_all.put(1)
        self._mode = 'idle'
        self.unstage()

    def resume(self):
        self.stage()


flying_zebra = SRXFlyer1Axis(zebra,xs,sclr1)
#flying_zebra = SRXFlyer1Axis(zebra)


def export_zebra_data(zebra, filepath):
    data = zebra.pc.data.get()
    size = (len(data.time),)
    with h5py.File(filepath, 'w') as f:
        dset0 = f.create_dataset("time",size,dtype='f')
        dset0[...] = np.array(data.time)
        dset1 = f.create_dataset("enc1",size,dtype='f')
        dset1[...] = np.array(data.enc1)
        f.close()

def export_sis_data(ion,filepath):
    t = ion.mca1.get()
    i = ion.mca2.get()
    size = (len(t),)
    with h5py.File(filepath, 'w') as f:
        dset0 = f.create_dataset("time",size,dtype='f')
        dset0[...] = np.array(t)
        dset1 = f.create_dataset("i0",size,dtype='f')
        dset1[...] = np.array(i)
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


db.fs.register_handler('SIS_HDF51', SISHDF5Handler, overwrite=True)
db.fs.register_handler('ZEBRA_HDF51', ZebraHDF5Handler, overwrite=True)


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


def scan_and_fly(xstart, xstop, xnum, ystart, ystop, ynum, dwell, *,
                 delta=None,
                 xmotor=hf_stage.x, ymotor=hf_stage.y,
                 xs=xs, ion=sclr1,
                 flying_zebra=flying_zebra, md=None):
    """

    Read IO from SIS3820.
    Zebra buffers x(t) points as a flyer.
    Xpress3 is our detector.
    The aerotech has the x and y positioners.
    """
    if md is None:
        md = {}
    if delta is None:
        delta=0.01
    md = ChainMap(md, {
        'plan_name': 'scan_and_fly',
        'detectors': [zebra.name,xs.name,ion.name],
        'dwell' : dwell,
        'shape' : (xnum,ynum)
        }
    )

    
    from bluesky.plans import stage, unstage
    @stage_decorator([xs])
    def fly_each_step(detectors, motor, step):
        "See http://nsls-ii.github.io/bluesky/plans.html#the-per-step-hook"
        # First, let 'scan' handle the normal y step, including a checkpoint.
        yield from one_1d_step(detectors, motor, step)

        # Now do the x steps.
        yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move
        v = (xstop - xstart) / xnum / dwell  # compute "stage speed"
        yield from abs_set(xmotor.velocity, v)  # set the "stage speed"
        yield from abs_set(xs.hdf5.num_capture, xnum)
        yield from abs_set(xs.settings.num_images, xnum)
        yield from abs_set(ion.nuse_all,xnum)
        # arm the Zebra (start caching x positions)
        yield from kickoff(flying_zebra, xstart=xstart, xstop=xstop, xnum=xnum, dwell=dwell, wait=True)
        yield from abs_set(xs.settings.acquire, 1)  # start acquiring images
        yield from abs_set(ion.erase_start, 1) # arm SIS3820, note that there is a 1 sec delay in setting X into motion 
                                               # so the first point *in each row* won't normalize...
#        xs.trigger()
        yield from abs_set(xmotor, xstop+delta, wait=True)  # move in x
        yield from abs_set(xs.settings.acquire, 0)  # stop acquiring images
        yield from abs_set(ion.stop_all, 1)  # stop acquiring scaler
        yield from complete(flying_zebra)  # tell the Zebra we are done
        yield from collect(flying_zebra)  # extract data from Zebra
        yield from abs_set(xmotor.velocity, 3.)  # set the "stage speed"

    #@subs_decorator([LiveTable([ymotor]), RowBasedLiveRaster((ynum, xnum), ion.name, row_key=ymotor.name), LiveZebraPlot()])
    #@subs_decorator([LiveTable([ymotor]), LiveRaster((ynum, xnum), sclr1.mca1.name)])
    @subs_decorator([LiveTable([ymotor])])
    @subs_decorator([LiveRaster((ynum, xnum+1), xs.channel1.rois.roi01.value.name)])
    #@monitor_during_decorator([ion], run=False)  # monitor values from ion
    @monitor_during_decorator([xs.channel1.rois.roi01.value])  # monitor values from xs
    #@monitor_during_decorator([xs], run=False)  # monitor values from xs
    @stage_decorator([flying_zebra])  # Below, 'scan' stage ymotor.
    @run_decorator(md=md)
    def plan():
        #yield from abs_set(xs.settings.trigger_mode, 'TTL Veto Only')
        yield from abs_set(xs.external_trig, True)
        for step in np.linspace(ystart, ystop, ynum):
            # 'arm' the xs for outputting fly data
            yield from abs_set(xs.hdf5.fly_next, True)

            yield from fly_each_step([], ymotor, step)
        #yield from abs_set(xs.settings.trigger_mode, 'Internal')
        yield from abs_set(xs.external_trig, False)
        yield from abs_set(ion.count_mode, 1)


    return (yield from plan())


class RowBasedLiveRaster(LiveRaster):
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
