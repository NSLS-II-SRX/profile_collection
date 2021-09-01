print(f'Loading {__file__}...')

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


import os
import uuid
import h5py
import numpy as np
import time as ttime
import matplotlib.pyplot as plt
from collections import ChainMap

from ophyd import Device
from ophyd.sim import NullStatus
from ophyd.areadetector.filestore_mixins import resource_factory

from bluesky.preprocessors import (stage_decorator,
                                   run_decorator, subs_decorator,
                                   monitor_during_decorator)
import bluesky.plan_stubs as bps
from bluesky.plan_stubs import (one_1d_step, kickoff, collect,
                                complete, abs_set, mv, checkpoint)
from bluesky.plans import (scan, )
from bluesky.callbacks import CallbackBase, LiveGrid

from hxntools.handlers import register

# Note: commenting the following line out due to the error during 2020-2
# deployment:
#   DuplicateHandler: There is already a handler registered for the spec 'XSP3'.
#   Use overwrite=True to deregister the original.
#   Original: <class 'area_detector_handlers._xspress3.Xspress3HDF5Handler'>
#   New: <class 'databroker.assets.handlers.Xspress3HDF5Handler'>
#
# register(db)


# Define wrapper to time a function
def timer_wrapper(func):
    def wrapper(*args, **kwargs):
        t0 = ttime.monotonic()
        yield from func(*args, **kwargs)
        dt = ttime.monotonic() - t0
        print('%s: dt = %f' % (func.__name__, dt))
    return wrapper


def tic():
    return ttime.monotonic()


def toc(t0, str=''):
    dt = ttime.monotonic() - t0
    print('%s: dt = %f' % (str, dt))


# changed the flyer device to be aware of fast vs slow axis in a 2D scan
# should abstract this method to use fast and slow axes, rather than x and y
def scan_and_fly_base(detectors, xstart, xstop, xnum, ystart, ystop, ynum, dwell, *,
                      flying_zebra, xmotor, ymotor,
                      delta=None, shutter=True, align=False, plot=True,
                      md=None, snake=False, verbose=False):
    """Read IO from SIS3820.
    Zebra buffers x(t) points as a flyer.
    Xpress3 is our detector.
    The aerotech has the x and y positioners.
    delta should be chosen so that it takes about 0.5 sec to reach the gate??
    ymotor  slow axis
    xmotor  fast axis

    Parameters
    ----------
    Detectors : List[Device]
       These detectors must be known to the zebra

    xstart, xstop : float
    xnum : int
    ystart, ystop : float
    ynum : int
    dwell : float
       Dwell time in seconds

    flying_zebra : SRXFlyer1Axis

    xmotor, ymotor : EpicsMotor, kwarg only
        These should be known to the zebra
        # TODO sort out how to check this

    delta : float, optional, kwarg only
       offset on the ystage start position.  If not given, derive from
       dwell + pixel size
    align : bool, optional, kwarg only
       If True, try to align the beamline
    shutter : bool, optional, kwarg only
       If True, try to open the shutter
    """

    # t_setup = tic()

    # Check for negative number of points
    if (xnum < 1 or ynum < 1):
        print('Error: Number of points is negative.')
        return

    # Set metadata
    if md is None:
        md = {}
    md = get_stock_md(md)

    # Assign detectors to flying_zebra, this may fail
    flying_zebra.detectors = detectors
    # Setup detectors, combine the zebra, sclr, and the just set detector list
    detectors = (flying_zebra.encoder, flying_zebra.sclr) + flying_zebra.detectors

    dets_by_name = {d.name : d
                    for d in detectors}

    # Set up the merlin
    if 'merlin' in dets_by_name:
        dpc = dets_by_name['merlin']
        # TODO use stage sigs
        # Set trigger mode
        # dpc.cam.trigger_mode.put(2)
        # Make sure we respect whatever the exposure time is set to
        if (dwell < 0.0066392):
            print('The Merlin should not operate faster than 7 ms.')
            print('Changing the scan dwell time to 7 ms.')
            dwell = 0.007
        # According to Ken's comments in hxntools, this is a de-bounce time
        # when in external trigger mode
        dpc.cam.stage_sigs['acquire_time'] = 0.50 * dwell - 0.0016392
        dpc.cam.stage_sigs['acquire_period'] = 0.75 * dwell
        dpc.cam.stage_sigs['num_images'] = 1
        dpc.stage_sigs['total_points'] = xnum
        dpc.hdf5.stage_sigs['num_capture'] = xnum
        del dpc

    # Setup dexela
    if ('dexela' in dets_by_name):
        xrd = dets_by_name['dexela']
        xrd.cam.stage_sigs['acquire_time'] = 0.50 * dwell - 0.050
        xrd.cam.stage_sigs['acquire_period'] = 0.50 * dwell - 0.020
        del xrd

    # If delta is None, set delta based on time for acceleration
    if (delta is None):
        MIN_DELTA = 0.100  # old default value
        v = ((xstop - xstart) / (xnum - 1)) / dwell  # compute "stage speed"
        t_acc = xmotor.acceleration.get()  # acceleration time
        delta = 0.5 * t_acc * v  # distance the stage will travel in t_acc
        delta = np.amax((delta, MIN_DELTA))
        # delta = 0.500 #was 2.5 when npoint scanner drifted

    # Move to start scanning location
    # Calculate move to scan start
    pxsize = (xstop - xstart) / (xnum - 1)
    row_start = xstart - delta - (pxsize / 2)
    row_stop = xstop + delta + (pxsize / 2)
    yield from mv(xmotor, row_start,
                  ymotor, ystart)

    # Run a peakup before the map?
    if (align):
        yield from peakup_fine(shutter=shutter)

    # Scan metadata
    md['scan']['type'] = 'XRF_FLY'
    md['scan']['scan_input'] = [xstart, xstop, xnum, ystart, ystop, ynum, dwell]
    md['scan']['sample_name'] = ''
    md['scan']['detectors'] = [d.name for d in detectors]
    md['scan']['dwell'] = dwell
    md['scan']['fast_axis'] = {'motor_name' : xmotor.name,
                               'units' : xmotor.motor_egu.get()}
    md['scan']['slow_axis'] = {'motor_name' : ymotor.name,
                               'units' : ymotor.motor_egu.get()}
    md['scan']['theta'] = {'val' : nano_stage.th.user_readback.get(),
                           'units' : nano_stage.th.motor_egu.get()}
    md['scan']['delta'] = {'val' : delta,
                           'units' : xmotor.motor_egu.get()}
    md['scan']['snake'] = snake
    md['scan']['shape'] = (xnum, ynum)
    

    @stage_decorator(flying_zebra.detectors)
    def fly_each_step(motor, step, row_start, row_stop):
        def move_to_start_fly():
            "See http://nsls-ii.github.io/bluesky/plans.html#the-per-step-hook"
            yield from abs_set(xmotor, row_start, group='row')
            yield from one_1d_step([temp_nanoKB], motor, step)
            yield from bps.wait(group='row')

        if verbose:
            t_mvstartfly = tic()
        yield from move_to_start_fly()

        # TODO  Why are we re-trying the move?  This should be fixed at
        # a lower level
        # yield from bps.sleep(1.0)  # wait for the "x motor" to move
        x_set = row_start
        x_dial = xmotor.user_readback.get()
        # Get retry deadband value and check against that
        i = 0
        DEADBAND = 0.050  # retry deadband of nPoint scanner
        while (np.abs(x_set - x_dial) > DEADBAND):
            if (i == 0):
                print('Waiting for motor to reach starting position...',
                      end='', flush=True)
            i = i + 1
            yield from mv(xmotor, row_start)
            yield from bps.sleep(0.1)
            x_dial = xmotor.user_readback.get()
        if (i != 0):
            print('done')

        if verbose:
            toc(t_mvstartfly, str='Move to start fly each')

        # Set the scan speed
        # Is abs_set(wait=True) or mv() faster?
        v = ((xstop - xstart) / (xnum - 1)) / dwell  # compute "stage speed"
        # yield from abs_set(xmotor.velocity, v, wait=True)  # set the "stage speed"
        if (v > xmotor.velocity.high_limit):
            raise ValueError(f'Desired motor velocity too high\nMax velocity: {xmotor.velocity.high_limit}')
        elif (v < xmotor.velocity.low_limit):
            raise ValueError(f'Desired motor velocity too low\nMin velocity: {xmotor.velocity.low_limit}')
        else:
            yield from mv(xmotor.velocity, v)

        # set up all of the detectors
        # TODO we should be able to move this out of the per-line call?!
        if ('xs' in dets_by_name):
            xs = dets_by_name['xs']
            yield from abs_set(xs.hdf5.num_capture, xnum, group='set')
            yield from abs_set(xs.settings.num_images, xnum, group='set')
            yield from bps.wait(group='set')
            # yield from mv(xs.hdf5.num_capture, xnum,
            #               xs.settings.num_images, xnum)
            # xs.hdf5.num_capture.put(xnum)
            # xs.settings.num_images.put(xnum)

        if ('xs2' in dets_by_name):
            xs2 = dets_by_name['xs2']
            # yield from abs_set(xs2.hdf5.num_capture, xnum, wait=True)
            # yield from abs_set(xs2.settings.num_images, xnum, wait=True)
            yield from mv(xs2.hdf5.num_capture, xnum,
                          xs2.settings.num_images, xnum)

        if ('merlin' in dets_by_name):
            merlin = dets_by_name['merlin']
            yield from abs_set(merlin.hdf5.num_capture, xnum, wait=True)
            yield from abs_set(merlin.cam.num_images, xnum, wait=True)

        if ('dexela' in dets_by_name):
            dexela = dets_by_name['dexela']
            yield from abs_set(dexela.hdf5.num_capture, xnum, wait=True)
            yield from abs_set(dexela.cam.num_images, xnum, wait=True)

        ion = flying_zebra.sclr
        yield from abs_set(ion.nuse_all, 2*xnum)

        # arm the Zebra (start caching x positions)
        # @timer_wrapper
        def zebra_kickoff():
            if row_start < row_stop:
                yield from kickoff(flying_zebra,
                                   xstart=xstart, xstop=xstop, xnum=xnum, dwell=dwell,
                                   wait=True)
            else:
                yield from kickoff(flying_zebra,
                                   xstart=xstop, xstop=xstart, xnum=xnum, dwell=dwell,
                                   wait=True)
        if verbose:
            t_zebkickoff = tic()
        yield from zebra_kickoff()
        if verbose:
            toc(t_zebkickoff, str='Zebra kickoff')

        if verbose:
            t_datacollect = tic()
        # arm SIS3820, note that there is a 1 sec delay in setting X
        # into motion so the first point *in each row* won't
        # normalize...
        yield from abs_set(ion.erase_start, 1)
        if verbose:
            toc(t_datacollect, str='  reset scaler')

        # trigger all of the detectors
        if verbose:
            print('Data collection:')
        for d in flying_zebra.detectors:
            if verbose:
                print(f'  triggering {d.name}')
            yield from bps.trigger(d, group='row')
            if (d.name == 'dexela'):
                yield from bps.sleep(1)
        if verbose:
            toc(t_datacollect, str='  trigger detectors')

        yield from bps.sleep(1.5)
        if verbose:
            toc(t_datacollect, str='  sleep')

        # start the 'fly'
        yield from abs_set(xmotor, row_stop, group='row')  # move in x
        if verbose:
            toc(t_datacollect, str='  move start')

        if verbose:
            ttime.sleep(1)
            while (xmotor.motor_is_moving.get()):
                ttime.sleep(0.001)
            toc(t_datacollect, str='  move end')
            while (xs.settings.detector_state.get()):
                ttime.sleep(0.001)
            toc(t_datacollect, str='  xs done')
            while (sclr1.acquiring.get()):
                ttime.sleep(0.001)
            toc(t_datacollect, str='  sclr1 done')
        # wait for the motor and detectors to all agree they are done
        yield from bps.wait(group='row')
        if verbose:
            toc(t_datacollect, str='Total time')

        # we still know about ion from above
        yield from abs_set(ion.stop_all, 1)  # stop acquiring scaler

        # @timer_wrapper
        def zebra_complete():
            yield from complete(flying_zebra)  # tell the Zebra we are done
        if verbose:
            t_zebcomplete = tic()
        yield from zebra_complete()
        if verbose:
            toc(t_zebcomplete, str='Zebra complete')


        # @timer_wrapper
        def zebra_collect():
            yield from collect(flying_zebra)  # extract data from Zebra
        if verbose:
            t_zebcollect = tic()
        yield from zebra_collect()
        if verbose:
            toc(t_zebcollect, str='Zebra collect')

    def at_scan(name, doc):
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        scanrecord.current_type.put(md['scan']['type'])
        scanrecord.scanning.put(True)
        scanrecord.time_remaining.put((dwell*xnum + 3.8)/3600)

    def finalize_scan(name, doc):
        logscan_detailed('XRF_FLY')
        scanrecord.scanning.put(False)
        scanrecord.time_remaining.put(0)

    # TODO remove this eventually?
    # xs = dets_by_name['xs']
    # xs = dets_by_name['xs2']
    # Not sure if this is always true
    xs = dets_by_name[flying_zebra.detectors[0].name]

    yield from mv(xs.erase, 0)

    # Setup LivePlot
    if plot:
        if (ynum == 1):
            livepopup = [SRX1DFlyerPlot(xs.channel1.rois.roi01.value.name,
                                        xstart=xstart,
                                        xstep=(xstop-xstart)/(xnum-1),
                                        xlabel=xmotor.name)]
        else:
            livepopup = [LiveGrid((ynum, xnum+1),
                                  xs.channel1.rois.roi01.value.name,
                                  extent=(xstart, xstop, ystart, ystop),
                                  x_positive='right', y_positive='down')]
    else:
        livepopup = []
    @subs_decorator(livepopup)
    @subs_decorator({'start': at_scan})
    @subs_decorator({'stop': finalize_scan})
    # monitor values from xs
    # @monitor_during_decorator([xs.channel1.rois.roi01.value])
    @monitor_during_decorator([xs.channel1.rois.roi01.value, xs.array_counter])
    @stage_decorator([flying_zebra])  # Below, 'scan' stage ymotor.
    @run_decorator(md=md)
    def plan():
        # TODO move this to stage sigs
        for d in flying_zebra.detectors:
            yield from bps.mov(d.total_points, xnum)

        # TODO move this to stage sigs
        yield from bps.mov(xs.external_trig, True)

        ystep = 0
        for step in np.linspace(ystart, ystop, ynum):
            yield from abs_set(scanrecord.time_remaining,
                               (ynum - ystep) * ( dwell * xnum + 3.8 ) / 3600.)
            # 'arm' the all of the detectors for outputting fly data
            for d in flying_zebra.detectors:
                yield from bps.mov(d.fly_next, True)
            # print('h5 armed\t',time.time())
            if (snake is False):
                direction = 0
                start = row_start
                stop = row_stop
            else:
                if ystep % 2 == 0:
                    direction = 0
                    start = row_start
                    stop = row_stop
                else:
                    direction = 1
                    start = row_stop
                    stop = row_start
            # Do work
            if verbose:
                print(f'Direction = {direction}')
                print(f'Start = {start}')
                print(f'Stop  = {stop}')
            flying_zebra._encoder.pc.dir.set(direction)
            yield from fly_each_step(ymotor, step, start, stop)
            # print('return from step\t',time.time())
            ystep = ystep + 1

        # TODO this should be taken care of by stage sigs
        ion = flying_zebra.sclr
        yield from bps.mov(xs.external_trig, False,
                           ion.count_mode, 1)
    # toc(t_setup, str='Setup time')

    # Setup the final scan plan
    if shutter:
        final_plan = finalize_wrapper(plan(),
                                      check_shutters(shutter, 'Close'))
    else:
        final_plan = plan()

    # Open the shutter
    if verbose:
        t_open = tic()
    yield from check_shutters(shutter, 'Open')
    if verbose:
        toc(t_open, str='Open shutter')

    # Run the scan
    uid = yield from final_plan

    return uid

def nano_scan_and_fly(*args, extra_dets=None, **kwargs):
    kwargs.setdefault('xmotor', nano_stage.sx)
    kwargs.setdefault('ymotor', nano_stage.sy)
    kwargs.setdefault('flying_zebra', nano_flying_zebra)
    # print(kwargs['xmotor'].name)
    # print(kwargs['ymotor'].name)
    yield from abs_set(nano_flying_zebra.fast_axis, 'NANOHOR')
    yield from abs_set(nano_flying_zebra.slow_axis, 'NANOVER')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)
    print('Scan finished. Centering the scanner...')
    yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)


def nano_y_scan_and_fly(*args, extra_dets=None, **kwargs):
    kwargs.setdefault('xmotor', nano_stage.sy)
    kwargs.setdefault('ymotor', nano_stage.sx)
    kwargs.setdefault('flying_zebra', nano_flying_zebra)
    yield from abs_set(nano_flying_zebra.fast_axis, 'NANOVER')
    yield from abs_set(nano_flying_zebra.slow_axis, 'NANOHOR')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)
    print('Scan finished. Centering the scanner...')
    yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)



def nano_z_scan_and_fly(*args, extra_dets=None, **kwargs):
    kwargs.setdefault('xmotor', nano_stage.sz)
    kwargs.setdefault('ymotor', nano_stage.sx)
    kwargs.setdefault('flying_zebra', nano_flying_zebra)
    yield from abs_set(nano_flying_zebra.fast_axis, 'NANOZ')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)
    print('Scan finished. Centering the scanner...')
    yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)



def scan_and_fly(*args, extra_dets=None, **kwargs):
    kwargs.setdefault('xmotor', hf_stage.x)
    kwargs.setdefault('ymotor', hf_stage.y)
    kwargs.setdefault('flying_zebra', flying_zebra)
    yield from abs_set(flying_zebra.fast_axis, 'HOR')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)


def y_scan_and_fly(*args, extra_dets=None, **kwargs):
    '''
    Convenience wrapper for scanning Y as the fast axis.
    Call scan_and_fly_base, forcing slow and fast axes to be X and Y.
    In this function, the first three scan parameters are for the *fast axis*,
    i.e., the vertical, and the second three for the *slow axis*, horizontal.
    '''

    kwargs.setdefault('xmotor', hf_stage.y)
    kwargs.setdefault('ymotor', hf_stage.x)
    kwargs.setdefault('flying_zebra', flying_zebra)
    yield from abs_set(flying_zebra.fast_axis, 'VER')

    _xs = kwargs.pop('xs', xs)
    if (extra_dets is None):
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)


# This class is not used in this file
class LiveZebraPlot(CallbackBase):
    """
    This is a really dumb approach but it gets the job done. To fix later.
    """

    def __init__(self, ax=None):
        self._uid = None
        self._desc_uid = None
        if (ax is None):
            _, ax = plt.subplots()
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
        events = db.get_events(db[self._uid], stream_name='stream0',
                               fields=['enc1', 'time'], fill=True)
        for event in events:
            if (event['uid'] in event_uids):
                self.ax.plot(event['data']['time'], event['data']['enc1'],
                             label=event['seq_num'])
        self.ax.legend(loc=0, title=self.legend_title)

    def stop(self, doc):
        self._uid = None
        self._desc_uid = None


# This class is not used in this file
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


class ArrayCounterLiveGrid(LiveGrid):
        def __init__(self, *args, array_counter_key, **kwargs):
            super().__init__(*args, **kwargs)
            self._array_counter_key = array_counter_key
            self._previous_roi = 0

        def event(self, doc):
            if self.I in doc['data']:
                self._previous_roi = doc['data'][self.I]
            elif self._array_counter_key in doc['data']:
                doc = doc.copy()
                doc['data'][self.I] = self._previous_roi
            else:
                # how did we get here?
                print(f'did not find {self.I} or {self._array_counter_key}')
            super().event(doc)


###############################################################################################
# Convenience wrappers for different scans
# This can probably move to a "user scans" file
def y_scan_and_fly_xs2(*args, extra_dets=None, **kwargs):
    '''
    Convenience wrapper for scanning Y as the fast axis.
    Call scan_and_fly_base, forcing slow and fast axes to be X and Y.
    In this function, the first three scan parameters are for the *fast axis*,
    i.e., the vertical, and the second three for the *slow axis*, horizontal.

    A copy of flying_zebra_y where the xspress3 mini is chosen to collect data.
    '''

    kwargs.setdefault('xmotor', hf_stage.y)
    kwargs.setdefault('ymotor', hf_stage.x)
    kwargs.setdefault('flying_zebra', flying_zebra_y_xs2)

    _xs = kwargs.pop('xs', xs2)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)


def y_scan_and_fly_xs2_yz(*args, extra_dets=None, **kwargs):
    '''
    convenience wrapper for scanning Y as the fast axis.
    ** This is a variant of y_scan_and_fly_xs2 but with Z and the slow motor (not X) ***
    call scan_and_fly, forcing slow and fast axes to be Z and Y.
    in this function, the first three scan parameters are for the *fast axis*,
    i.e., the vertical, and the second three for the *slow axis*, horizontal.

    A copy of flying_zebra_y where the xspress3 mini is chosen to collect data.
    '''

    kwargs.setdefault('xmotor', hf_stage.y)
    kwargs.setdefault('ymotor', hf_stage.z)
    kwargs.setdefault('flying_zebra', flying_zebra_y_xs2)

    _xs = kwargs.pop('xs', xs2)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)


def scan_and_fly_xs2(*args, extra_dets=None, **kwargs):
    '''
    A copy of flying_zebra where the xspress3 mini is chosen to collect data on the X axis
    '''

    kwargs.setdefault('xmotor', hf_stage.x)
    kwargs.setdefault('ymotor', hf_stage.y)
    kwargs.setdefault('flying_zebra', flying_zebra_x_xs2)

    _xs = kwargs.pop('xs', xs2)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)


def scan_and_fly_xs2_xz(*args, extra_dets=None, **kwargs):
    '''
    A copy of flying_zebra where the xspress3 mini is chosen to collect data on the X axis
    '''

    kwargs.setdefault('xmotor', hf_stage.x)
    kwargs.setdefault('ymotor', hf_stage.z)
    kwargs.setdefault('flying_zebra', flying_zebra_x_xs2)

    _xs = kwargs.pop('xs', xs2)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    yield from scan_and_fly_base(dets, *args, **kwargs)

