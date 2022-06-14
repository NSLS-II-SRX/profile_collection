print(f'Loading {__file__}...')
import datetime
import json
from bluesky.utils import short_uid

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
from ophyd.status import WaitTimeoutError
from ophyd.sim import NullStatus
from ophyd.areadetector.filestore_mixins import resource_factory

from bluesky.preprocessors import (stage_decorator,
                                   run_decorator, subs_decorator,
                                   monitor_during_decorator)
import bluesky.plan_stubs as bps
from bluesky.plan_stubs import (kickoff, collect,
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
def timer_wrapper(func, *args, **kwargs):
    def wrapper():
        t0 = ttime.monotonic()
        yield from func(*args, **kwargs)
        dt = ttime.monotonic() - t0
        print(f'{func.__name__}: dt = {dt:.6f}')
    ret = yield from wrapper()
    return ret


def tic():
    return ttime.monotonic()


def toc(t0, str=''):
    dt = ttime.monotonic() - t0
    print('%s: dt = %f' % (str, dt))


# changed the flyer device to be aware of fast vs slow axis in a 2D scan
# should abstract this method to use fast and slow axes, rather than x and y
def scan_and_fly_base(detectors, xstart, xstop, xnum, ystart, ystop, ynum, dwell, *,
                      flying_zebra, xmotor, ymotor,
                      delta=None, shutter=True, plot=True,
                      md=None, snake=False, verbose=False):
    """Read IO from SIS3820.
    Zebra buffers x(t) points as a flyer.
    Xpress3 is our detector.
    ## CHECK! delta should be chosen so that it takes about 0.5 sec to reach the gate??
    ymotor  slow (stepping) axis
    xmotor  fast (flying) axis

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
    shutter : bool, optional, kwarg only
       If True, try to open the shutter
    """

    # It is not desirable to display plots when the plan is executed by Queue Server.
    if is_re_worker_active():
        plot = False

    t_setup = tic()

    # Check for negative number of points
    if (xnum < 1 or ynum < 1):
        raise ValueError('Number of points must be positive!')

    # Get the scan speed
    v = ((xstop - xstart) / (xnum - 1)) / dwell  # compute "stage speed"
    if (v > xmotor.velocity.high_limit):
        raise ValueError(f'Desired motor velocity too high\n' \
                         f'Max velocity: {xmotor.velocity.high_limit}')
    elif (v < xmotor.velocity.low_limit):
        raise ValueError(f'Desired motor velocity too low\n' \
                         f'Min velocity: {xmotor.velocity.low_limit}')
    else:
        xmotor.stage_sigs[xmotor.velocity] = v

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
        xrd.cam.stage_sigs['acquire_time'] = dwell
        xrd.cam.stage_sigs['acquire_period'] = dwell
        del xrd

    # If delta is None, set delta based on time for acceleration
    if (delta is None):
        MIN_DELTA = 0.100  # default value
        v = ((xstop - xstart) / (xnum - 1)) / dwell  # compute "stage speed"
        t_acc = xmotor.acceleration.get()  # acceleration time
        delta = 0.5 * t_acc * v  # distance the stage will travel in t_acc
        delta = np.amax((delta, MIN_DELTA))

    # Move to start scanning location
    # Calculate move to scan start
    pxsize = (xstop - xstart) / (xnum - 1)
    row_start = xstart - delta - (pxsize / 2)
    row_stop = xstop + delta + (pxsize / 2)

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
            row_str = short_uid('row')
            yield from bps.checkpoint()
            yield from bps.abs_set(xmotor, row_start, group=row_str)
            yield from bps.abs_set(motor, step, group=row_str)
            yield from bps.wait(group=row_str)
            # Beginning of line read
            yield from bps.trigger_and_read([motor, nano_stage_interferometer])

        if verbose:
            yield from timer_wrapper(move_to_start_fly)
        else:
            yield from move_to_start_fly()

        # TODO  Why are we re-trying the move?  This should be fixed at
        # a lower level
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

        if ('xs2' in dets_by_name):
            xs2 = dets_by_name['xs2']
            yield from mv(
                xs2.hdf5.num_capture, xnum,
                xs2.cam.num_images, xnum   # JL changed settings to cam
            )

        if ('merlin' in dets_by_name):
            merlin = dets_by_name['merlin']
            yield from abs_set(merlin.hdf5.num_capture, xnum, wait=True)
            yield from abs_set(merlin.cam.num_images, xnum, wait=True)

        if ('dexela' in dets_by_name):
            dexela = dets_by_name['dexela']
            yield from abs_set(dexela.hdf5.num_capture, xnum, wait=True)
            # yield from abs_set(dexela.hdf5.num_frames_chunks, xnum, wait=True)
            yield from abs_set(dexela.cam.num_images, xnum, wait=True)

        ion = flying_zebra.sclr
        # TODO Can this be done just once per scan instead of each line?
        yield from abs_set(ion.nuse_all, 2*xnum)

        # arm the Zebra (start caching x positions)
        def zebra_kickoff():
            if row_start < row_stop:
                _row_start = xstart
                _row_stop = xstop
            else:
                _row_start = xstop
                _row_stop = xstart

            yield from kickoff(flying_zebra,
                               xstart=_row_start,
                               xstop=_row_stop,
                               xnum=xnum,
                               dwell=dwell,
                               tacc=xmotor.acceleration.get(),
                               wait=True)
        if verbose:
            yield from timer_wrapper(zebra_kickoff)
        else:
            yield from zebra_kickoff()

        # Need this tic for detector timing
        if verbose:
            t_datacollect = tic()

        def fly_scan_reset_scaler():
            yield from abs_set(ion.erase_start, 1)
        if verbose:
            yield from timer_wrapper(fly_scan_reset_scaler)
        else:
            yield from fly_scan_reset_scaler()

        # trigger all of the detectors
        row_str = short_uid('row')
        if verbose:
            print('Data collection:')
        for d in flying_zebra.detectors:
            if verbose:
                print(f'  triggering {d.name}')
            st = yield from bps.trigger(d, group=row_str)
            if verbose:
                st.add_callback(lambda x: toc(t_datacollect, str=f"  status object  {datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S.%f')}"))
            if (d.name == 'dexela'):
                yield from bps.sleep(1)
        # AMK paranoid check
        t0 = ttime.monotonic()
        while (ts_state.get() != 1 or xs.cam.detector_state.get() != 1):
            if verbose:
                print(f"{ttime.ctime(t0)}\tParanoid check was worth it...")
            yield from mov(ts_state, 1)
            # yield from bps.trigger(xs, group=row_str)
            yield from bps.sleep(0.1)
            if (ttime.monotonic() - t0) > 10:
                print('XS Acquire timeout!')
                raise Exception

        # The zebra needs to be armed last for time-based scanning.
        # If it is armed too early, the timing may be off and the xs3 will miss the first point
        yield from mov(flying_zebra._encoder.pc.arm, 1)
        if verbose:
            toc(t_datacollect, str='  trigger detectors')

        # Move from start to end
        if verbose:
            toc(t_datacollect, str='  move start')
        @stage_decorator([xmotor])
        def move_row():
            yield from abs_set(xmotor, row_stop, wait=True)
        if verbose:
            yield from timer_wrapper(move_row)
        else:
            yield from move_row()

        if verbose and True:
            # ttime.sleep(0.1)
            # while (xmotor.motor_is_moving.get()):
            #     ttime.sleep(0.001)
            # toc(t_datacollect, str='  move end')
            while (get_me_the_cam(xs).detector_state.get()):  # switched to get_me_cam
                ttime.sleep(0.001)
            toc(t_datacollect, str='  xs done')
            while (sclr1.acquiring.get()):
                ttime.sleep(0.001)
            toc(t_datacollect, str='  sclr1 done')
        # wait for the motor and detectors to all agree they are done
        try:
            st.wait(timeout=15)
            yield from bps.wait(group=row_str)
        except WaitTimeoutError as e:
            N_xs = get_me_the_cam(xs).array_counter.get()
            if N_xs == 0:
                print("X3X did not receive any pulses!")
            elif N_xs != xnum:
                print("X3X did not receive {xnum} pulses! ({N}/{xnum})")
            else:
                print("Unknown error!")
                print(e)
            # yield from bps.mov(microZebra.pc.arm, 1)
            raise e

        if verbose:
            toc(t_datacollect, str='Total time')

        # we still know about ion from above
        yield from abs_set(ion.stop_all, 1)  # stop acquiring scaler

        def zebra_complete():
            yield from complete(flying_zebra)  # tell the Zebra we are done
        if verbose:
            yield from timer_wrapper(zebra_complete)
        else:
            yield from zebra_complete()


        def zebra_collect():
            yield from collect(flying_zebra)  # extract data from Zebra
        if verbose:
            yield from timer_wrapper(zebra_collect)
        else:
            yield from zebra_collect()

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
    # xs could be xs, xs2, xs4 ...
    xs = dets_by_name[flying_zebra.detectors[0].name]

    yield from mv(get_me_the_cam(xs).erase, 0)  # Changed to use helper function

    # Setup LivePlot
    # Set the ROI pv
    if hasattr(xs, 'channel01'):
        # if ynum == 1:
        if True:
            roi_pv = EpicsSignalRO('XF:05IDD-ES{Xsp:3}:MCA1ROI:1:TSTotal', name=xs.channel01.mcaroi01.roi_name.get())
            ts_start = EpicsSignal('XF:05IDD-ES{Xsp:3}:MCA1ROI:TSControl', name='ts_start')
            ts_N = EpicsSignal('XF:05IDD-ES{Xsp:3}:MCA1ROI:TSNumPoints', name='ts_N')
            ts_state = EpicsSignal('XF:05IDD-ES{Xsp:3}:MCA1ROI:TSAcquiring', name='ts_state')
            ## Erase the TS buffer
            yield from mov(xs.cam.acquire, 'Done')
            # yield from mov(ts_start, 0)  # Start time series collection
            yield from mov(ts_start, 2)  # Stop time series collection
        else:
            roi_pv = xs.channel01.mcaroi01.total_rbv
    else:
        roi_pv = xs.channel1.rois.roi01.value

    if plot:
        if (ynum == 1):
            livepopup = [
                # SRX1DFlyerPlot(
                SRX1DTSFlyerPlot(
                    roi_pv.name,
                    xstart=xstart,
                    xstep=(xstop-xstart)/(xnum-1),
                    xlabel=xmotor.name
                )
            ]
        else:
            livepopup = [
                # LiveGrid(
                TSLiveGrid(
                    (ynum, xnum),
                    roi_pv.name,
                    extent=(xstart, xstop, ystart, ystop),
                    x_positive='right',
                    y_positive='down'
                )
            ]
    else:
        livepopup = []

    @subs_decorator(livepopup)
    @subs_decorator({'start': at_scan})
    @subs_decorator({'stop': finalize_scan})
    @ts_monitor_during_decorator([roi_pv])
    # @monitor_during_decorator([roi_pv])
    @stage_decorator([flying_zebra])  # Below, 'scan' stage ymotor.
    @run_decorator(md=md)
    def plan():
        if verbose:
            toc(t_setup, str='Setup time + into plan()')

        # TODO move this to stage sigs
        for d in flying_zebra.detectors:
            yield from bps.mov(d.total_points, xnum)

        # TODO move this to stage sigs
        yield from bps.mov(xs.external_trig, True)
        if xs2 in flying_zebra.detectors:
            yield from bps.mov(xs2.external_trig, True)

        # TESTING TimeSeries
        # if ynum == 1:
        if True:
            yield from mov(ts_N, xnum)
            # yield from mov(ts_start, 0)

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
            # if verbose:
            #     print(f'Direction = {direction}')
            #     print(f'Start = {start}')
            #     print(f'Stop  = {stop}')
            yield from mov(ts_start, 0)
            flying_zebra._encoder.pc.dir.set(direction)
            if verbose:
                yield from timer_wrapper(fly_each_step, ymotor, step, start, stop)
                print('\n')
            else:
                yield from fly_each_step(ymotor, step, start, stop)
            # print('return from step\t',time.time())
            ystep = ystep + 1

        # TODO this should be taken care of by stage sigs
        ion = flying_zebra.sclr
        yield from bps.mov(xs.external_trig, False,
                           ion.count_mode, 1)
        if xs2 in flying_zebra.detectors:
            yield from bps.mov(xs2.external_trig, False)
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)

    # Setup the final scan plan
    if shutter:
        if verbose:
            final_plan = finalize_wrapper(plan(),
                                          timer_wrapper(check_shutters, shutter, 'Close'))
        else:
            final_plan = finalize_wrapper(plan(),
                                          check_shutters(shutter, 'Close'))
    else:
        final_plan = plan()

    if verbose:
        toc(t_setup, str='Setup time')

    # Open the shutter
    if verbose:
        yield from timer_wrapper(check_shutters, shutter, 'Open')
    else:
        yield from check_shutters(shutter, 'Open')

    # Run the scan
    uid = yield from final_plan

    # Testing
    # if ynum == 1:
    if True:
        yield from mov(ts_start, 2)

    return uid


def nano_scan_and_fly(*args, extra_dets=None, center=True, **kwargs):
    kwargs.setdefault('xmotor', nano_stage.sx)
    kwargs.setdefault('ymotor', nano_stage.sy)
    kwargs.setdefault('flying_zebra', nano_flying_zebra)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR', wait=True)
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets

    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)
    yield from scan_and_fly_base(dets, *args, **kwargs)
    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)


def nano_y_scan_and_fly(*args, extra_dets=None, center=True, **kwargs):
    kwargs.setdefault('xmotor', nano_stage.sy)
    kwargs.setdefault('ymotor', nano_stage.sx)
    kwargs.setdefault('flying_zebra', nano_flying_zebra)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOVER', wait=True)
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOHOR')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets

    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)
    yield from scan_and_fly_base(dets, *args, **kwargs)
    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)



def nano_z_scan_and_fly(*args, extra_dets=None, center=True, **kwargs):
    kwargs.setdefault('xmotor', nano_stage.sz)
    kwargs.setdefault('ymotor', nano_stage.sx)
    kwargs.setdefault('flying_zebra', nano_flying_zebra)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOZ')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets

    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)
    yield from scan_and_fly_base(dets, *args, **kwargs)
    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)


def coarse_scan_and_fly(*args, extra_dets=None, center=True, **kwargs):
    kwargs.setdefault('xmotor', nano_stage.x)
    kwargs.setdefault('ymotor', nano_stage.y)
    kwargs.setdefault('flying_zebra', nano_flying_zebra_coarse)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR')
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets

    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)
    yield from scan_and_fly_base(dets, *args, **kwargs)
    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)


def coarse_y_scan_and_fly(*args, extra_dets=None, center=True, **kwargs):
    '''
    Convenience wrapper for scanning Y as the fast axis.
    Call scan_and_fly_base, forcing slow and fast axes to be X and Y.
    In this function, the first three scan parameters are for the *fast axis*,
    i.e., the vertical, and the second three for the *slow axis*, horizontal.
    '''

    kwargs.setdefault('xmotor', nano_stage.y)
    kwargs.setdefault('ymotor', nano_stage.x)
    kwargs.setdefault('flying_zebra', nano_flying_zebra_coarse)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOVER')
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOHOR')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets

    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)
    yield from scan_and_fly_base(dets, *args, **kwargs)
    if center:
        yield from mv(nano_stage.sx, 0, nano_stage.sy, 0, nano_stage.sz, 0)


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
