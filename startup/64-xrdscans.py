print(f'Loading {__file__}...')

import skimage.io as io
import numpy as np
import time as ttime

from bluesky.plans import count, list_scan
from bluesky.utils import short_uid as _short_uid


def _setup_xrd_dets(dets,
                    dwell,
                    N_images):
    # Convenience function for setting up xrd detectors
    dets_by_name = {d.name : d for d in dets}

    # Setup merlin
    if 'merlin' in dets_by_name:
        xrd = dets_by_name['merlin']
        # Make sure we respect whatever the exposure time is set to
        if (dwell < 0.0066392):
            print('The Merlin should not operate faster than 7 ms.')
            print('Changing the scan dwell time to 7 ms.')
            dwell = 0.007
        # According to Ken's comments in hxntools, this is a de-bounce time
        # when in external trigger mode
        # xrd.cam.stage_sigs['acquire_time'] = 0.75 * dwell  # - 0.0016392
        # xrd.cam.stage_sigs['acquire_period'] = 0.75 * dwell + 0.0016392
        xrd.cam.stage_sigs['acquire_time'] = 0.9 * dwell - 0.002
        xrd.cam.stage_sigs['acquire_period'] = 0.9 * dwell
        xrd.stage_sigs['total_points'] = N_images
        xrd.cam.stage_sigs['num_images'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images
        del xrd

    # Setup dexela
    if 'dexela' in dets_by_name:
        xrd = dets_by_name['dexela']
        # If the dexela is acquiring, stop
        if xrd.cam.detector_state.get() == 1:
            xrd.cam.acquire.set(0)
        xrd.cam.stage_sigs['acquire_time'] = dwell
        xrd.cam.stage_sigs['acquire_period'] = dwell * 0.9
        xrd.cam.stage_sigs['num_images'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images
        del xrd


# Assumes detector stage sigs are already set
def _acquire_dark_fields(dets,
                         N_dark=10,
                         shutter=True):                

    dets_by_name = {d.name : d for d in dets}
    xrd_dets = []
    reset_sigs = []

    if 'merlin' in dets_by_name:
        xrd = dets_by_name['merlin']
        sigs = OrderedDict(
            [   
            (xrd.cam, 'trigger_mode', 0),
            (xrd, 'total_points', 1),
            (xrd.cam, 'num_images', 1),
            (xrd.hdf5, 'num_capture', N_dark)
            ]
        )

        original_sigs = []
        for obj, key, value in sigs:
            if key in obj.stage_sigs:
                original_sigs.append((obj, key, obj.stage_sigs[key]))
            obj.stage_sigs[key] = value
        
        xrd_dets.append(xrd)
        reset_sigs.extend(original_sigs)

    if 'dexela' in dets_by_name:
        xrd = dets_by_name['dexela']
        sigs = [
                (xrd, 'total_points', 1),
                (xrd, 'cam.image_mode', 'Single'),
                (xrd.cam, 'trigger_mode', 'Int. Fixed Rate'),
                (xrd.cam, 'image_mode', 'Single'),
                (xrd.cam, 'num_images', 1),
                (xrd.hdf5, 'num_capture', N_dark),
                ]

        original_sigs = []
        for obj, key, value in sigs:
            if key in obj.stage_sigs:
                original_sigs.append((obj, key, obj.stage_sigs[key]))
            obj.stage_sigs[key] = value
        
        xrd_dets.append(xrd)
        reset_sigs.extend(original_sigs)
    
    if len(xrd_dets) > 0:
        d_status = shut_d.read()['shut_d_request_open']['value'] == 1 # is open
        yield from check_shutters(shutter, 'Close')
        print('Acquiring dark-field...')
        staging_list = [det._staged == Staged.yes for det in xrd_dets]
        for staged, det in zip(staging_list, xrd_dets):
            if staged:
                yield from bps.unstage(det)
            yield from bps.stage(det)
        
        # EJM deeply despises this type of acquisition.
        # Takes 3000% longer than it needs to
        for _ in range(N_dark):
            yield from bps.trigger_and_read(xrd_dets, name='dark')

        # Reset to original stage_sigs    
        for obj, key, value in reset_sigs:
            obj.stage_sigs[key] = value   
        
        for staged, det in zip(staging_list, xrd_dets):
            yield from bps.unstage(det)
            if staged:
                yield from bps.stage(det)

        if d_status:
            yield from check_shutters(shutter, 'Open')


# Assumes detector stage sigs are already set
def _continuous_dark_fields(dets,
                            N_dark=10,
                            shutter=False):                

    dets_by_name = {d.name : d for d in dets}
    xrd_dets = []
    reset_sigs = []

    print(f'{dexela.cam.num_images.get()=}')
    print(f'{dexela.hdf5.capture.get()=}')
    print(f'{dexela.cam.acquire.get()=}')
    print(f'{dexela.cam.detector_state.get()=}')

    if 'merlin' in dets_by_name:
        xrd = dets_by_name['merlin']
        sigs = OrderedDict(
            [   
            (xrd.cam, 'trigger_mode', 0),
            (xrd, 'total_points', 1),
            (xrd.cam, 'num_images', 1),
            (xrd.hdf5, 'num_capture', N_dark)
            ]
        )

        original_sigs = []
        for obj, key, value in sigs:
            if key in obj.stage_sigs:
                original_sigs.append((obj, key, obj.stage_sigs[key]))
            obj.stage_sigs[key] = value
        
        xrd_dets.append(xrd)
        reset_sigs.extend(original_sigs)

    if 'dexela' in dets_by_name:
        xrd = dets_by_name['dexela']
        sigs = [
                (xrd, 'total_points', N_dark),
                (xrd, 'cam.image_mode', 'Multiple'),
                (xrd.cam, 'trigger_mode', 'Int. Fixed Rate'),
                (xrd.cam, 'image_mode', 'Multiple'), # redundant, but already in staging...
                (xrd.cam, 'num_images', N_dark),
                (xrd.hdf5, 'num_capture', N_dark),
                ]

        original_sigs = []
        for obj, key, value in sigs:
            if key in obj.stage_sigs:
                original_sigs.append((obj, key, obj.stage_sigs[key]))
            obj.stage_sigs[key] = value
        
        xrd_dets.append(xrd)
        reset_sigs.extend(original_sigs)
    
    if len(xrd_dets) > 0:
        d_status = shut_d.read()['shut_d_request_open']['value'] == 1 # is open
        yield from check_shutters(shutter, 'Close')
        print('Acquiring dark-field...')
        staging_list = [det._staged == Staged.yes for det in xrd_dets]
        print(f'{dexela.cam.num_images.get()=}')
        print(f'{dexela.hdf5.capture.get()=}')
        print(f'{dexela.cam.acquire.get()=}')
        print(f'{dexela.cam.detector_state.get()=}')
        for staged, det in zip(staging_list, xrd_dets):
            if staged:
                print('Unstaging')
                yield from bps.unstage(det)
            yield from bps.stage(det)
        print(f'{dexela.cam.num_images.get()=}')
        print(f'{dexela.hdf5.capture.get()=}')
        print(f'{dexela.cam.acquire.get()=}')
        print(f'{dexela.cam.detector_state.get()=}')
        yield from bps.sleep(1)

        # Pseudo continous triggering'
        # print('save')
        # yield from bps.drop() # Tie up previous stream ('baseline?')

        # print('pseudo-trigger')
        # for det in xrd_dets:
        #     yield from det.generate_datum(det._image_name, ttime.time(), {})        

        # print('create')
        # yield from bps.create('dark')
        # print(f'{dexela.cam.num_images.get()=}')
        # print(f'{dexela.hdf5.capture.get()=}')
        # print(f'{dexela.cam.acquire.get()=}')
        # print(f'{dexela.cam.detector_state.get()=}')



        # # print('Move to start saving')
        # # for det in xrd_dets:
        # #     yield from mv(det.hdf5.capture, 1)
        # yield from bps.sleep(1)


        print('Moving to acquire images')
        print(f'{dexela.cam.num_images.get()=}')
        print(f'{dexela.hdf5.capture.get()=}')
        print(f'{dexela.cam.acquire.get()=}')
        print(f'{dexela.cam.detector_state.get()=}')
        for det in xrd_dets:  
            yield from mv(det.cam.acquire, 1)
            yield from det.hdf5._generate_resource({})
        yield from bps.sleep(1)

        print('create')
        yield from bps.create('dark')
        print(f'{dexela.cam.num_images.get()=}')
        print(f'{dexela.hdf5.capture.get()=}')
        print(f'{dexela.cam.acquire.get()=}')
        print(f'{dexela.cam.detector_state.get()=}')     
        
        print('Wait for acquisition')
        print(f'{dexela.cam.num_images.get()=}')
        print(f'{dexela.hdf5.capture.get()=}')
        print(f'{dexela.cam.acquire.get()=}')
        print(f'{dexela.cam.detector_state.get()=}')
        # yield from bps.sleep(11)
        MAX_WAIT = 600 # in seconds
        t0 = ttime.time()
        for det in xrd_dets:
            print(f'waiting for {det.name}')
            print(det.cam.detector_state.get())
            print(f'{dexela.hdf5.capture.get()=}')
            print(f'{dexela.cam.acquire.get()=}')
            print(f'{dexela.cam.detector_state.get()=}')
            # While detector is still acquiring
            while det.hdf5.num_captured.get() != det.hdf5.num_capture.get():
                if det.hdf5.capture.get() == 0:
                    print('done!')
                    # Stop acquisition and break if hdf5 has all images
                    yield from mv(det.cam.acquire, 0)
                    break
                elif ttime.time() - t0 > MAX_WAIT:
                    warn_str = 'WARNING: Max wait time for continous acquistion has been exceeded. Breaking loop.'
                    print(warn_str)
                    yield from mv(det.hdf5.capture, 0,
                                  det.cam.acquire, 0)
                else:
                    print('still waiting!')
                    # Otherwise wait
                    yield from bps.sleep(0.1)
        print('after acquisition')
        
        # print('create')
        # yield from bps.create('dark')
        print('reading')
        for det in xrd_dets:
            yield from bps.read(det)

        print('saving?')        
        yield from bps.save()

        # Reset to original stage_sigs    
        for obj, key, value in reversed(reset_sigs):
            obj.stage_sigs[key] = value   
        
        for staged, det in zip(staging_list, xrd_dets):
            yield from bps.unstage(det)
            if staged:
                yield from bps.stage(det)

        if d_status:
            yield from check_shutters(shutter, 'Open')


def dark_count():

    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'DARK_TEST'
    scan_md['scan']['detectors'] = [dexela.name]                                  
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    @bpp.stage_decorator([dexela])
    @bpp.run_decorator(md=scan_md)
    def plan():
        yield from _continuous_dark_fields([dexela])
    
    return (yield from plan())



# WIP: Continous acquisition would be so much better!!!
# # Assumes detector stage sigs are already set
# def _acquire_dark_fields(dets,
#                          N_dark=10,
#                          shutter=True):                

#     dets_by_name = {d.name : d for d in dets}
#     xrd_dets = []
#     reset_sigs = []

#     if 'merlin' in dets_by_name:
#         xrd = dets_by_name['merlin']
#         sigs = OrderedDict(
#             [
#                 (xrd.cam.trigger_mode, 0),
#                 (xrd.total_points, N_dark),
#                 (xrd.cam.num_images, N_dark),
#                 (xrd.hdf5.num_capture, N_dark),
#             ]
#         )
#         # if 'acquire_time' in xrd.cam.stage_sigs:
#         #     sigs[xrd.cam.acquire_time] = xrd.cam.stage_sigs['acquire_time']
#         # if 'acquire_period' in xrd.cam.stage_sigs:
#         #     sigs[xrd.cam.acquire_period] = xrd.cam.stage_sigs['acquire_period']

#         original_vals = {sig: sig.get() for sig in sigs}
#         for sig, val in sigs.items():
#             yield from abs_set(sig, val)
#         xrd_dets.append(xrd)
#         reset_sigs.update(original_vals)

#     if 'dexela' in dets_by_name:
#         xrd = dets_by_name['dexela']
#         # sigs = OrderedDict(
#         #     [   
#         #         (xrd.total_points, N_dark),
#         #         (xrd.cam.trigger_mode, 'Int. Fixed Rate'),
#         #         (xrd.cam.image_mode, 'Multiple'),
#         #         (xrd.cam.num_images, N_dark),
#         #         (xrd.hdf5.num_capture, N_dark),
#         #     ]
#         # )

#         sigs = [
#                 # (xrd, 'total_points', 1),
#                 # (xrd, 'cam.image_mode', 'Single'),
#                 # (xrd.cam, 'trigger_mode', 'Int. Fixed Rate'),
#                 # (xrd.cam, 'image_mode', 'Single'),
#                 # (xrd.cam, 'num_images', 1),
#                 # (xrd.hdf5, 'num_capture', N_dark),

#                 (xrd, 'total_points', N_dark),
#                 (xrd, 'cam.image_mode', 'Multiple'),
#                 (xrd.cam, 'trigger_mode', 'Int. Fixed Rate'),
#                 (xrd.cam, 'image_mode', 'Multiple'),
#                 (xrd.cam, 'num_images', N_dark),
#                 (xrd.hdf5, 'num_capture', N_dark),
#                 ]
#         # if 'acquire_time' in xrd.cam.stage_sigs:
#         #     sigs[xrd.cam.acquire_time] = xrd.cam.stage_sigs['acquire_time']
#         # if 'acquire_period' in xrd.cam.stage_sigs:
#         #     sigs[xrd.cam.acquire_period] = xrd.cam.stage_sigs['acquire_period']

#         original_sigs = []
#         for obj, key, value in sigs:
#             if key in obj.stage_sigs:
#                 original_sigs.append((obj, key, obj.stage_sigs[key]))
            
#             obj.stage_sigs[key] = value
        
#         # original_vals = {sig: sig.get() for sig in sigs}
#         # xrd.stage()
#         # for sig, val in sigs.items():
#         #     yield from abs_set(sig, val, wait=True)
#         xrd_dets.append(xrd)
#         reset_sigs.extend(original_sigs)
    
#     if len(xrd_dets) > 0:
#         d_status = shut_d.read()['shut_d_request_open']['value'] == 1 # is open
#         yield from check_shutters(shutter, 'Close')
#         print('Acquiring dark-field...')
#         staging_list = [not det._staged for det in xrd_dets]
#         for stage, det in zip(staging_list, xrd_dets):
#             if stage:
#                 yield from bps.stage(det)
#         print([det.name for det in xrd_dets])
#         print([det._staged for det in xrd_dets])
#         print(dexela.cam.stage_sigs)
#         print(dexela.hdf5.stage_sigs['num_capture'])

#         # yield from bps.trigger_and_read(xrd_dets, name='dark')
#         # for _ in range(N_dark):
#         #     yield from bps.trigger_and_read(xrd_dets, name='dark')
#         # for _ in range(N_dark):

#         # ### COUNT-LIKE
#         # print('triggering')
#         # grp = _short_uid('trigger')
#         # no_wait = True
#         # for det in xrd_dets:
#         #     if hasattr(det, 'trigger'):
#         #         no_wait = False
#         #         yield from bps.trigger(det, group=grp)
#         # # Skip 'wait' if none of the devices implemented a trigger method.
#         # if not no_wait:
#         #     yield from wait(group=grp)
#         # print('creating')
#         # yield from bps.create('dark')
#         # print('reading')
#         # ret = {}
#         # for det in xrd_dets:
#         #     reading = (yield from bps.read(det))
#         #     if reading is not None:
#         #         ret.update(reading)
#         # print('saving')
#         # yield from save()

#         ### STREAM-LIKE
#         # print('creating')
#         # yield from bps.create('dark')
#         print('monitor')
#         for det in xrd_dets:
#             yield from bps.monitor(det, name='dark')
        
#         print('triggering')
#         for det in xrd_dets:
#             yield from bps.trigger(det)
        
#         print('waiting...')
#         yield from bps.sleep(5)
        
#         # print('reading')
#         # ret = {}
#         # for det in xrd_dets:
#         #     reading = (yield from bps.read(det))
#         #     if reading is not None:
#         #         ret.update(reading)
#         # print('saving')
#         # yield from save()

#         print('dark-field immediately acquired')
#         print(dexela.cam.stage_sigs)
#         print(dexela.hdf5.stage_sigs['num_capture'])
        
#         for stage, det in zip(staging_list, xrd_dets):
#             if stage:
#                 yield from bps.unstage(det)
#         if d_status:
#             yield from check_shutters(shutter, 'Open')
        
#         for obj, key, value in reset_sigs:
#             obj.stage_sigs[key] = value
        
#         # for sig, val in reset_sigs.items():
#         #     if val is None:
#         #         print(sig.parent.name)
#         #         del sig.parent.stage_sigs[sig.attr_name]
#         #     else:
#         #         sig.parent.stage_sigs[sig.attr_name] = val






def dark_energy_rocking_curve(e_low,
                              e_high,
                              e_num,
                              dwell,
                              xrd_dets,
                              shutter=True,
                              peakup_flag=True,
                              plotme=False,
                              return_to_start=True):

    start_energy = energy.energy.position

    # Convert to keV
    if e_low > 1000:
        e_low /= 1000
    if e_high > 1000:
        e_high /= 1000

    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    e_range = np.linspace(e_low, e_high, e_num)

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    _setup_xrd_dets(dets, dwell, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'ENERGY_RC'
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [d.name for d in dets]
    scan_md['scan']['energy'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    # Move to center energy and perform peakup
    if peakup_flag:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)
    
    @bpp.stage_decorator(list(dets) + [energy])
    @bpp.run_decorator(md=scan_md)
    @bpp.subs_wrapper(livecallbacks)
    def inner_plan():
        yield from _acquire_dark_fields(dets,
                                        N_dark=10,
                                        shutter=shutter)
        yield from check_sutters(shutter, 'Open')
        yield from list_scan(dets, energy, e_range)
        yield from check_sutters(shutter, 'Close')

        if return_to_start:
            yield from mov(energy, start_energy)
    
    return (yield from inner_plan())
    



def energy_rocking_curve(e_low,
                         e_high,
                         e_num,
                         dwell,
                         xrd_dets,
                         shutter=True,
                         peakup_flag=True,
                         plotme=False,
                         return_to_start=True):

    start_energy = energy.energy.position

    # Convert to keV
    if e_low > 1000:
        e_low /= 1000
    if e_high > 1000:
        e_high /= 1000

    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    e_range = np.linspace(e_low, e_high, e_num)

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    _setup_xrd_dets(dets, dwell, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'ENERGY_RC'
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [d.name for d in dets]
    scan_md['scan']['energy'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    # Move to center energy and perform peakup
    if peakup_flag:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)
    
    # yield from list_scan(dets, energy, e_range, md=scan_md)
    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(list_scan(dets, energy, e_range, md=scan_md),
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')

    if return_to_start:
        yield from mov(energy, start_energy)


def relative_energy_rocking_curve(e_range,
                                  e_num,
                                  dwell,
                                  xrd_dets,
                                  peakup_flag=False, # rewrite default
                                  **kwargs):
    
    en_current = energy.energy.position

    # Convert to keV. Not as straightforward as endpoint inputs
    if en_range > 5:
        warn_str = (f'WARNING: Assuming energy range of {en_range} '
                    + 'was given in eV.')
        print(warn_str)
        en_range /= 1000
    
    # Ensure energy.energy.positiion is reading correctly
    if en_current > 1000:
        en_current /= 1000

    e_low = en_current - (e_range / 2)
    e_high = en_current + (e_range / 2)
    
    yield from energy_rocking_curve(e_low,
                                    e_high,
                                    e_num,
                                    dwell,
                                    xrd_dets,
                                    peakup_flag=peakup_flag
                                    **kwargs)


def extended_energy_rocking_curve(e_low,
                                  e_high,
                                  e_num,
                                  dwell,
                                  xrd_dets,
                                  shutter=True):

    # Breaking an extended energy rocking curve up into smaller pieces
    # The goal is to allow for multiple intermittent peakups

    # Convert to ev
    if e_low > 1000:
        e_low /= 1000
    if e_high > 1000:
        e_high /= 1000

    # Loose chunking at about 1000 eV
    e_range = e_high - e_low

    e_step = e_range / e_num
    e_chunks = int(np.round(e_num / e_range))

    e_vals = np.linspace(e_low, e_high, e_num)

    e_rcs = [list(e_vals[i:i + e_chunks]) for i in range(0, len(e_vals), e_chunks)]
    e_rcs[-2].extend(e_rcs[-1])
    e_rcs.pop(-1)

    for e_rc in e_rcs:
        yield from energy_rocking_curve(e_rc[0],
                                        e_rc[-1],
                                        len(e_rc),
                                        dwell,
                                        xrd_dets,
                                        shutter=shutter,
                                        peakup_flag=True,
                                        plotme=False,
                                        return_to_start=False)


def dark_angle_rocking_curve(th_low,
                             th_high,
                             th_num,
                             dwell,
                             xrd_dets,
                             shutter=True,
                             plotme=False,
                             return_to_start=True):
    # th in mdeg!!!
    start_th = nano_stage.th.user_readback.get()

    # Define some useful variables
    th_range = np.linspace(th_low, th_high, th_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'ANGLE_RC'
    scan_md['scan']['scan_input'] = [th_low, th_high, th_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['angles'] = th_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    livecallbacks = [LiveTable(['nano_stage_th_user_setpoint', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='nano_stage_th_user_setpoint'))

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    _setup_xrd_dets(dets, dwell, th_num)

    @bpp.stage_decorator(list(dets) + [nano_stage.th])
    @bpp.run_dectorator(md=scan_md)
    @bpp.subs_wrapper(livecallbacks)
    def inner_plan():
        yield from _acquire_dark_fields(dets,
                                        N_dark=10,
                                        shutter=shutter)
        yield from check_sutters(shutter, 'Open')
        yield from list_scan(dets, nano_stage.th, th_range)
        yield from check_sutters(shutter, 'Close')

        if return_to_start:
            yield from mov(nano_stage.th, start_th)
    
    return (yield from inner_plan())


def angle_rocking_curve(th_low,
                        th_high,
                        th_num,
                        dwell,
                        xrd_dets,
                        shutter=True,
                        plotme=False,
                        return_to_start=True):
    # th in mdeg!!!

    start_th = nano_stage.th.user_readback.get()

    # Define some useful variables
    th_range = np.linspace(th_low, th_high, th_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'ANGLE_RC'
    scan_md['scan']['scan_input'] = [th_low, th_high, th_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['angles'] = th_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    livecallbacks = [LiveTable(['nano_stage_th_user_setpoint', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='nano_stage_th_user_setpoint'))

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    _setup_xrd_dets(dets, dwell, th_num)
    
    # yield from list_scan(dets, energy, e_range, md=scan_md)
    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(list_scan(dets, nano_stage.th, th_range, md=scan_md),
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')

    if return_to_start:
        yield from mov(nano_stage.th, start_th)


def relative_angle_rocking_curve(th_range,
                                 th_num,
                                 dwell,
                                 xrd_dets,
                                 **kwargs):
    
    th_current = nano_stage.th.user_readback.get()
    th_low = th_current - (th_range / 2)
    th_high = th_current + (th_range / 2)

    yield from angle_rocking_curve(th_low,
                                   th_high,
                                   th_num,
                                   dwell,
                                   xrd_dets,
                                   **kwargs)


def flying_angle_rocking_curve(th_low,
                               th_high,
                               th_num,
                               dwell,
                               xrd_dets,
                               return_to_start=True,
                               **kwargs):
    # More direct convenience wrapper for scan_and_fly

    start_th = nano_stage.th.user_readback.get()
    y_current = nano_stage.y.user_readback.get()

    kwargs.setdefault('xmotor', nano_stage.th)
    kwargs.setdefault('ymotor', nano_stage.y)
    kwargs.setdefault('flying_zebra', nano_flying_zebra_coarse)
    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR')
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

    _xs = kwargs.pop('xs', xs)
    if xrd_dets is None:
        xrd_dets = []
    dets = [_xs] + xrd_dets

    yield from scan_and_fly_base(dets,
                                 th_low,
                                 th_high,
                                 th_num,
                                 y_current,
                                 y_current,
                                 1,
                                 dwell,
                                 **kwargs)
    
    # Is this needed for scan_and_fly_base???
    if return_to_start:
        yield from mov(nano_stage.th, start_th)

    
def relative_flying_angle_rocking_curve(th_range,
                                        th_num,
                                        dwell,
                                        xrd_dets,
                                        **kwargs):
    
    th_current = nano_stage.th.user_readback.get()
    th_low = th_current - (th_range / 2)
    th_high = th_current + (th_range / 2)

    yield from flying_angle_rocking_curve(th_low,
                                          th_high,
                                          th_num,
                                          dwell,
                                          xrd_dets,
                                          **kwargs)
    

# def continous_xrd(xrd_dets,
#                   num,
#                   dwell,
#                   shutter,
#                   plotme=False):
    

#     # Defining scan metadata
#     scan_md = {}
#     get_stock_md(scan_md)
#     scan_md['scan']['type'] = 'STATIC_XRD'
#     scan_md['scan']['scan_input'] = [num, dwell]
#     scan_md['scan']['dwell'] = dwell
#     scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
#     scan_md['scan']['energy'] = f'{energy.energy.position:.5f}'                                 
#     scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

#     # Live Callbacks
#     # What does energy_energy read??
#     # livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
#     # if plotme:
#     #     livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

#     # Define detectors
#     dets = [xs, sclr1] + xrd_dets
#     _setup_xrd_dets(dets, dwell, num)
    
#     dets_by_name = {det.name : det for det in dets}
#     if 'dexela' in dets_by_name:
#         xrd = dets_by_name['dexela']



#     @bpp.stage_decorator(list(dets) + [energy])
#     @bpp.run_dectorator(md=scan_md)
#     # @bpp.subs_wrapper(livecallbacks)
#     def inner_plan():
#         yield from check_sutters(shutter, 'Open')

#         for det in dets

#         yield from check_sutters(shutter, 'Close')
    
#     return (yield from inner_plan())  


# This would be so much better if the data were streamed continously...
# A static xrd measurement without changing energy or moving stages
def dark_static_xrd(xrd_dets,
               num,
               dwell,
               shutter=True,
               plotme=False):

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'STATIC_XRD'
    scan_md['scan']['scan_input'] = [num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energy'] = f'{energy.energy.position:.5f}'                                 
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    # What does energy_energy read??
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    _setup_xrd_dets(dets, dwell, num)

    @bpp.stage_decorator(list(dets) + [energy])
    @bpp.run_dectorator(md=scan_md)
    @bpp.subs_wrapper(livecallbacks)
    def inner_plan():
        yield from _acquire_dark_fields(dets,
                                        N_dark=10,
                                        shutter=shutter)
        yield from check_sutters(shutter, 'Open')
        yield from count(dets, num)
        yield from check_sutters(shutter, 'Close')
    
    return (yield from inner_plan())


# This would be so much better if the data were streamed continously...
# A static xrd measurement without changing energy or moving stages
def static_xrd(xrd_dets,
               num,
               dwell,
               shutter=True,
               plotme=False):

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'STATIC_XRD'
    scan_md['scan']['scan_input'] = [num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energy'] = f'{energy.energy.position:.5f}'                                 
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    # What does energy_energy read??
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total'])]
    
    if plotme:
        livecallbacks.append(LivePlot('dexela_stats2_total', x='energy_energy'))

    # Define detectors
    dets = [xs, sclr1] + xrd_dets
    _setup_xrd_dets(dets, dwell, num)

    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(count(dets, num, md=scan_md), # I guess dwell info is carried by the detector
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')


# EJM: Commented out 20250228. To be removed.
# Previous Scans
# def update_scan_id():
#     scanrecord.current_scan.put(db[-1].start['uid'][:6])
#     scanrecord.current_scan_id.put(str(RE.md['scan_id']))


# def collect_xrd(pos=[], empty_pos=[], acqtime=1, N=1,
#                 dark_frame=False, shutter=True):
#     # Scan parameters
#     if (acqtime < 0.0066392):
#         print('The detector should not operate faster than 7 ms.')
#         print('Changing the scan dwell time to 7 ms.')
#         acqtime = 0.007

#     N_pts = len(pos)
#     if (pos == []):
#         pos = [[hf_stage.x.position, hf_stage.y.position, hf_stage.z.position]]
#         # pos = [[hf_stage.topx.position, hf_stage.y.position]]
#     if (empty_pos != []):
#         N_pts = N_pts + 1
#     if (dark_frame):
#         N_pts = N_pts + 1

#     N_tot = N_pts * N

#     # Setup detector
#     xrd_det = [dexela]
#     for d in xrd_det:
#         # d.cam.stage_sigs['acquire_time'] = acqtime
#         # d.cam.stage_sigs['acquire_period'] = acqtime + 0.100
#         # d.cam.stage_sigs['num_images'] = 1
#         # d.stage_sigs['total_points'] = N_tot
#         # d.hdf5.stage_sigs['num_capture'] = N_tot
#         d.stage_sigs['cam.acquire_time'] = acqtime
#         d.stage_sigs['cam.acquire_period'] = acqtime + 0.100
#         d.stage_sigs['cam.image_mode'] = 1
#         d.stage_sigs['cam.num_images'] = N
#         d.stage_sigs['total_points'] = N_tot
#         d.stage_sigs['hdf5.num_capture'] = N_tot


#     # Collect dark frame
#     if (dark_frame):
#         # Check shutter
#         if (shutter):
#            yield from bps.mov(shut_b, 'Close')

#         # Trigger detector
#         yield from count(xrd_det, num=N)
#         # yield from count(xrd_det)
#         update_scan_id()

#         # Write to logfile
#         # logscan('xrd_count')

#     if (empty_pos != []):
#         # Move into position
#         yield from bps.mov(hf_stage.x, empty_pos[0])
#         # yield from bps.mov(hf_stage.topx, empty_pos[0])
#         yield from bps.mov(hf_stage.y, empty_pos[1])
#         if (len(empty_pos) == 3):
#             yield from bps.mov(hf_stage.z, empty_pos[2])

#         # Check shutter
#         if (shutter):
#             yield from bps.mov(shut_b, 'Open')

#         # Trigger the detector
#         yield from count(xrd_det, num=N)
#         # yield from count(xrd_det)
#         # yield from bps.trigger_and_read(xrd_det)
#         update_scan_id()

#         # Close shutter
#         if (shutter):
#             yield from bps.mov(shut_b, 'Close')

#         # Write to logfile
#         # logscan('xrd_count')

#     # Loop through positions
#     if (pos == []):
#         pos = [[hf_stage.x.position, hf_stage.y.position]]
#     for i in range(len(pos)):
#         i = int(i)
#         # Move into position
#         yield from bps.mov(hf_stage.x, pos[i][0])
#         # yield from bps.mov(hf_stage.topx, pos[i][0])
#         yield from bps.mov(hf_stage.y, pos[i][1])
#         if (len(pos[i]) == 3):
#             yield from bps.mov(hf_stage.z, pos[i][2])

#         # Check shutter
#         if (shutter):
#             yield from bps.mov(shut_b, 'Open')

#         # Trigger the detector
#         # Keep getting TimeoutErrors setting capture complete to False
#         # This is a lot of duct tape, but hopefully this will let the scans
#         # work instead of timing out
#         need_data = True
#         num_tries = 0
#         while (need_data):
#             try:
#                 num_tries = num_tries + 1
#                 yield from count(xrd_det, num=N)
#                 # yield from count(xrd_det)
#                 # yield from bps.trigger_and_read(xrd_det)
#                 need_data = False
#                 update_scan_id()
#             except TimeoutError:
#                 dexela.unstage()
#                 if (num_tries >= 5):
#                     print('Timeout has occured 5 times.')
#                     raise
#                 print('TimeoutError: Trying again...')
#                 yield from bps.sleep(5)
#             except:
#                 raise


#         # Close shutter
#         if (shutter):
#             yield from bps.mov(shut_b, 'Close')

#         # Write to logfile
#         # logscan('xrd_count')


# def collect_xrd_map(xstart, xstop, xnum,
#                     ystart, ystop, ynum, acqtime=1,
#                     dark_frame=False, shutter=True):

#     # Scan parameters
#     if (acqtime < 0.0066392):
#         print('The detector should not operate faster than 7 ms.')
#         print('Changing the scan dwell time to 7 ms.')
#         acqtime = 0.007

#     N = xnum * ynum

#     # Setup detector
#     xrd_det = [dexela]
#     for d in xrd_det:
#         d.stage_sigs['cam.acquire_time'] = acqtime
#         d.stage_sigs['cam.acquire_period'] = acqtime + 0.100
#         d.stage_sigs['cam.image_mode'] = 1
#         d.stage_sigs['cam.num_images'] = 1
#         d.stage_sigs['total_points'] = N
#         d.stage_sigs['hdf5.num_capture'] = N

#     # Collect dark frame
#     if (dark_frame):
#         # Check shutter
#         if (shutter):
#            yield from bps.mov(shut_b, 'Close')

#         # Trigger detector
#         yield from count(xrd_det, num=N)
#         update_scan_id()

#         # Write to logfile
#         # logscan('xrd_count')

#     if shutter:
#         yield from mv(shut_b, 'Open')

#     #scan_dets = xrd_det.append(sclr1)
#     #print(xrd_det)
#     # yield from outer_product_scan(scan_dets,
#     #                               hf_stage.x, xstart, xstop, xnum,
#     #                               hf_stage.y, ystart, ystop, ynum, False)
#     yield from grid_scan(xrd_det,
#                          hf_stage.x, xstart, xstop, xnum,
#                          hf_stage.y, ystart, ystop, ynum, False)

#     if shutter:
#         yield from mv(shut_b, 'Close')

#     # Write to logfile
#     # logscan('xrd_count')
#     update_scan_id()


# @parameter_annotation_decorator({
#     "parameters": {
#         "extra_dets": {"default": "['dexela']"},
#     }
# })
# def xrd_fly(*args, extra_dets=[dexela], **kwargs):
#     kwargs.setdefault('xmotor', nano_stage.sx)
#     kwargs.setdefault('ymotor', nano_stage.sy)
#     kwargs.setdefault('flying_zebra', nano_flying_zebra)

#     yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR', wait=True)
#     yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

#     _xs = kwargs.pop('xs', xs)
#     if extra_dets is None:
#         extra_dets = []
#     dets = [_xs] + extra_dets
#     # To fly both xs and merlin
#     # yield from scan_and_fly_base([_xs, merlin], *args, **kwargs)
#     # To fly only xs
#     yield from scan_and_fly_base(dets, *args, **kwargs)


# def make_tiff(scanid, *, scantype='', fn=''):
#     h = db[int(scanid)]
#     if scanid == -1:
#         scanid = int(h.start['scan_id'])

#     if scantype == '':
#         scantype = h.start['scan']['type']

#     if ('FLY' in scantype):
#         d = list(h.data('dexela_image', stream_name='stream0', fill=True))
#         d = np.array(d)

#         (row, col, imgY, imgX) = d.shape
#         if (d.size == 0):
#             print('Error collecting dexela data...')
#             return
#         d = np.reshape(d, (row*col, imgY, imgX))
#     elif ('STEP' in scantype):
#         d = list(h.data('dexela_image', fill=True))
#         d = np.array(d)
#         d = np.squeeze(d)
#     # elif (scantype == 'count'):
#     #     d = list(h.data('dexela_image', fill=True))
#     #     d = np.array(d)
#     else:
#         print('I don\'t know what to do.')
#         return

#     if fn == '':
#         fn = f"scan{scanid}_xrd.tiff"
#     try:
#         io.imsave(fn, d.astype('uint16'))
#     except:
#         print(f'Error writing file!')
