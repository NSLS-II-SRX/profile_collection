print(f'Loading {__file__}...')


import epics
import os
import collections
import numpy as np
import time as ttime
import matplotlib.pyplot as plt

import bluesky.plans as bp
from bluesky.plans import outer_product_scan, scan
from bluesky.callbacks import LiveGrid
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import subs_wrapper
import bluesky.plan_stubs as bps
from bluesky.plan_stubs import mv, abs_set
from bluesky.simulators import plot_raster_path


def hf2dxrf(*, xstart, xnumstep, xstepsize,
            ystart, ynumstep, ystepsize, acqtime,
            shutter=True, align=False, xmotor=hf_stage.x, ymotor=hf_stage.y,
            numrois=1, extra_dets=[],
            setenergy=None, u_detune=None, echange_waittime=10, samplename=None, snake=True):

    '''input:
        xstart, xnumstep, xstepsize : float
        ystart, ynumstep, ystepsize : float
        acqtime : float
             acqusition time to be set for both xspress3 and F460
        numrois : integer
            number of ROIs set to display in the live raster scans.
            This is for display ONLY.  The actualy number of ROIs
            saved depend on how many are enabled and set in the
            read_attr However noramlly one cares only the raw XRF
            spectra which are all saved and will be used for fitting.
        energy (float): set energy, use with caution, hdcm might
            become misaligned
        u_detune (float): amount of undulator to
            detune in the unit of keV

    '''
    # Record relevant metadata in the Start document, defined in 90-usersetup.py
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['sample']  = {'name': samplename}
    scan_md['scan_input'] = str([xstart, xnumstep, xstepsize, ystart, ynumstep, ystepsize, acqtime])
    scan_md['scaninfo']  = {'type': 'XRF',
                            'raster' : True}

    # Setup detectors
    dets = [sclr1, xs]
    dets = dets + extra_dets
    dets_by_name = {d.name : d
                    for d in dets}
    # Scaler
    if (acqtime < 0.001):
        acqtime = 0.001
    sclr1.preset_time.put(acqtime)
    # XS3
    xs.external_trig.put(False)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put((xnumstep + 1) * (ynumstep + 1))

    if ('merlin' in dets_by_name):
        dpc = dets_by_name['merlin']

        # Setup Merlin
        dpc.cam.trigger_mode.put(0)
        dpc.cam.acquire_time.put(acqtime)
        dpc.cam.acquire_period.put(acqtime + 0.005)
        dpc.cam.num_images.put(1)
        dpc.hdf5.stage_sigs['num_capture'] = (xnumstep + 1) * (ynumstep + 1)
        dpc._mode = SRXMode.step
        dpc.total_points.put((xnumstep + 1) * (ynumstep + 1))

    if ('xs2' in dets_by_name):
        xs2 = dets_by_name['xs2']
        xs2.external_trig.put(False)
        xs2.settings.acquire_time.put(acqtime)
        xs2.total_points.put((xnumstep + 1) * (ynumstep + 1))

    # Setup the live callbacks
    livecallbacks = []
    # Setup scanbroker to update time remaining
    def time_per_point(name, doc, st=ttime.time()):
        if ('seq_num' in doc.keys()):
            scanrecord.scan0.tpp.put((doc['time'] - st) / doc['seq_num'])
            scanrecord.scan0.curpt.put(int(doc['seq_num']))
            scanrecord.time_remaining.put((doc['time'] - st) / doc['seq_num'] *
                                          ((xnumstep + 1) * (ynumstep + 1) - doc['seq_num']) / 3600)
    livecallbacks.append(time_per_point)

    # Setup LiveTable
    livetableitem = [xmotor.name, ymotor.name, i0.name]
    xstop = xstart + xnumstep * xstepsize
    ystop = ystart + ynumstep * ystepsize

    for roi_idx in range(numrois):
        roi_name = 'roi{:02}'.format(roi_idx+1)
        roi_key = getattr(xs.channel1.rois, roi_name).value.name
        livetableitem.append(roi_key)
        fig = plt.figure('xs_ROI{:02}'.format(roi_idx+1))
        fig.clf()
        roimap = LiveGrid((ynumstep+1, xnumstep+1), roi_key,
                          clim=None, cmap='viridis',
                          xlabel='x (mm)', ylabel='y (mm)',
                          extent=[xstart, xstop, ystart, ystop],
                          x_positive='right', y_positive='down',
                          ax=fig.gca())
        livecallbacks.append(roimap)
    
    if ('xs2' in dets_by_name):
        for roi_idx in range(numrois):
            roi_key = getattr(xs2.channel1.rois, roi_name).value.name
            livetableitem.append(roi_key)
            fig = plt.figure('xs2_ROI{:02}'.format(roi_idx+1))
            fig.clf()
            roimap = LiveGrid((ynumstep+1, xnumstep+1), roi_key,
                              clim=None, cmap='viridis',
                              xlabel='x (mm)', ylabel='y (mm)',
                              extent=[xstart, xstop, ystart, ystop],
                              x_positive='right', y_positive='down',
                              ax=fig.gca())
            livecallbacks.append(roimap)

    if ('merlin' in dets_by_name) and (hasattr(dpc, 'stats1')):
        fig = plt.figure('DPC')
        fig.clf()
        dpc_tmap = LiveGrid((ynumstep+1, xnumstep+1),
                            dpc.stats1.total.name, clim=None, cmap='viridis',
                            xlabel='x (mm)', ylabel='y (mm)',
                            x_positive='right', y_positive='down',
                            extent=[xstart, xstop, ystart, ystop],
                            ax=fig.gca())
        livecallbacks.append(dpc_tmap)

    # Change energy (if provided)
    if (setenergy is not None):
        if (u_detune is not None):
            energy.detune.put(u_detune)
        print('Changing energy to ', setenergy)
        yield from mv(energy, setenergy)
        print('Waiting time (s) ', echange_waittime)
        yield from bps.sleep(echange_waittime)

    def at_scan(name, doc):
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        scanrecord.current_type.put(scan_md['scaninfo']['type'])
        scanrecord.scanning.put(True)

    def finalize_scan(name, doc):
        scanrecord.scanning.put(False)

    # Setup the scan
    hf2dxrf_scanplan = outer_product_scan(dets,
                                          ymotor, ystart, ystop, ynumstep+1,
                                          xmotor, xstart, xstop, xnumstep+1, snake,
                                          md=scan_md)
    hf2dxrf_scanplan = subs_wrapper(hf2dxrf_scanplan,
                                    {'all': livecallbacks,
                                     'start': at_scan,
                                     'stop': finalize_scan})
    # Move to starting position
    yield from mv(xmotor, xstart,
                  ymotor, ystart)

    # Peak up monochromator at this energy
    if (align):
        yield from peakup_fine(shutter=shutter)
    
    # Open shutter
    if (shutter):
        yield from mv(shut_b,'Open')

    # Run the scan
    scaninfo = yield from hf2dxrf_scanplan

    #TO-DO: implement fast shutter control (close)
    if (shutter):
        yield from mv(shut_b, 'Close')

    # Write to scan log
    if ('merlin' in dets_by_name):
        logscan_event0info('2dxrf_withdpc')
        # Should this be here?
        merlin.hdf5.stage_sigs['num_capture'] = 0
    else:
        logscan_detailed('2dxrf')

    return scaninfo


# I'm not sure how to use this function
def multi_region_h(regions, energy_list=None, **kwargs):
    ret = []

    for r in regions:
        inp = {}
        inp.update(kwargs)
        inp.update(r)
        rs_uid = yield from hf2dxrf(**inp)
        ret.extend(rs_uid)
    return ret


# Not sure how often this will be used....but at least it's updated
def hf2dxrf_repeat(num_scans=None, waittime=10,
                   xstart=None, xnumstep=None, xstepsize=None,
                   ystart=None, ynumstep=None, ystepsize=None,
                   acqtime=None, numrois=0, i0map_show=False, itmap_show = False
                   ):
    '''
    This function will repeat the 2D XRF scans on the same spots for specified number of the scans.
    input:
        num_scans (integer): number of scans to be repeated on the same position.
        waittime (float): wait time in sec. between each scans. Recommand to have few seconds for the HDF5 to finish closing.
        Other inputs are described as in hf2dxrf.
    '''
    if num_scans is None:
        raise Exception('Please specify "num_scans" as the number of scans to be run. E.g. num_scans = 3.')

    for i in range(num_scans):
        yield from hf2dxrf(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize,
                           ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize,
                           acqtime=acqtime, numrois=numrois)
        if (i != num_scans-1):
            print(f'Waiting {waittime} seconds between scans...')
            yield from bps.sleep(waittime)


def hf2dxrf_ioc(samplename=None, align=False, numrois=1, shutter=True, waittime=10):
    '''
    invokes hf2dxrf repeatedly with parameters provided separately.
        waittime                [sec]       time to wait between scans
        shutter                 [bool]      scan controls shutter
    '''
    scanlist = [ scanrecord.scan15, scanrecord.scan14, scanrecord.scan13,
                 scanrecord.scan12, scanrecord.scan11, scanrecord.scan10,
                 scanrecord.scan9, scanrecord.scan8, scanrecord.scan7,
                 scanrecord.scan6, scanrecord.scan5, scanrecord.scan4,
                 scanrecord.scan3, scanrecord.scan2, scanrecord.scan1,
                 scanrecord.scan0 ]

    Nscan = 0
    for scannum in range(len(scanlist)):
        thisscan = scanlist.pop()
        Nscan = Nscan + 1
        if thisscan.ena.get() == 1:
            scanrecord.current_scan.put('Scan {}'.format(Nscan))
            xstart = thisscan.p1s.get()
            xnumstep = int(thisscan.p1stp.get())
            xstepsize = thisscan.p1i.get()
            ystart = thisscan.p2s.get()
            ynumstep = int(thisscan.p2stp.get())
            ystepsize = thisscan.p2i.get()
            acqtime = thisscan.acq.get()

            hf2dxrf_gen = yield from hf2dxrf(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize,
                                             ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize,
                                             acqtime=acqtime, samplename=None, align=False, numrois=1,
                                             shutter=True)

            if (len(scanlist) is not 0):
                yield from bps.sleep(waittime)
    scanrecord.current_scan.put('')


def fermat_master_plan(*args, exp_time=0.2, testing=True, **kwargs):
    plan = bp.rel_spiral_fermat(*args, **kwargs)
    d = plot_raster_path(plan, args[1].name, args[2].name, probe_size=.001, lw=0.5)
    num_points = d['path'].get_path().vertices.shape[0]
    print(f"Number of points: {num_points}")
    if not testing:
        yield from bps.mv(merlin.total_points, num_points)
        yield from rel_spiral_fermat(*args, **kwargs)
