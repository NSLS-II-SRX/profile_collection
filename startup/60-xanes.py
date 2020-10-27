print(f'Loading {__file__}...')

import collections
import numpy as np
import time as ttime
import matplotlib.pyplot as plt

import bluesky.plans as bp
from bluesky.plans import list_scan
from bluesky.plan_stubs import (mv, one_1d_step)
from bluesky.preprocessors import (finalize_wrapper, subs_wrapper)
from bluesky.utils import short_uid as _short_uid
from epics import PV
# from databroker import DataBroker as db
from databroker import get_events


def xanes_textout(scan=-1, header=[], userheader={}, column=[], usercolumn={},
                  usercolumnname=[], output=True, filename_add='', filedir=None):
    '''
    scan: can be scan_id (integer) or uid (string). default = -1 (last scan run)
    header: a list of items that exist in the event data to be put into the header
    userheader: a dictionary defined by user to put into the hdeader
    column: a list of items that exist in the event data to be put into the column data
    output: print all header fileds. if output = False, only print the ones that were able to be written
            default = True

    '''
    if (filedir is None):
        filedir = userdatadir
    h = db[scan]
    # get events using fill=False so it does not look for the metadata in filestorage with reference (hdf5 here)
    events = list(get_events(h, fill=False, stream_name='primary'))

    if (filename_add is not ''):
        filename = 'scan_' + str(h.start['scan_id']) + '_' + filename_add
    else:
        filename = 'scan_' + str(h.start['scan_id'])

    f = open(filedir+filename, 'w')

    staticheader = '# XDI/1.0 MX/2.0\n' \
              + '# Beamline.name: ' + h.start['beamline_id'] + '\n' \
              + '# Facility.name: NSLS-II\n' \
              + '# Facility.ring_current:' + str(events[0]['data']['ring_current']) + '\n' \
              + '# Scan.start.uid: ' + h.start['uid'] + '\n' \
              + '# Scan.start.time: '+ str(h.start['time']) + '\n' \
              + '# Scan.start.ctime: ' + ttime.ctime(h.start['time']) + '\n' \
              + '# Mono.name: Si 111\n'

    f.write(staticheader)

    for item in header:
        if (item in events[0].data.keys()):
            f.write('# ' + item + ': ' + str(events[0]['data'][item]) + '\n')
            if (output is True):
                print(item + ' is written')
        else:
            print(item + ' is not in the scan')

    for key in userheader:
        f.write('# ' + key + ': ' + str(userheader[key]) + '\n')
        if (output is True):
            print(key + ' is written')

    for idx, item in enumerate(column):
        if (item in events[0].data.keys()):
            f.write('# Column.' + str(idx+1) + ': ' + item + '\n')

    f.write('# ')
    for item in column:
        if (item in events[0].data.keys()):
            f.write(str(item) + '\t')

    for item in usercolumnname:
        f.write(item + '\t')

    f.write('\n')
    f.flush()

    idx = 0
    for event in events:
        for item in column:
            if (item in events[0].data.keys()):
                f.write('{0:8.6g}  '.format(event['data'][item]))
        for item in usercolumnname:
            try:
                f.write('{0:8.6g}  '.format(usercolumn[item][idx]))
            except KeyError:
                idx += 1
                f.write('{0:8.6g}  '.format(usercolumn[item][idx]))
        idx = idx + 1
        f.write('\n')

    f.close()


def xanes_afterscan_plan(scanid, filename, roinum):
    # Custom header list
    headeritem = []
    # Load header for our scan
    h = db[scanid]

    # Construct basic header information
    userheaderitem = {}
    userheaderitem['uid'] = h.start['uid']
    userheaderitem['sample.name'] = h.start['sample']['name']
    userheaderitem['initial_sample_position.hf_stage.x'] = h.start['initial_sample_position']['hf_stage_x']
    userheaderitem['initial_sample_position.hf_stage.y'] = h.start['initial_sample_position']['hf_stage_y']
    userheaderitem['hfm.y'] = h.start['hfm']['y']
    userheaderitem['hfm.bend'] = h.start['hfm']['bend']

    # Create columns for data file
    columnitem = ['energy_energy', 'energy_bragg', 'energy_c2_x']
    # Include I_M, I_0, and I_t from the SRS
    if ('sclr1' in h.start['detectors']):
        columnitem = columnitem + ['sclr_im', 'sclr_i0', 'sclr_it']
    else:
        raise KeyError("SRS not found in data!")
    # Include fluorescence data if present, allow multiple rois
    if ('xs' in h.start['detectors']):
        if (type(roinum) is not list):
            roinum = [roinum]
        for i in roinum:
            roi_name = 'roi{:02}'.format(i)
            roi_key = []
            roi_key.append(getattr(xs.channel1.rois, roi_name).value.name)
            roi_key.append(getattr(xs.channel2.rois, roi_name).value.name)
            roi_key.append(getattr(xs.channel3.rois, roi_name).value.name)
            roi_key.append(getattr(xs.channel4.rois, roi_name).value.name)

        [columnitem.append(roi) for roi in roi_key]
    if ('xs2' in h.start['detectors']):
        if (type(roinum) is not list):
            roinum = [roinum]
        for i in roinum:
            roi_name = 'roi{:02}'.format(i)
            roi_key = []
            roi_key.append(getattr(xs2.channel1.rois, roi_name).value.name)

        [columnitem.append(roi) for roi in roi_key]
    # Construct user convenience columns allowing prescaling of ion chamber, diode and
    # fluorescence detector data
    usercolumnitem = {}
    datatablenames = []

    if ('xs' in h.start['detectors']):
        datatablenames = datatablenames + [str(roi) for roi in roi_key]
    if ('xs2' in h.start['detectors']):
        datatablenames = datatablenames + [str(roi) for roi in roi_key]
    if ('sclr1' in  h.start['detectors']):
        datatablenames = datatablenames + ['sclr_im', 'sclr_i0', 'sclr_it']
        datatable = h.table(stream_name='primary', fields=datatablenames)
        im_array = np.array(datatable['sclr_im'])
        i0_array = np.array(datatable['sclr_i0'])
        it_array = np.array(datatable['sclr_it'])
    else:
        raise KeyError
    # Calculate sums for xspress3 channels of interest
    if ('xs' in h.start['detectors']):
        for i in roinum:
            roi_name = 'roi{:02}'.format(i)
            roisum = datatable[getattr(xs.channel1.rois, roi_name).value.name]
            roisum = roisum + datatable[getattr(xs.channel2.rois, roi_name).value.name]
            roisum = roisum + datatable[getattr(xs.channel3.rois, roi_name).value.name]
            roisum = roisum + datatable[getattr(xs.channel4.rois, roi_name).value.name]
            usercolumnitem['If-{:02}'.format(i)] = roisum
            usercolumnitem['If-{:02}'.format(i)].round(0)
    if ('xs2' in h.start['detectors']):
        for i in roinum:
            roi_name = 'roi{:02}'.format(i)
            roisum = datatable[getattr(xs2.channel1.rois, roi_name).value.name]
            usercolumnitem['If-{:02}'.format(i)] = roisum
            usercolumnitem['If-{:02}'.format(i)].round(0)

    xanes_textout(scan = scanid, header = headeritem,
                  userheader = userheaderitem, column = columnitem,
                  usercolumn = usercolumnitem,
                  usercolumnname = usercolumnitem.keys(),
                  output = False, filename_add = filename, filedir=userdatadir)


def xanes_plan(erange=[], estep=[], acqtime=1., samplename='', filename='',
               det_xs=xs, harmonic=1, detune=0, align=False, align_at=None,
               roinum=1, shutter=True, per_step=None):

    '''
    erange (list of floats): energy ranges for XANES in eV, e.g. erange = [7112-50, 7112-20, 7112+50, 7112+120]
    estep  (list of floats): energy step size for each energy range in eV, e.g. estep = [2, 1, 5]
    acqtime (float): acqusition time to be set for both xspress3 and preamplifier
    samplename (string): sample name to be saved in the scan metadata
    filename (string): filename to be added to the scan id as the text output filename

    det_xs (xs3 detector): the xs3 detector used for the measurement
    harmonic (odd integer): when set to 1, use the highest harmonic achievable automatically.
                                    when set to an odd integer, force the XANES scan to use that harmonic
    detune:  add this value to the gap of the undulator to reduce flux [keV]
    align:  perform peakup_fine before scan [bool]
    align_at:  energy at which to align, default is average of the first and last energy points
    roinum: select the roi to be used to calculate the XANES spectrum
    shutter:  instruct the scan to control the B shutter [bool]
    per_step:  use a custom function for each energy point
    '''

    # Make sure user provided correct input
    if (erange is []):
        raise AttributeError("An energy range must be provided in a list by means of the 'erange' keyword.")
    if (estep is []):
        raise AttributeError("A list of energy steps must be provided by means of the 'esteps' keyword.")
    if (not isinstance(erange,list)) or (not isinstance(estep,list)):
        raise TypeError("The keywords 'estep' and 'erange' must be lists.")
    if (len(erange)-len(estep) is not 1):
        raise ValueError("The 'erange' and 'estep' lists are inconsistent; " \
                         + 'c.f., erange = [7000, 7100, 7150, 7500], estep = [2, 0.5, 5] ')
    if (type(roinum) is not list):
        roinum = [roinum]
    if (detune is not 0):
        yield from abs_set(energy.detune, detune)

    # Record relevant meta data in the Start document, defined in 90-usersetup.py
    # Add user meta data
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['sample'] = {'name' : samplename}
    scan_md['scaninfo'] = {'type' : 'XANES', 
                           'ROI' : roinum, 
                           'raster' : False,
                           'dwell' : acqtime}
    scan_md['scan_input'] = str(np.around(erange, 2)) + ', ' + str(np.around(estep, 2))
    
    # Convert erange and estep to numpy array
    ept = np.array([])
    erange = np.array(erange)
    estep = np.array(estep)
    # Calculation for the energy points
    for i in range(len(estep)):
        ept = np.append(ept, np.arange(erange[i], erange[i+1], estep[i]))
    ept = np.append(ept, np.array(erange[-1]))

    # Debugging, is this needed? is this recorded in scanoutput?
    # Convert energy to bragg angle
    egap = np.array(())
    ebragg = np.array(())
    exgap = np.array(())
    for i in ept:
        eg, eb, ex = energy.forward(i)
        egap = np.append(egap, eg)
        ebragg = np.append(ebragg, eb)
        exgap = np.append(exgap, ex)

    # Register the detectors
    det = [ring_current, sclr1, xbpm2, det_xs]
    # Setup xspress3
    yield from abs_set(det_xs.external_trig, False)
    yield from abs_set(det_xs.settings.acquire_time, acqtime)
    yield from abs_set(det_xs.total_points, len(ept))

    # Setup the scaler
    yield from abs_set(sclr1.preset_time, acqtime)

    # Setup DCM/energy options
    if (harmonic != 1):
        yield from abs_set(energy.harmonic, harmonic)

    # Prepare to peak up DCM at middle scan point
    if (align_at is not None):
        align = True
    if (align is True):
        if (align_at is None):
            align_at = 0.5*(ept[0] + ept[-1])
            print("Aligning at ", align_at)
            yield from abs_set(energy, align_at, wait=True)
        else:
            print("Aligning at ", align_at)
            yield from abs_set(energy, float(align_at), wait=True)
    
    # Peak up DCM at first scan point
    if (align is True):
        yield from peakup_fine(shutter=shutter)

    # Setup the live callbacks
    livecallbacks = []
    # Setup Raw data
    livetableitem = ['energy_energy', 'sclr_i0', 'sclr_it']
    roi_name = 'roi{:02}'.format(roinum[0])
    roi_key = []
    roi_key.append(getattr(det_xs.channel1.rois, roi_name).value.name)
    try:
        roi_key.append(getattr(det_xs.channel2.rois, roi_name).value.name)
        roi_key.append(getattr(det_xs.channel3.rois, roi_name).value.name)
        roi_key.append(getattr(det_xs.channel4.rois, roi_name).value.name)
    except NameError:
        pass
    livetableitem.append(roi_key[0])
    livecallbacks.append(LiveTable(livetableitem))
    liveploty = roi_key[0]
    liveplotx = energy.energy.name
    
    def my_factory(name):
        fig = plt.figure(num=name)
        ax = fig.gca()
        return fig, ax

    # liveplotfig = plt.figure('Raw XANES')
    # livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))
    livecallbacks.append(HackLivePlot(liveploty, x=liveplotx,
                                      fig_factory=partial(my_factory, name='Raw XANES')))

    # Setup normalization
    liveploty = 'sclr_i0'
    i0 = 'sclr_i0'
    # liveplotfig2 = plt.figure('I0')
    # livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig2))
    livecallbacks.append(HackLivePlot(liveploty, x=liveplotx,
                                      fig_factory=partial(my_factory, name='I0')))

    # Setup normalized XANES    
    # livenormfig = plt.figure('Normalized XANES')
    # livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0, fig=livenormfig))
    livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0,
                                           fig_factory=partial(my_factory, name='Normalized XANES')))


    def after_scan(name, doc):
        if (name != 'stop'):
            print("You must export this scan data manually: xanes_afterscan_plan(doc[-1], <filename>, <roinum>)")
            return
        xanes_afterscan_plan(doc['run_start'], filename, roinum)
        logscan_detailed('xanes')


    def at_scan(name, doc):
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        # Not sure if RE should be here, but not sure what to make it
        # scanrecord.current_type.put(RE.md['scaninfo']['type'])
        scanrecord.current_type.put(scan_md['scaninfo']['type'])
        scanrecord.scanning.put(True)


    def finalize_scan():
        yield from abs_set(energy.move_c2_x, True)
        yield from abs_set(energy.harmonic, 1)
        if (shutter is True):
            yield from mv(shut_b,'Close')
        if (detune is not None):
            yield from abs_set(energy.detune, 0)
        scanrecord.scanning.put(False)


    energy.move(ept[0])
    myscan = list_scan(det, energy, list(ept), per_step=per_step, md=scan_md)
    myscan = finalize_wrapper(myscan, finalize_scan)

    # Open b shutter
    if (shutter is True):
        yield from mv(shut_b, 'Open')

    return (yield from subs_wrapper(myscan, {'all' : livecallbacks,
                                             'stop' : after_scan,
                                             'start' : at_scan}))


def xanes_batch_plan(xypos=[], erange=[], estep=[], acqtime=1.0,
                     waittime=10, peakup_N=2, peakup_E=None):
    """
    Setup a batch XANES scan at multiple points.
    This scan can also run peakup_fine() between points.

    xypos       <list>  A list of points to run XANES scans
    erange      <list>  A list of energy points to send to the XANES plan
    estep       <list>  A list of energy steps to send to the XANES plan
    acqtime     <float> Acquisition time for each data point.
    peakup_N    <int>   Run a peakup every peakup_N scans. Default is no peakup
    peakup_E    <float> The energy to run peakup at. Default is current energy

    """

    # Check positions
    if (xypos == []):
        print('You need to enter positions.')
        return

    # Check erange and estep
    if (erange == []):
        print('You need to enter erange.')
        return
    if (estep == []):
        print('You need to enter estep.')
        return

    # Get current energy and use it for peakup
    if (peakup_E == None):
        peakup_E = energy.position.energy

    # Convert keV to eV
    if (peakup_E < 1000):
        peakup_E = peakup_E * 1000

    # Loop through positions
    N = len(xypos)
    for i in range(N):
        print(f'Moving to:')
        print(f'\tx = {xypos[i][0]}')
        print(f'\ty = {xypos[i][1]}')
        hf_stage.x.move(xypos[i][0])
        hf_stage.y.move(xypos[i][1])
        if (len(xypos[i]) == 3):
            print(f'\tz = {xypos[i][2]}')
            hf_stage.z.move(xypos[i][2])

        # Move above edge and peak up
        if (i % peakup_N == 0):
            yield from mv(energy, peakup_E)
            yield from peakup_fine()

        # Run the energy scan
        yield from xanes_plan(erange=erange, estep=estep, acqtime=acqtime)

        # Wait
        if (i != (N-1)):
            print(f'Scan complete. Waiting {waittime} seconds...')
            bps.sleep(waittime)


def hfxanes_ioc(erange=[], estep=[], acqtime=1.0, samplename='', filename='',
                harmonic=1, detune=0, align=False, align_at=None,
                roinum=1, shutter=True, per_step=None, waittime=0):
    '''
    invokes hf2dxrf repeatedly with parameters provided separately.
        waittime                [sec]       time to wait between scans
        shutter                 [bool]      scan controls shutter
        align                   [bool]      optimize beam location on each scan
        roinum                  [1,2,3]     ROI number for data output

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
        if (thisscan.Eena.get() == 1):
            scanrecord.current_scan.put('Scan {}'.format(Nscan))
            erange = [thisscan.e1s.get(), thisscan.e2s.get(), thisscan.e3s.get(), thisscan.efs.get()]
            estep = [thisscan.e1i.get(), thisscan.e2i.get(), thisscan.e3i.get()]
            waittime = thisscan.Ewait.get()

            xstart = thisscan.p1s.get()
            ystart = thisscan.p2s.get()
            # Move stages to the next point
            yield from mv(hf_stage.x, xstart,
                          hf_stage.y, ystart)
            acqtime = thisscan.acq.get()

            hfxanes_gen = yield from xanes_plan(erange=erange, estep=estep, acqtime=thisscan.acq.get(),
                                                samplename=thisscan.sampname.get(), filename=thisscan.filename.get(),
                                                roinum=int(thisscan.roi.get()), detune=thisscan.detune.get(), shutter=shutter)
            if (len(scanlist) is not 0):
                yield from bps.sleep(waittime)
    scanrecord.current_scan.put('')


def fast_shutter_per_step(detectors, motor, step):
    def move():
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, step, group=grp)
        yield Msg('wait', None, group=grp)

    yield from move()
    # Open and close the fast shutter (Mo Foil) between XANES points
    # Open the shutter
    yield from mv(Mo_shutter, 0)
    yield from bps.sleep(1.0)
    # Step? trigger xspress3
    yield from trigger_and_read(list(detectors) + [motor])
    # Close the shutter
    yield from mv(Mo_shutter, 1)
