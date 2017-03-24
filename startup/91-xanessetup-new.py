from bluesky.plans import AbsListScanPlan
import bluesky.plans as bp
import scanoutput
import numpy
import time
from epics import PV
from databroker import get_table
import collections

def xanes_afterscan_plan(scanid, filename, roinum):
    #print(scanid,filename,roinum)
    # custom header list 
    headeritem = [] 
    # load header for our scan
    h=db[scanid]

    # construct basic header information
    userheaderitem = {}
    userheaderitem['uid'] = h.start['uid']
    userheaderitem['sample.name'] = h.start['sample']['name']
    userheaderitem['initial_sample_position.hf_stage.x'] = h.start['initial_sample_position']['hf_stage_x']
    userheaderitem['initial_sample_position.hf_stage.y'] = h.start['initial_sample_position']['hf_stage_y']
    userheaderitem['hfm.y'] = h.start['hfm']['y']
    userheaderitem['hfm.bend'] = h.start['hfm']['bend']

    # create columns for data file
    columnitem = ['energy_energy', 'energy_u_gap_readback', 'energy_bragg', 'energy_c2_x']
    # include I_0 and I_t from either the SRS or Oxford preamp, raise expection
    # if neither present
    if 'sclr1' in h.start['detectors']:
        columnitem = columnitem + ['sclr_i0', 'sclr_it']
    elif 'current_preamp' in h.start['detectors']:
        columnitem = columnitem + ['current_preamp_ch0', 'current_preamp_ch2']
    else:
        raise KeyError("Neither SRS nor Oxford preamplifier found in data!")
    # include fluorescence data if present, allow multiple rois
    if 'xs' in h.start['detectors']:
        if type(roinum) is not list:
            roinum = [roinum]
        for i in roinum:
            roi_name = 'roi{:02}'.format(i)
            roi_key = []
            roi_key.append(getattr(xs.channel1.rois, roi_name).value.name)
            roi_key.append(getattr(xs.channel2.rois, roi_name).value.name)
            roi_key.append(getattr(xs.channel3.rois, roi_name).value.name)

        [ columnitem.append(roi) for roi in roi_key ]
    # construct user convenience columns allowing prescaling of ion chamber, diode and
    # fluorescence detector data
    usercolumnitem = {}
    datatablenames = []
    
    # assume that we are using either the SRS or Oxford preamp for both I_0 and I_T
    if 'xs' in h.start['detectors']:
        datatablenames = datatablenames + [ str(roi) for roi in roi_key]
    if 'sclr1' in  h.start['detectors']:
        datatablenames = datatablenames + ['sclr_i0', 'sclr_it']
        datatable = get_table(h, datatablenames, stream_name='primary')        
        i0_array = numpy.array(datatable['sclr_i0'])
        it_array = numpy.array(datatable['sclr_it'])
    elif 'current_preamp' in h.start['detectors']:
        datatablenames = datatablenames + ['current_preamp_ch2', 'current_preamp_ch0']
        datatable = get_table(h, datatablenames, stream_name='primary')        
        i0_array = numpy.array(datatable['current_preamp_ch2'])
        it_array = numpy.array(datatable['current_preamp_ch0'])
    else:
        raise KeyError
    # calculate sums for xspress3 channels of interest
    if 'xs' in h.start['detectors']:
        for i in roinum:
            roi_name = 'roi{:02}'.format(i)
            roisum = datatable[getattr(xs.channel1.rois, roi_name).value.name] 
            #roisum.index += -1
            roisum = roisum + datatable[getattr(xs.channel2.rois, roi_name).value.name] 
            roisum = roisum + datatable[getattr(xs.channel3.rois, roi_name).value.name] 
            usercolumnitem['If-{:02}'.format(i)] = roisum
            usercolumnitem['If-{:02}'.format(i)].round(0)

    scanoutput.textout(scan = scanid, header = headeritem, 
                        userheader = userheaderitem, column = columnitem, 
                        usercolumn = usercolumnitem, 
                        usercolumnname = usercolumnitem.keys(), 
                        output = False, filename_add = filename) 

def xanes_plan(erange = [], estep = [],  
            harmonic = None, correct_c2_x=True, correct_c1_r = False, detune = None,
            acqtime=1., roinum=1, delaytime = 0.00, struck=True, fluor = True,
            samplename = '', filename = '', shutter = True, align = False):
                
    '''
    erange (list of floats): energy ranges for XANES in eV, e.g. erange = [7112-50, 7112-20, 7112+50, 7112+120]
    estep  (list of floats): energy step size for each energy range in eV, e.g. estep = [2, 1, 5]
    
    harmonic (None or odd integer): when set to None, use the highest harmonic achievable automatically. 
                                    when set to an odd integer, force the XANES scan to use that harmonic
    correct_c2_x (boolean or float): when True, automatically correct the c2x 
                                     when False, c2x will not be moved during the XANES scan
    correct_c1_r (False or float): when False, c1r will not be moved during a XANES scan
                                   when set to a float, c1r will be set to that value before a XANES scan but will remain the same during the whole scan
    detune:  add this value to the gap of the undulator to reduce flux [mm]

    acqtime (float): acqusition time to be set for both xspress3 and preamplifier                                   
    roinum: select the roi to be used to calculate the XANES spectrum
    delaytime:  reduce acquisition time of F460 by this value [sec]
    struck:  Use the SRS and Struck scaler for the ion chamber and diode.  Set to False to use the F460.
    fluorescence:  indicate the presence of fluorescence data [bool]

    samplename (string): sample name to be saved in the scan metadata
    filename (string): filename to be added to the scan id as the text output filename

    shutter:  instruct the scan to control the B shutter [bool]
    align:  control the tuning of the DCM pointing before each XANES scan [bool]
    '''                                
                
    ept = numpy.array([])
    det = []
    filename=filename
    last_time_pt = time.time()
    ringbuf = collections.deque(maxlen=10)

    #make sure user provided correct input
    if erange is []:
        raise AttributeError("An energy range must be provided in a list by means of the 'erange' keyword.")
    if estep is []:
        raise AttributeError("A list of energy steps must be provided by means of the 'esteps' keyword.")
    if (not isinstance(erange,list)) or (not isinstance(estep,list)):
        raise TypeError("The keywords 'estep' and 'erange' must be lists.")
    if len(erange)-len(estep) is not 1:
        raise ValueError("The 'erange' and 'estep' lists are inconsistent;"\
                         +'c.f., erange = [7000, 7100, 7150, 7500], estep = [2, 0.5, 5] ')
    if type(roinum) is not list:
        roinum = [roinum]
    if detune is not None:
        yield from abs_set(energy.detune,detune)

    #record relevant meta data in the Start document, defined in 90-usersetup.py
    metadata_record()
    #add user meta data
    gs.RE.md['sample']  = {'name': samplename}
   
    #convert erange and estep to numpy array
    erange = numpy.array(erange)
    estep = numpy.array(estep)
    #calculation for the energy points        
    for i in range(len(estep)):
        ept = numpy.append(ept, numpy.arange(erange[i], erange[i+1], estep[i]))
    ept = numpy.append(ept, numpy.array(erange[-1]))
    #register the detectors
    det = [ring_current]
    if struck == True:
        det.append(sclr1)
    else:
        det.append(current_preamp)
    if fluor == True:
        det.append(xs)
        #setup xspress3
        yield from abs_set(xs.settings.acquire_time,acqtime)
        yield from abs_set(xs.total_points,len(ept))

    #setup the preamp
    if struck == True:
        yield from abs_set(sclr1.preset_time,acqtime)
    else:
        yield from abs_set(current_preamp.exp_time,acqtime-delaytime)
    #setup dcm/energy options
    if correct_c2_x is False:
        yield from abs_set(energy.move_c2_x,False)
    if correct_c1_r is not False:
        yield from abs_set(dcm.c1_roll,correct_c1_r)
    if harmonic is not None:        
        yield from abs_set(energy.harmonic,harmonic)
    energy.u_gap.corrfunc_dis.put(1)
    #prepare to peak up DCM at first scan point
    if align is True:
        yield from abs_set(energy, ept[0], wait = True)
    #open b shutter
    if shutter is True:
        #shut_b.open()
        yield from bp.mv(shut_b, 'Open')
        #yield from abs_set(shut_b,1,wait=True)
    #peak up DCM at first scan point
    if align is True:
        ps = PeakStats(dcm.c2_pitch.name,'sclr_i0')
        e_value = energy.energy.get()[1]
#        if e_value < 10.:
#            yield from abs_set(sclr1.preset_time,0.1, wait = True)
#            peakup = scan([sclr1], dcm.c2_pitch, -19.335, -19.305, 31)
#        else:
#            yield from abs_set(sclr1.preset_time,1., wait = True)
#            peakup = scan([sclr1], dcm.c2_pitch, -19.355, -19.320, 36)
        if e_value < 10.:
            sclr1.preset_time.put(0.1)
        else:
            sclr1.preset_time.put(1.)
        peakup = scan([sclr1], dcm.c2_pitch, -19.250, -19.200, 51)
        peakup = bp.subs_wrapper(peakup,ps)
        yield from peakup
        yield from abs_set(dcm.c2_pitch, ps.cen, wait = True)

    #setup the live callbacks
    livecallbacks = []    
    livetableitem = ['energy_energy']
    if struck == True:
        livetableitem = livetableitem + ['sclr_i0', 'sclr_it']  
    else:
        livetableitem = livetableitem + ['current_preamp_ch0', 'current_preamp_ch2']  
    if fluor == True:
        roi_name = 'roi{:02}'.format(roinum[0])
        roi_key = []
        roi_key.append(getattr(xs.channel1.rois, roi_name).value.name)
        roi_key.append(getattr(xs.channel2.rois, roi_name).value.name)
        roi_key.append(getattr(xs.channel3.rois, roi_name).value.name)
        livetableitem.append(roi_key[0])    
        livecallbacks.append(LiveTable(livetableitem))
        liveploty = roi_key[0]
        liveplotx = energy.energy.name
        liveplotfig = plt.figure('raw xanes')
    elif struck == True:
        liveploty = 'sclr_it' 
        liveplotx = energy.energy.name
        liveplotfig = plt.figure('raw xanes')
    
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))
    #livecallbacks.append(LivePlot(liveploty, x=liveplotx, ax=plt.gca(title='raw xanes')))
        
    if struck == True:
        liveploty = 'sclr_i0'
        i0 = 'sclr_i0'
    else:
        liveploty = 'current_preamp_ch2'
        i0 = 'current_preamp_ch2'
    liveplotfig2 = plt.figure('i0')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig2))
    #livecallbacks.append(LivePlot(liveploty, x=liveplotx, ax=plt.gca(title='incident intensity')))
    livenormfig = plt.figure('normalized xanes')    
    if fluor == True:
        livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0, fig=livenormfig))  
        #livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0, ax=plt.gca(title='normalized xanes')))  
    else:
        livecallbacks.append(NormalizeLivePlot('sclr_it', x=liveplotx, norm_key = i0, fig=livenormfig))  
        #livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0, ax=plt.gca(title='normalized xanes')))  
    def after_scan(name, doc):
        if name != 'stop':
            print("You must export this scan data manually: xanes_afterscan_plan(doc[-1], <filename>, <roinum>)")
            return
        xanes_afterscan_plan(doc['run_start'], filename, roinum)
        logscan('xanes')

    def finalize_scan():
        yield from abs_set(energy.u_gap.corrfunc_en,1)
        yield from abs_set(energy.move_c2_x, True)
        yield from abs_set(energy.harmonic, None)
        if shutter == True:
            yield from bp.mv(shut_b,'Close')
        if detune is not None:
            energy.detune.put(0)

    myscan = AbsListScanPlan(det, energy, list(ept))
    myscan = bp.finalize_wrapper(myscan,finalize_scan)

    return (yield from bp.subs_wrapper(myscan,{'all':livecallbacks,'stop':after_scan})) 

#not up to date, ignore for now
def xanes_batch_plan(xylist=[], waittime = [2], 
                    samplename = None, filename = None,
                    erange = [], estep = [], struck = True, align = False, 
                    harmonic = None, correct_c2_x=True, delaytime=0.0, detune = None,            
                    acqtime=None, roinum=1, shutter = True, fluor = True
                    ):
                        
    '''
    Running batch XANES scans on different locations, defined as in xylist.
    input: 
        xylist (list of x,y positions in float): pairs of x, y positions on which XANES scans will be collected
            E.g. xylist = [[10.4, 20.4], [10.5, 20.8]] 
        waitime (list of float): wait time between scans, if not specified, 2 seconds will be used
            E.g. waittime = [10] #10 sec. wait time will be used between all scans
            E.g. waititme = [10, 20] #10 sec. will be used between 1st and 2nd scans; 20 sec. will be used after the 2nd scan. The number of scans need to match with the number of waittime listed
        samplename (list of string): list of sample names to be used.
            If with one component, all scans will be set to the same sample name
            If with more than one component, the lenth of the list must match the lenth of the xylist. The sample name will then be assigned 1-1.
            E.g. samplename = ['sample1']: all scans will have the same sample name
            E.g. samplename = ['sample1', 'sample2']: two points in the xylist will have different sample names
        filename (list of string): list of file names to be used
            same rules as in sample name is used.
            E.g. filename = ['sample1']: all scans will have the same file name
            E.g. filename = ['sample1', 'sample2']: two points in the xylist will have different file names attached to their scan ids.
                       
        other inputs are same as in the xanes funciton.
    '''

    if type(xylist) is not list:
        raise AttributeError("xylist must be a python list, e.g., [ [x0,y0], [x1,y1] ]")
    
    for pt_num, position in enumerate(xylist):
        #move stages to the next point
        yield from abs_set(hf_stage.x, position[0]) 
        yield from abs_set(hf_stage.y, position[1])

        #check bragg temperature before start the scan
#        if dcm.temp_pitch.get() > 110:
#            print('bragg temperature too high, wait ' + str(bragg_waittime) + ' s.')            
#            time.sleep(bragg_waittime)
        
        if samplename is None:
            pt_samplename = ''
        else:
            if type(samplename) is not list:
                samplename = [samplename]
            if len(samplename) is 1:
                pt_samplename = samplename[0]                
            elif len(samplename) is not len(xylist):
                err_msg = 'number of samplename is different from the number of points'
                raise ValueError(err_msg)            
            else:
                pt_samplename = samplename[pt_num]

        if filename is None:
            pt_filename = ''
        else:
            if type(filename) is not list:
                filename = [filename]
            if len(filename) is 1:
                pt_filename = filename[0]     
            elif len(filename) is not len(xylist):
                err_msg = 'number of filename is different from the number of points'
                raise ValueError(err_msg)
            else:
                pt_filename = filename[pt_num]
                
        if type(waittime) is not list:
            waittime = [waittime]
        if len(waittime) is not len(xylist) and len(waittime) is not 1:
            err_msg = 'number of waittime is different from the number of points'
            raise ValueError(err_msg)
        
        yield from xanes_plan(erange = erange, estep = estep,  
            harmonic = harmonic, correct_c2_x= correct_c2_x, detune = detune,              
            acqtime = acqtime, roinum = roinum, align = align, 
            delaytime=delaytime, samplename = pt_samplename, 
            filename = pt_filename, struck=struck, fluor=fluor,
            shutter=shutter)
            
        #wait for specified time period in sec.
        if len(waittime) is 1:
            time.sleep(waittime[0])
        elif len(xylist) is len(waittime):
            print('waiting: ',waittime[pt_num])
#            time.sleep(waittime[pt_num])
            try:
                time.sleep(waittime[pt_num])
            except KeyboardInterrupt:
                pass

def hfxanes_ioc(waittime = None, samplename = None, filename = None,
                erange = [], estep = [], struck = True, align = False,
                harmonic = None, correct_c2_x= True, delaytime=0.0, detune = None,
                acqtime=None, roinum=1, shutter = True, fluor = True, 
                ):
    '''
    invokes hf2dxrf repeatedly with parameters provided separately.
        waittime                [sec]       time to wait between scans
        shutter                 [bool]      scan controls shutter
        struck                  [bool]      use scaler for I_0
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
        if thisscan.Eena.get() == 1:
            scanrecord.current_scan.put('Scan {}'.format(Nscan))
            erange = [thisscan.e1s.get(),thisscan.e2s.get(),thisscan.e3s.get(),thisscan.efs.get()]
            estep = [thisscan.e1i.get(), thisscan.e2i.get(), thisscan.e3i.get()]
            waittime = thisscan.Ewait.get()

            xstart = thisscan.p1s.get()
            ystart = thisscan.p2s.get()
            #move stages to the next point
            yield from abs_set(hf_stage.x, xstart, wait=True) 
            yield from abs_set(hf_stage.y, ystart, wait=True)
#            print(xstart,ystart)
            acqtime = thisscan.acq.get()

            hfxanes_gen = yield from xanes_plan(erange = erange, estep = estep,  
                harmonic = harmonic, correct_c2_x= correct_c2_x,              
                acqtime = thisscan.acq.get(), roinum = int(thisscan.roi.get()), align = align, 
                delaytime=delaytime, samplename = thisscan.sampname.get(), 
                filename = thisscan.filename.get(), struck=struck, fluor=fluor, detune=thisscan.detune.get(),
                shutter=shutter)
            if len(scanlist) is not 0:
                time.sleep(waittime)
#            print(erange, estep, thisscan.acq.get(), thisscan.roi.get(), thisscan.sampname.get(), thisscan.filename.get())
    scanrecord.current_scan.put('')

