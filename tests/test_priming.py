from ophyd.sim import motor1, motor2


RE.clear_suspenders()
#changed the flyer device to be aware of fast vs slow axis in a 2D scan
#should abstract this method to use fast and slow axes, rather than x and y
def scan_and_fly_test(xstart, xstop, xnum, ystart, ystop, ynum, dwell, *,
                 delta=0.002, shutter = True,
                 xmotor=hf_stage.x, ymotor=hf_stage.y,
                 xs=xs, ion=sclr1, align = False,
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
    c2pitch_kill=EpicsSignal("XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Cmd:Kill-Cmd")
    if md is None:
        md = {}
    if delta is None:
        delta=0.002
    yield from abs_set(ymotor, ystart, wait=True) # ready to move
    yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move

    if shutter is True:
        yield from mv(shut_b, 'Open')

    if align == True:
        fly_ps = PeakStats(dcm.c2_pitch.name,i0.name)
        align_scan = scan([sclr1], dcm.c2_pitch, -19.320, -19.360, 41)
        align_scan = bp.subs_wrapper(align_scan,fly_ps)
        yield from align_scan
        yield from abs_set(dcm.c2_pitch,fly_ps.max[0],wait=True)
        #ttime.sleep(10)
        #yield from abs_set(c2pitch_kill, 1)
    else:
        ttime.sleep(0.)

    md = ChainMap(md, {
        'plan_name': 'scan_and_fly',
        'detectors': [zebra.name,xs.name,ion.name],
        'dwell' : dwell,
        'shape' : (xnum,ynum),
        'scaninfo' : {'type': 'XRF_fly', 'raster' : False, 'fast_axis':flying_zebra._fast_axis},
        'scan_params' : [xstart,xstop,xnum,ystart,ystop,ynum,dwell]
        }
    )


    from bluesky.plan_stubs import stage, unstage
    @stage_decorator([xs])
    def fly_each_step(detectors, motor, step, firststep):
        "See http://nsls-ii.github.io/bluesky/plans.html#the-per-step-hook"
        # First, let 'scan' handle the normal y step, including a checkpoint.
        yield from one_1d_step(detectors, motor, step)

        # Now do the x steps.
        v = (xstop - xstart) / (xnum-1) / dwell  # compute "stage speed"
        yield from abs_set(xmotor, xstart - delta, wait=True) # ready to move
        yield from abs_set(xmotor.velocity, v, wait=True)  # set the "stage speed"
        yield from abs_set(xs.hdf5.num_capture, xnum)
        yield from abs_set(xs.settings.num_images, xnum)
        yield from abs_set(ion.nuse_all,xnum)
        # arm the Zebra (start caching x positions)
        yield from kickoff(flying_zebra, xstart=xstart, xstop=xstop, xnum=xnum, dwell=dwell, wait=True)
        yield from abs_set(xs.settings.acquire, 1)  # start acquiring images
        yield from abs_set(ion.erase_start, 1) # arm SIS3820, note that there is a 1 sec delay in setting X into motion
                                               # so the first point *in each row* won't normalize...
        #if firststep == True:
        #    ttime.sleep(0.)
        ttime.sleep(1.5)
        yield from abs_set(xmotor, xstop+1*delta, wait=True)  # move in x
        yield from abs_set(xs.settings.acquire, 0)  # stop acquiring images
        yield from abs_set(ion.stop_all, 1)  # stop acquiring scaler
        yield from complete(flying_zebra)  # tell the Zebra we are done
        yield from collect(flying_zebra)  # extract data from Zebra
        yield from abs_set(xmotor.velocity, 1.,wait=True)  # set the "stage speed"

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
    @subs_decorator([LiveTable([ymotor])])
    @subs_decorator([LiveGrid((ynum, xnum+1), xs.channel1.rois.roi01.value.name,extent=(xstart,xstop,ystop,ystart))])
    @subs_decorator({'start':at_scan})
    @subs_decorator({'stop':finalize_scan})
    @monitor_during_decorator([xs.channel1.rois.roi01.value])  # monitor values from xs
    #@monitor_during_decorator([xs], run=False)  # monitor values from xs
    @stage_decorator([flying_zebra])  # Below, 'scan' stage ymotor.
    @run_decorator(md=md)
    def plan():
        #yield from abs_set(xs.settings.trigger_mode, 'TTL Veto Only')
        yield from abs_set(xs.external_trig, True)
        ystep = 0
        for step in np.linspace(ystart, ystop, ynum):
            scanrecord.time_remaining.put( (ynum - ystep) * ( dwell * xnum + 3.8 ) / 3600.)
            ystep = ystep + 1
            # 'arm' the xs for outputting fly data
            yield from abs_set(xs.hdf5.fly_next, True)
#            print('h5 armed\t',time.time())
            if step == ystart:
                firststep = True
            else:
                firststep = False
            yield from fly_each_step([], ymotor, step, firststep)
#            print('return from step\t',time.time())
        yield from abs_set(xs.external_trig, False)
        yield from abs_set(ion.count_mode, 1)
        if shutter is True:
            yield from mv(shut_b, 'Close')


    return (yield from plan())

def hf2dxrf_test(*, xstart, xnumstep, xstepsize, 
            ystart, ynumstep, ystepsize, 
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            shutter = True, align = False, xmotor = hf_stage.x, ymotor = hf_stage.y,
            acqtime, numrois=1, i0map_show=True, itmap_show=False, record_cryo = False,
            dpc = None, e_tomo=None, struck = True, srecord = None, 
            setenergy=None, u_detune=None, echange_waittime=10,samplename=None):
    '''
    input:
        xstart, xnumstep, xstepsize (float)
        ystart, ynumstep, ystepsize (float)
        acqtime (float): acqusition time to be set for both xspress3 and F460
        numrois (integer): number of ROIs set to display in the live raster scans. This is for display ONLY. 
                           The actualy number of ROIs saved depend on how many are enabled and set in the read_attr
                           However noramlly one cares only the raw XRF spectra which are all saved and will be used for fitting.
        i0map_show (boolean): When set to True, map of the i0 will be displayed in live raster, default is True
        itmap_show (boolean): When set to True, map of the trasnmission diode will be displayed in the live raster, default is True   
        energy (float): set energy, use with caution, hdcm might become misaligned
        u_detune (float): amount of undulator to detune in the unit of keV
    '''

    #record relevant meta data in the Start document, defined in 90-usersetup.py
    xs.external_trig.put(False)

    #setup the detector
    # TODO do this with configure

    if acqtime < 0.001:
        acqtime = 0.001
    if struck == False:
        current_preamp.exp_time.put(acqtime)
    else:
        sclr1.preset_time.put(acqtime)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put((xnumstep+1)*(ynumstep+1))
    
    #saturn.mca.preset_real_time.put(acqtime)
    #saturn.mca.preset_live_time.put(acqtime)

    #hfvlmAD.cam.acquire_time.put(acqtime)

    #for roi_idx in range(numrois):
    #    saturn.read_attrs.append('mca.rois.roi'+str(roi_idx)+'.net_count')
    #    saturn.read_attrs.append('mca.rois.roi'+str(roi_idx)+'.count')
       
    #det = [current_preamp, saturn]        
        
    #gjw
    #det = [xs, hfvlmAD]        
    #gjw



    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
    
    

    
    #setup the plan  
    #outer_product_scan(detectors, *args, pre_run=None, post_run=None)
    #outer_product_scan(detectors, motor1, start1, stop1, num1, motor2, start2, stop2, num2, snake2, pre_run=None, post_run=None)

    if setenergy is not None:
        if u_detune is not None:
            # TODO maybe do this with set
            energy.detune.put(u_detune)
        # TODO fix name shadowing
        print('changing energy to', setenergy)
        yield from bp.abs_set(energy, setenergy, wait=True)
        time.sleep(echange_waittime)
        print('waiting time (s)', echange_waittime)
    

    #TO-DO: implement fast shutter control (open)
    #TO-DO: implement suspender for all shutters in genral start up script


    def finalize_scan(name, doc):
        scanrecord.scanning.put(False)


    det = [xs, sclr1]
    hf2dxrf_scanplan = outer_product_scan(det, ymotor, ystart, ystop, ynumstep+1, xmotor, xstart, xstop, xnumstep+1, True)
    print(hf2dxrf_scanplan)
    scaninfo = yield from hf2dxrf_scanplan
    
    return scaninfo

def prime():
    '''
        From : https://github.com/NSLS-II/ophyd/blob/master/ophyd/areadetector/plugins.py#L854
        Doesn't work for now
    '''
    set_and_wait(xs.hdf5.enable, 1)
    sigs = OrderedDict([(xs.settings.array_callbacks, 1),
                        (xs.settings.image_mode, 'Single'),
                        (xs.settings.trigger_mode, 'Internal'),
                        # just in case tha acquisition time is set very long...
                        (xs.settings.acquire_time , 1),
                        #(xs.settings.acquire_period, 1),
                        #(xs.settings.acquire, 1),
                        ])

    original_vals = {sig: sig.get() for sig in sigs}

    RE(bp.count([xs]))
    for sig, val in sigs.items():
        ttime.sleep(0.1)  # abundance of caution
        set_and_wait(sig, val)

    ttime.sleep(2)  # wait for acquisition

    for sig, val in reversed(list(original_vals.items())):
        ttime.sleep(0.1)
        set_and_wait(sig, val)

'''
    fast axis : xmotor
        xstart, xstop, num : fast axis
'''

xmotor = motor1
xmotor.velocity = Signal(name="xmotor_velocity")
ymotor = motor1
ymotor.velocity = Signal(name="ymotor_velocity")

# fast motor
xstart = 0
xstop = 10
xnum = 1000

# slow motor : dummy values
# ynum will be the number of times to move in slow axis
ystart = 0
ystop= 0
ynum = 1

# dwell time?
dwell = .1

plan_test=scan_and_fly_test(xstart, xstop, xnum, ystart, ystop, ynum, dwell,
                  delta=.002, xmotor=xmotor, ymotor=ymotor, xs=xs, ion=sclr1, align=False,
                  flying_zebra=flying_zebra, shutter=False)


xmotor = motor1
xmotor.velocity = Signal(name="xmotor_velocity")
ymotor = motor1
ymotor.velocity = Signal(name="ymotor_velocity")

# fast motor
xstart = 0
xstop = 10
xnumstep= 1
xstepsize = .1

# slow motor : dummy values
# ynum will be the number of times to move in slow axis
ystart = 0
ynumstep = 1
ystepsize = .1

# dwell time?
dwell = .1
acqtime = .1

prime_plan = hf2dxrf_test(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
            ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, 
            shutter = False, align = False, xmotor = motor1, ymotor = motor2,
            acqtime=acqtime, numrois=1, i0map_show=False, itmap_show=False, record_cryo = False,
            dpc = None, e_tomo=None, struck = True, srecord = None, 
            setenergy=None, u_detune=None, echange_waittime=10,samplename=None)
