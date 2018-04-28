from functools import partial

# here is what we used to figure out how to prime the detector before we
# stripped it down
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


    det = [xs]
    hf2dxrf_scanplan = outer_product_scan(det, ymotor, ystart, ystop, ynumstep+1, xmotor, xstart, xstop, xnumstep+1, True)
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

xmotor = motor1
xmotor.velocity = Signal(name="xmotor_velocity")
ymotor = motor1
ymotor.velocity = Signal(name="ymotor_velocity")

# fast motor
xstart = 0
xnumstep= 100
xstepsize = .01

# slow motor : dummy values
# ynum will be the number of times to move in slow axis
ystart = 0
ynumstep = 0
ystepsize = .1

# dwell time?
dwell = .1
acqtime = .001

prime_plan = partial(hf2dxrf_test, xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
            ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, 
            shutter = False, align = False, xmotor = motor1, ymotor = motor2,
            acqtime=acqtime, numrois=1, i0map_show=False, itmap_show=False, record_cryo = False,
            dpc = None, e_tomo=None, struck = True, srecord = None, 
            setenergy=None, u_detune=None, echange_waittime=10,samplename=None)

