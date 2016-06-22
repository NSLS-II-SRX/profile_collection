# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:43:13 2016

@author: xf05id1
"""

#for tomography
from bluesky.plans import scan_nd, subs_wrapper
from cycler import cycler

def tomo_xrf_proj_realmotor(xcen, zcen, hstepsize, hnumstep,
                  ycen, ystepsize, ynumstep,
                  dets = []):
    '''
    collect an XRF 'projection' map at the current angle
    zcen should be defined as the position when the sample is in focus at zero degree; if it is not given, the program should take the current z position
    '''
    theta = tomo_stage.theta.position
    
    #horizontal axes
    x_motor = tomo_stage.finex_top
    z_motor = tomo_stage.finez_top
    
    #vertical axis
    y_motor = tomo_stage.finey_top
    
    #stepsize setup    
    xstepsize = hstepsize * numpy.cos(numpy.deg2rad(theta))
    zstepsize = hstepsize * numpy.sin(numpy.deg2rad(theta))
        
    #start and end point setup
    
    xstart = xcen - xstepsize * hnumstep/2
    xstop  = xcen + xstepsize * hnumstep/2    

    zstart = zcen - zstepsize * hnumstep/2
    zstop   = zcen + zstepsize * hnumstep/2    
    
    ystart = ycen - ystepsize * ynumstep/2
    ystop  = ycen + ystepsize * ynumstep/2
    
    xlist = numpy.linspace(xstart, xstop, hnumstep+1) #some theta dependent function    
    zlist = numpy.linspace(zstart, zstop, hnumstep+1)
    
    ylist = numpy.linspace(ystart, ystop, ynumstep+1)
    
    xz_cycler = cycler(x_motor, xlist) + cycler(z_motor, zlist)
    yxz_cycler = cycler(y_motor, ylist) * xz_cycler
    
    # The scan_nd plan expects a list of detectors and a cycler.
    plan = scan_nd(dets, yxz_cycler)
    # Optionally, add subscritpions.

    #TO-DO: need to figure out how to add LiveRaster with the new x/z axis 
    plan = subs_wrapper(plan, [LiveTable([x_motor, y_motor, z_motor])])
#                                         LiveMesh(...)]                      
    scaninfo = yield from plan
    return scaninfo

def add_tomo_md(current_md = None):
    if current_md is None:
        md_with_tomo_stage_info = {}        
    else:
        md_with_tomo_stage_info = current_md
    
    md_with_tomo_stage_info['tomo_stage'] = {'x':  tomo_stage.x.position, 
                                'y':  tomo_stage.y.position,
                                'z':  tomo_stage.z.position,
                                'finex_top': tomo_stage.finex_top.position,
                                'finez_top': tomo_stage.finez_top.position,
                                'finex_bot': tomo_stage.finex_bot.position,
                                'finez_bot': tomo_stage.finez_bot.position,
                                'theta': tomo_stage.theta.position}

                               
    md_with_tomo_stage_info['tomo_lab'] = {'lab_x':  tomo_lab.lab_x.position, 
                              'lab_z':  tomo_lab.lab_z.position}

    return md_with_tomo_stage_info


def tomo_xrf_proj(*, xstart, xnumstep, xstepsize, 
            ystart, ynumstep, ystepsize, 
            #lab_z_setpt = 0,  #default lab_z initilization point, user can change it
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime, numrois=1, i0map_show=True, itmap_show=True,
            energy=None, u_detune=None):
    '''
    collect an XRF 'projection' map at the current angle
    note that the x-axis is in the laboraotry frame
    see motion definition in 06-endstation_pseudomotor_tomo.py
    it should be checked at both 0 and 90/-90 degree. 
   '''
    #record relevant meta data in the Start document, defined in 90-usersetup.py
    md = get_stock_md()
    md = add_tomo_md(current_md = md)

    #setup the detector
    # TODO do this with configure
    current_preamp.exp_time.put(acqtime)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put((xnumstep+1)*(ynumstep+1))
         
    det = [current_preamp, xs, tomo_stage.finey_top, tomo_lab]        

    #setup the live callbacks
    livecallbacks = []
    
    livetableitem = [tomo_stage.finex_top, tomo_stage.finey_top, tomo_lab, 'current_preamp_ch0', 'current_preamp_ch2']

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
    print('xstop = '+str(xstop))  
    print('ystop = '+str(ystop)) 
    
    
    for roi_idx in range(numrois):
        roi_name = 'roi{:02}'.format(roi_idx+1)
        
        roi_key = getattr(xs.channel1.rois, roi_name).value.name
        livetableitem.append(roi_key)

        colormap = 'jet' #previous set = 'viridis'

        roimap = LiveRaster((ynumstep+1, xnumstep+1), roi_key, clim=None, cmap='jet', 
                            xlabel='x (um)', ylabel='y (um)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(roimap)


    if i0map_show is True:
        i0map = LiveRaster((ynumstep+1, xnumstep+1), 'current_preamp_ch2', clim=None, cmap='jet', 
                        xlabel='x (um)', ylabel='y (um)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(i0map)

    if itmap_show is True:
        itmap = LiveRaster((ynumstep+1, xnumstep+1), 'current_preamp_ch0', clim=None, cmap='jet', 
                        xlabel='x (um)', ylabel='y (um)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(itmap)

    livecallbacks.append(LiveTable(livetableitem)) 

    
    #setup the plan  
    #OuterProductAbsScanPlan(detectors, *args, pre_run=None, post_run=None)
    #OuterProductAbsScanPlan(detectors, motor1, start1, stop1, num1, motor2, start2, stop2, num2, snake2, pre_run=None, post_run=None)

    if energy is not None:
        if u_detune is not None:
            # TODO maybe do this with set
            energy.detune.put(u_detune)
        # TODO fix name shadowing
        yield from bp.abs_set(energy, energy, wait=True)
    

    #TO-DO: implement fast shutter control (open)
    #TO-DO: implement suspender for all shutters in genral start up script
    
#    shut_b.open_cmd.put(1)
#    while (shut_b.close_status.get() == 1):
#        epics.poll(.5)
#        shut_b.open_cmd.put(1)    
    
    tomo_xrf_proj_plan = OuterProductAbsScanPlan(det, tomo_stage.finey_top, ystart, ystop, ynumstep+1, tomo_lab.lab_x, xstart, xstop, xnumstep+1, True, md=md)
    tomo_xrf_proj_plan = bp.subs_wrapper(tomo_xrf_proj_plan, livecallbacks)
    tomo_xrf_proj_ren = yield from tomo_xrf_proj_plan

    #TO-DO: implement fast shutter control (close)    
#    shut_b.close_cmd.put(1)
#    while (shut_b.close_status.get() == 0):
#        epics.poll(.5)
#        shut_b.close_cmd.put(1)

    #write to scan log    
    logscan('2dxrf_hr_topy_labx')    
    
    return tomo_xrf_proj_ren
   
    
def tomo_xrf(*, xstart, xnumstep, xstepsize, 
            ystart, ynumstep, ystepsize, 
            thetastart = 80, thetastop = 90, numproj = 3,
            lab_z_setpt = 0,
            acqtime, numrois=1, i0map_show=True, itmap_show=True,
            energy=None, u_detune=None
            ):
                
    theta_traj = np.linspace(thetastart, thetastop, numproj)
    tomo_scan_output = []

    for theta_setpt in theta_traj:
        print('current angle')
        print(tomo_stage.theta.position)
        print('move angle to '+str(theta_setpt))        
        tomo_theta_set_gen = yield from list_scan([tomo_stage], tomo_stage.theta, [theta_setpt])
        print('angle in position')
        
        print('initilize tomo_lab.lab_z to the aligned point', lab_z_setpt)
        tomo_lab_z_initi = yield from list_scan([], tomo_lab.lab_z, [lab_z_setpt])
        print('tomo_lab.lab_z in set point position; its position should remain unchanged')

        print('start running tomo_xrf_proj to collect xrf projection for the current angle')
        tomo_xrf_gen = yield from tomo_xrf_proj(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
            ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, 
            acqtime=acqtime, numrois=numrois, i0map_show=i0map_show, itmap_show=itmap_show,
            energy=energy, u_detune=u_detune)
        tomo_scan_output.append(tomo_xrf_gen)
        
        print('done running tomo_xrf_proj')
    
    print('tomography data collection completed')
    return tomo_scan_output