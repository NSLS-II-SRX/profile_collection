# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:43:13 2016

@author: xf05id1
"""

#for tomography
from bluesky.plans import scan_nd, subs_wrapper
from cycler import cycler

def tomo_xrf_proj(xcen, zcen, hstepsize, hnumstep,
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
    
def tomo_xrf(thetastart = -90, thetastop = 90, thetanumstep = 31):
    dets = []
    theta = np.linspace(thetastart, thetastop, thetanumstep)
    