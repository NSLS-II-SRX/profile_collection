# -*- coding: utf-8 -*-
"""
set up for 2D XRF scan for HF mode

Created on Fri Feb 19 12:16:52 2016
Modified on Wed Wed 02 14:14 to comment out the saturn detector which is not in use

@author: xf05id1
"""

#TO-DOs:
    #1. add suspender once it's fixed
    #2. put snake back to the scan once it's fixed
    #3. check the x/y are correct
    #4. put x/y axes onto the live plot
    #5. add i0 into the default figure

from bluesky.plans import OuterProductAbsScanPlan
from bluesky.callbacks import LiveRaster
import matplotlib
import time
import epics
import os
import numpy

#matplotlib.pyplot.ticklabel_format(style='plain')

def hf2dxrf(xstart=None, xnumstep=None, xstepsize=None, 
            ystart=None, ynumstep=None, ystepsize=None, 
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime=None, numrois=1, i0map_show=True, itmap_show = True,
            energy = None, u_detune = None,
            ):

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

    #make sure user provided correct input

    if xstart is None:
        raise Exception('xstart = None, must specify an xstart position')
    if xnumstep is None:
        raise Exception('xnumstep = None, must specify an xnumstep position')
    if xstepsize is None:
        raise Exception('xstepsize = None, must specify an xstepsize position')
    if ystart is None:
        raise Exception('ystart = None, must specify an ystart position')
    if ynumstep is None:
        raise Exception('ynumstep = None, must specify an ynumstep position')
    if ystepsize is None:
        raise Exception('ystepsize = None, must specify an ystepsize position')
    if acqtime is None:
        raise Exception('acqtime = None, must specify an acqtime position')

    #record relevant meta data in the Start document, defined in 90-usersetup.py
    metadata_record()

    #setup the detector
    current_preamp.exp_time.put(acqtime)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put((xnumstep+1)*(ynumstep+1))
    
    #saturn.mca.preset_real_time.put(acqtime)
    #saturn.mca.preset_live_time.put(acqtime)

    #for roi_idx in range(numrois):
    #    saturn.read_attrs.append('mca.rois.roi'+str(roi_idx)+'.net_count')
    #    saturn.read_attrs.append('mca.rois.roi'+str(roi_idx)+'.count')
       
    #det = [current_preamp, saturn]        
    det = [current_preamp, xs]        


    #setup the live callbacks
    livecallbacks = []
    
    livetableitem = [hf_stage.x, hf_stage.y, 'current_preamp_ch0', 'current_preamp_ch2']

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
    print('xstop = '+str(xstop))  
    print('ystop = '+str(ystop)) 
    
    
    for roi_idx in range(numrois):
        roi_name = 'roi{:02}'.format(roi_idx+1)
        
        roi_key = getattr(xs.channel1.rois, roi_name).value.name
        livetableitem.append(roi_key)
        
    #    livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_net_count')
    #    livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_count')
    #    #roimap = LiveRaster((xnumstep, ynumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_net_count', clim=None, cmap='viridis', xlabel='x', ylabel='y', extent=None)
        colormap = 'jet' #previous set = 'viridis'
    #    roimap = LiveRaster((ynumstep, xnumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_count', clim=None, cmap='jet', 
    #                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])

        roimap = LiveRaster((ynumstep+1, xnumstep+1), roi_key, clim=None, cmap='jet', 
                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(roimap)


    if i0map_show is True:
        i0map = LiveRaster((ynumstep+1, xnumstep+1), 'current_preamp_ch2', clim=None, cmap='jet', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(i0map)

    if itmap_show is True:
        itmap = LiveRaster((ynumstep+1, xnumstep+1), 'current_preamp_ch0', clim=None, cmap='jet', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(itmap)


#    commented out liveTable in 2D scan for now until the prolonged time issue is resolved
    livecallbacks.append(LiveTable(livetableitem)) 

    
    #setup the plan  
    #OuterProductAbsScanPlan(detectors, *args, pre_run=None, post_run=None)
    #OuterProductAbsScanPlan(detectors, motor1, start1, stop1, num1, motor2, start2, stop2, num2, snake2, pre_run=None, post_run=None)

    if energy is not None:
        if u_detune is not None:
            energy.detune.put(u_detune)
        energy.set(energy)
        time.sleep(5)
    
    shut_b.open_cmd.put(1)
    while (shut_b.close_status.get() == 1):
        epics.poll(.5)
        shut_b.open_cmd.put(1)    
    
    hf2dxrf_scanplan = OuterProductAbsScanPlan(det, hf_stage.y, ystart, ystop, ynumstep+1, hf_stage.x, xstart, xstop, xnumstep+1, True)
    scaninfo = gs.RE(hf2dxrf_scanplan, livecallbacks, raise_if_interrupted=True)
    
    shut_b.close_cmd.put(1)
    while (shut_b.close_status.get() == 0):
        epics.poll(.5)
        shut_b.close_cmd.put(1)

    #write to scan log    
    logscan('2dxrf')    
    
    return scaninfo
    
    


def hf2dxrf_estack(batch_dir = None, batch_filename = None,
            erange = [], estep = [],
            energy_pt = None,  waittime = 5, energy_waittime = 5,
            harmonic = None, correct_c2_x=True, correct_c1_r = False, 
          #same parameters as in hd2dxrf
            xstart=None, xnumstep=None, xstepsize=None, 
            ystart=None, ynumstep=None, ystepsize=None, 
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime=None, numrois=1, i0map_show=False, itmap_show = False):
    '''
    A function under development that will provide the ability to do xrf stack imaging.
    Warning: this function is not complete and has not been tested. 
    '''
    if erange is []:
        raise Exception('erange = [], must specify energy ranges')
    if estep is []:
        raise Exception('estep = [], must specify energy step sizes')
    if len(erange)-len(estep) is not 1:
        raise Exception('must specify erange and estep correctly.'\
                         +'e.g. erange = [7000, 7100, 7150, 7500], estep = [2, 0.5, 5] ')
                         
    if acqtime is None:
        raise Exception('acqtime = None, must specify an acqtime position')
        
    #initializing the batch log file
    batchlogfile = batch_dir+'/logfile_'+batch_filename
    batchlogf = open(batchlogfile, 'w')
    batchlogf.write('energylist = []\n')
    batchlogf.write('scanlist = []\n')    
    batchlogf.close()

    #convert erange and estep to numpy array
    erange = numpy.array(erange)
    estep = numpy.array(estep)

    #calculation for the energy points        
    ept = numpy.array([])
    for i in range(len(estep)):
        ept = numpy.append(ept, numpy.arange(erange[i], erange[i+1], estep[i]))
    ept = numpy.append(ept, numpy.array(erange[-1]))
    ept = ept/1000

    if correct_c2_x is False:
        energy.move_c2_x.put(False)
    elif correct_c2_x is True:
    #this line sets the current xoffset to the energy axis, might want to see if that's what we want to do in the long run
        energy._xoffset = energy.crystal_gap()
    else:
    #user can set correct_c2x to a float value and force the energy._xoffset to that value, useful when changing between edges in a batch
        energy._xoffset = correct_c2_x
        
    if correct_c1_r is not False:
        dcm.c1_roll.set(correct_c1_r)
        
    if harmonic is not None:        
        energy.harmonic.put(harmonic)
                                
    for energy_setpt in ept:
        energy.set(energy_setpt)  
        time.sleep(energy_waittime)
        
        batchlogf = open(batchlogfile, 'a')
        batchlogf.write('energylist.append('+str(energy_setpt)+')\n')
        batchlogf.close()
        #run hf2dxrf scans
        hf2dxrf(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
            ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize,  i0map_show=i0map_show, itmap_show = itmap_show,
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime=acqtime, numrois=numrois)
        time.sleep(waittime)
        batchlogf = open(batchlogfile, 'a')
        batchlogf.write('scanlist.append('+ str(db[-1].start['scan_id'])+')\n')
        batchlogf.close()
    
    #clean up when the scan is done    
    energy.move_c2_x.put(True)
    energy.harmonic.put(None)

def hf2dxrf_repeat(num_scans = None, waittime = 10,
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
        hf2dxrf(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
                ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, 
                acqtime=acqtime, numrois=numrois, i0map_show=i0map_show, itmap_show = itmap_show)
        time.sleep(waittime)
        
def hf2dxrf_xybatch(batch_dir = None, batch_filename = None, waittime = 5, repeat = 1):
    '''
    This function will load from the batch text input file and run the 2D XRF according to the set points in the text file.
    input:
        batch_dir (string): directory for the input batch file
        batch_filename (string): text file name that defines the set points for batch scans
        repeat (integer): number to repeat the scans in the batch; repeat = 1 is to run only ones, no repeat  
        
        see below for examples:
        batch_dir = '/nfs/xf05id1/userdata/2016_cycle1/300358_Woloschak/'
        batch_filename = 'xrf_batch_config1.txt' 
        

        waittime (float): wait time in sec. between each scans. Recommand to have few seconds for the HDF5 to finish closing.
    '''
        
    zstage_range = (-17, 40) 
    xstage_range = (0, 60) #need to check
    ystage_range = (0, 60) #need to check

    numpoints_range = (1, 1600)
    
    stepsize_range = (0.0002, 10) 
    acqtime_range = (0.1, 5) 

    numrois_range = (0, 4)    

 
    #checking the batch file directory and file name settings           
    if batch_dir is None:
        batch_dir = os.getcwd()
        print("No batch_dir was assigned, using the current directory")
    else:
        if not os.path.isdir(batch_dir):
            raise Exception("Please provide a valid batch_dir for the batch file path.")

    print("batch_dir: "+batch_dir)

    if batch_filename is None:
        raise Exception("Please provide a batch file name, e.g. batch_file = 'xrf_batch_test.txt'.")

    batchfile = batch_dir+'/'+batch_filename        
    if not os.path.isfile(batchfile):
        raise Exception("The batch_filename is not valid")
        
    print("batch_filename: "+batch_filename)
    
    
    #checking if the batch_file is correctly written:
    with open(batchfile, 'r') as batchf:
        for line in batchf:                     
            #open the log file for recodring the batch scans
            print(line)
            setpoints = line.split()
            #print(len(setpoints))
            
            if setpoints[0][0] is '#':                
                print('commented line, not for scan')
            elif len(setpoints) is not 9:
                raise Exception('The number of set points is not correct (9)')                
            else:
                print('The number of set points is correct (9)')
                                
                zposition = float(setpoints[0])
                xstart = float(setpoints[1])
                xnumstep = int(setpoints[2])
                xstepsize = float(setpoints[3]) 
                ystart = float(setpoints[4])
                ynumstep = int(setpoints[5])
                ystepsize = float(setpoints[6])
                acqtime = float(setpoints[7])
                numrois = int(setpoints[8])  
                
                xstop = xstart + xnumstep*xstepsize
                ystop = ystart + ynumstep*ystepsize 
                
                numpoints = (xnumstep+1) * (ynumstep + 1)
                
                if zposition < zstage_range[0] or zposition > zstage_range[1]:
                    raise Exception('zposition is not within range', str(zstage_range)) 
                elif xstart < xstage_range[0] or xstart > xstage_range[1]:
                    raise Exception('xstart is not within range', str(xstage_range))  
                elif xstop < xstage_range[0] or xstop > xstage_range[1]:
                    raise Exception('x finale point willnot within range', str(xstage_range))                  
                    
                elif ystart < ystage_range[0] or ystart > ystage_range[1]:
                    raise Exception('ystart is not within range', str(ystage_range)) 
                elif ystop < ystage_range[0] or ystop > ystage_range[1]:
                    raise Exception('y finale point will not be within range', str(ystage_range)) 
               
                elif numpoints < numpoints_range[0] or numpoints > numpoints_range[1]:
                    raise Exception('total scan number of point is not within range', str(numpoints_range))
                elif xstepsize < stepsize_range[0] or xstepsize > stepsize_range[1]:
                    raise Exception('xstepsize is not within range', str(stepsize_range)) 
                elif ystepsize < stepsize_range[0] or ystepsize > stepsize_range[1]:
                    raise Exception('xstepsize is not within range', str(stepsize_range))              
                elif acqtime < acqtime_range[0] or acqtime > acqtime_range[1]:
                    raise Exception('acqtime is not within range', str(acqtime_range))             
                elif numrois < numrois_range[0] or numrois > numrois_range[1]:
                    raise Exception('acqtime is not within range', str(numrois_range))                       
                else:
                    print('line is ok.')                
            
    batchf.close()   

    print('batchfile is ok, start scans.')
    
    #initializing the batch log file
    batchlogfile = batch_dir+'/logfile_'+batch_filename
    batchlogf = open(batchlogfile, 'w')
    batchlogf.close()

    for run_num in range(repeat):           
        batchlogf = open(batchlogfile, 'a')
        batchlogf.write('run number:'+str(run_num+1))
        batchlogf.close()

        with open(batchfile, 'r') as batchf:
            for line in batchf:                     
                #open the log file for recodring the batch scans
                setpoints = numpy.array(line.split())
                
                if setpoints[0][0] is '#':
                    print(line)
                    print('commented line, not for scan')
                    batchlogf = open(batchlogfile, 'a')
                    batchlogf.write(line)
                    batchlogf.close()                
                        
                #elif len(setpoints) is not 8:
                #    print(line)
                #    print('The set points of this line in the file is not correct')
                #    batchlogf = open(batch_dir+'logfile'+batch_filename, 'a')
                #    batchlogf.write(line+' '+'inccorrect set points, scan did not run')
                #    batchlogf.close()
                else:
                    
                    zposition = float(setpoints[0])
                    xstart = float(setpoints[1])
                    xnumstep = int(setpoints[2])
                    xstepsize = float(setpoints[3]) 
                    ystart = float(setpoints[4])
                    ynumstep = int(setpoints[5])
                    ystepsize = float(setpoints[6])
                    acqtime = float(setpoints[7])
                    numrois = int(setpoints[8])          
               
                    print('setting:' + 'zposition=' + str(zposition) + '\n'
                                     + 'xstart=' + str(xstart) + '\n'
                                     + 'xnumstep=' + str(xnumstep) + '\n'
                                     + 'xstepsize=' + str(xstepsize) + '\n'  
                                     + 'ystart=' + str(ystart) + '\n' 
                                     + 'ynumstep=' + str(ynumstep) + '\n'
                                     + 'ystepsize=' + str(ystepsize) + '\n'
                                     + 'acqtime=' + str(acqtime) + '\n'
                                     + 'numrois=' + str(numrois)
                         )
                    
                    hf_stage.z.move(zposition)                
                    
                    hf2dxrf(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
                        ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, 
                        acqtime=acqtime, numrois=numrois, i0map_show=False, itmap_show = False)
                        
                    batchlogf = open(batchlogfile, 'a')
                    batchlogf.write(line+' scan_id:'+ str(db[-1].start['scan_id'])+'\n')
                    batchlogf.close()
                    time.sleep(waittime)
    batchf.close()    
