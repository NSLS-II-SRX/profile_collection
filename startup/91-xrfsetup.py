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

from bluesky.plans import outer_product_scan, scan
import bluesky.plans as bp
from bluesky.callbacks import LiveGrid
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import subs_wrapper
from bluesky.plan_stubs import mv
import matplotlib
import time
import epics
import os
import numpy
import collections

#matplotlib.pyplot.ticklabel_format(style='plain')
def get_stock_md():
    md = {}
    md['beamline_status']  = {'energy':  energy.energy.position 
                                #'slt_wb': str(slt_wb.position),
                                #'slt_ssa': str(slt_ssa.position)
                                }
                                
    md['initial_sample_position'] = {'hf_stage_x': hf_stage.x.position,
                                     'hf_stage_y': hf_stage.y.position,
                                     'hf_stage_z': hf_stage.z.position}
    md['wb_slits'] = {'v_gap' : slt_wb.v_gap.position,
                            'h_gap' : slt_wb.h_gap.position,
                            'v_cen' : slt_wb.v_cen.position,
                            'h_cen' : slt_wb.h_cen.position
                            }
    md['hfm'] = {'y' : hfm.y.position,
                               'bend' : hfm.bend.position} 
    md['ssa_slits'] = {'v_gap' : slt_ssa.v_gap.position,
                            'h_gap' : slt_ssa.h_gap.position,
                            'v_cen' : slt_ssa.v_cen.position,
                            'h_cen' : slt_ssa.h_cen.position                                      
                             }                                      
    return md

def get_stock_md_xfm():
    md = {}
    md['beamline_status']  = {'energy':  energy.energy.position 
                                #'slt_wb': str(slt_wb.position),
                                #'slt_ssa': str(slt_ssa.position)
                                }
                                
    md['initial_sample_position'] = {'stage27a_x': stage.x.position,
                                       'stage27a_y': stage.y.position,
                                       'stage27a_z': stage.z.position}
    md['wb_slits'] = {'v_gap' : slt_wb.v_gap.position,
                            'h_gap' : slt_wb.h_gap.position,
                            'v_cen' : slt_wb.v_cen.position,
                            'h_cen' : slt_wb.h_cen.position
                            }
    md['hfm'] = {'y' : hfm.y.position,
                               'bend' : hfm.bend.position} 
    md['ssa_slits'] = {'v_gap' : slt_ssa.v_gap.position,
                            'h_gap' : slt_ssa.h_gap.position,
                            'v_cen' : slt_ssa.v_cen.position,
                            'h_cen' : slt_ssa.h_cen.position                                      
                             }                                      
    return md                                       

def hf2dxrf(*, xstart, xnumstep, xstepsize, 
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
    c2pitch_kill=EpicsSignal("XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Cmd:Kill-Cmd")

    #record relevant meta data in the Start document, defined in 90-usersetup.py
    md = get_stock_md()
    md['sample']  = {'name': samplename}
    md['scaninfo']  = {'type': 'XRF', 'raster' : True}
    if e_tomo is not None:
        md['scaninfo']  = {'type': 'E_Tomo', 'raster' : True}
    h=db[-1]
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
    if record_cryo is True:
        det = [current_preamp, xs, cryo_v19, cryo_lt19, cryo_pt1, 
           hdcm_Si111_1stXtalrtd, hdcm_Si111_2ndXtal_rtd,  hdcm_1stXtal_ThermStab_rtd, hdcm_ln2out_rtd, hdcm_water_rtd,
           dBPM_h, dBPM_v, dBPM_t, dBPM_i, dBPM_o, dBPM_b]
    else: 
        if struck == False:
            det = [current_preamp, xs]
        else:
            det = [sclr1, xs]
        
    #gjw
    #det = [xs, hfvlmAD]        
    #gjw

    if dpc is not None:
        det.append(dpc)
        dpc.cam.acquire.put(0)
        dpc.cam.image_mode.put(0)
        #dpc.cam.acquire_time.put(acqtime)
        dpc.cam.acquire_time.put(acqtime*0.2)
    if e_tomo is not None:
        md = ChainMap( md, {'hf_stage_th': hf_stage.th.position})
        det.append(e_tomo)
        e_tomo.external_trig.put(False)
        e_tomo.settings.acquire_time.put(acqtime)
        e_tomo.total_points.put((xnumstep+1)*(ynumstep+1))



    #setup the live callbacks
    livecallbacks = []
    
    def time_per_point(name,doc, st=time.time()):
        if 'seq_num' in doc.keys():
            #print((doc['time'] - st) / doc['seq_num'])
            if srecord is None:
                scanrecord.scan0.tpp.put((doc['time'] - st) / doc['seq_num'])
                scanrecord.scan0.curpt.put(int(doc['seq_num']))
            else:
                srecord.tpp.put((doc['time'] - st) / doc['seq_num'])
                srecord.curpt.put(int(doc['seq_num']))
            scanrecord.time_remaining.put( (doc['time'] - st) / doc['seq_num'] * 
                                           ( (xnumstep + 1) * (ynumstep + 1) - doc['seq_num']) / 3600)
                

        #write current point and the time per point
    livecallbacks.append(time_per_point)
                                
    if struck == False:
        livetableitem = [xmotor.name, ymotor.name, 'current_preamp_ch0', 'current_preamp_ch2']
    else:
        livetableitem = [xmotor.name, ymotor.name, i0.name]

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
#    print('xstop = '+str(xstop))  
#    print('ystop = '+str(ystop)) 
    
    
    for roi_idx in range(numrois):
        roi_name = 'roi{:02}'.format(roi_idx+1)
        
        roi_key = getattr(xs.channel1.rois, roi_name).value.name
        livetableitem.append(roi_key)
        

        if e_tomo is None:
            roimap = LiveGrid((ynumstep+1, xnumstep+1), roi_key, clim=None, cmap='inferno', 
                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
            livecallbacks.append(roimap)

    if e_tomo is not None:
        #livecallbacks.remove(roimap)
        for roi_idx in range(numrois):
            roi_name = 'roi{:02}'.format(roi_idx+1)
        
            roi_key = getattr(xs2.channel1.rois, roi_name).value.name
            livetableitem.append(roi_key)
        
            roimap2 = LiveGrid((ynumstep+1, xnumstep+1), roi_key, clim=None, cmap='inferno', 
                                xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
            livecallbacks.append(roimap2)

    if dpc is not None:
        dpc_tmap = LiveGrid((ynumstep+1, xnumstep+1), dpc.stats1.total.name, clim=None, cmap='magma',
                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(dpc_tmap)
#        dpc_hmap = LiveGrid((ynumstep+1, xnumstep+1), dpc.stats1.centroid.x.name, clim=None, cmap='magma',
#                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
#        livecallbacks.append(dpc_hmap)
#        dpc_vmap = LiveGrid((ynumstep+1, xnumstep+1), dpc.stats1.centroid.y.name, clim=None, cmap='magma',
#                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
#        livecallbacks.append(dpc_vmap)


    if i0map_show is True:
        if struck == False:
            i0map = LiveGrid((ynumstep+1, xnumstep+1), 'current_preamp_ch2', clim=None, cmap='viridis', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        else:
            i0map = LiveGrid((ynumstep+1, xnumstep+1), i0.name, clim=None, cmap='viridis', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(i0map)

    if itmap_show is True:
        itmap = LiveGrid((ynumstep+1, xnumstep+1), 'current_preamp_ch0', clim=None, cmap='magma', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(itmap)
    
    #this does not seem to work
    if record_cryo is True:
        cryo_v19map = LiveGrid((ynumstep+1, xnumstep+1), 'cryo_v19', clim=None, cmap='jet', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(cryo_v19map)

        cryo_lt19map = LiveGrid((ynumstep+1, xnumstep+1), 'cryo_lt19', clim=None, cmap='jet', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(cryo_lt19map)

        dBPM_hmap = LiveGrid((ynumstep+1, xnumstep+1), 'dBPM_h', clim=None, cmap='jet', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(dBPM_hmap)

        dBPM_vmap = LiveGrid((ynumstep+1, xnumstep+1), 'dBPM_v', clim=None, cmap='jet', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(dBPM_vmap)


    #gjw
    #vlmmap=LiveGrid((ynumstep+1, xnumstep+1), 'hfvlm_stats3_total', clim=None, cmap='inferno',\
    #    xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
    #livecallbacks.append(vlmmap)
    #gjw

#    commented out liveTable in 2D scan for now until the prolonged time issue is resolved
    livecallbacks.append(LiveTable(livetableitem)) 

    
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
    if shutter is True: 
        yield from abs_set(xmotor,xstart, wait = True)
        yield from abs_set(ymotor,ystart, wait = True)
        yield from mv(shut_b,'Open')

    #peak up monochromator at this energy
    if align == True:
        ps = PeakStats(dcm.c2_pitch.name,i0.name)
        e_value = energy.energy.get()[1]
        if e_value < 14. and struck==True:
            sclr1.preset_time.put(0.1, wait = True)
        elif (struck == True):
            sclr1.preset_time.put(1., wait = True)
        peakup = scan([sclr1], dcm.c2_pitch, -19.320, -19.360, 41)
        peakup = subs_wrapper(peakup,ps)
        yield from peakup
        yield from abs_set(dcm.c2_pitch, ps.cen, wait = True)
        #yield from abs_set(c2pitch_kill,1)

    def at_scan(name, doc):
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        scanrecord.current_type.put(md['scaninfo']['type'])
        scanrecord.scanning.put(True)

    def finalize_scan(name, doc):
        scanrecord.scanning.put(False)


    hf2dxrf_scanplan = outer_product_scan(det, ymotor, ystart, ystop, ynumstep+1, xmotor, xstart, xstop, xnumstep+1, True, md=md)
#    hf2dxrf_scanplan = bp.subs_wrapper( hf2dxrf_scanplan, livecallbacks)
    hf2dxrf_scanplan = subs_wrapper( hf2dxrf_scanplan, {'all':livecallbacks,'start':at_scan,'stop':finalize_scan})
    scaninfo = yield from hf2dxrf_scanplan
    #TO-DO: implement fast shutter control (close)    
    if shutter is True:
        yield from mv(shut_b,'Close')

    #write to scan log    

    if dpc is not None:    
        logscan_event0info('2dxrf_withdpc', event0info = [dpc.tiff.file_name.name])
    else:
        logscan('2dxrf')    
    
    return scaninfo
#    return yield from bp.subs_wrapper(hf2dxrf_scanplan,{'start':at_scan})
    
    
def multi_region_h(regions, energy_list=None, **kwargs):
    ret = []
    
    for r in regions:
        inp = {}
        inp.update(kwargs)
        inp.update(r)
        rs_uid = yield from hf2dxrf(**inp)
        ret.extend(rs_uid)
    return ret


def hf2dxrf_estack(batch_dir = None, batch_filename = None,
            erange = [], estep = [],
            energy_pt = None,  echange_waittime = 5, energy_waittime = 5,
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
    
    if energy_pt is None:
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

    if energy_pt is not None:
        ept = energy_pt
    else:
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
        #energy.move(energy_setpt)  
        #time.sleep(energy_waittime)
        
        batchlogf = open(batchlogfile, 'a')
        batchlogf.write('energylist.append('+str(energy_setpt)+')\n')
        batchlogf.close()
        #run hf2dxrf scans
        yield from hf2dxrf(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
            ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize,  i0map_show=i0map_show, itmap_show = itmap_show,
            setenergy=energy_setpt, echange_waittime=echange_waittime,
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime=acqtime, numrois=numrois)
        #time.sleep(waittime)
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
        
def hf2dxrf_xybatch(batch_dir = None, batch_filename = None, waittime = 5, repeat = 1, batch_filelog_ext = '', shutter=True, dpc = None,i0map_show=False,itmap_show=False, struck = True, align = False):
    '''
    This function will load from the batch text input file and run the 2D XRF according to the set points in the text file.
    input:
        batch_dir (string): directory for the input batch file
        batch_filename (string): text file name that defines the set points for batch scans
        repeat (integer): number to repeat the scans in the batch; repeat = 1 is to run only ones, no repeat  
        dpc: pass dpc keyword to hf2dxrf
        
        see below for examples:
        batch_dir = '/nfs/xf05id1/userdata/2016_cycle1/300358_Woloschak/'
        batch_filename = 'xrf_batch_config1.txt'         

        waittime (float): wait time in sec. between each scans. Recommand to have few seconds for the HDF5 to finish closing.
        batch_filelog_ext (string): default is empty; any string can be assigned and will be inserted as part of the file name of the log file
    '''
        
    zstage_range = (-28, 80)
    xstage_range = (0, 70) #need to check
    ystage_range = (-5, 60) #need to check

    numpoints_range = (1, 160000)
    
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
    batchlogfile = batch_dir+'/logfile_' + batch_filelog_ext + batch_filename
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
                    
                    hf2dxrf_gen = yield from hf2dxrf(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
                        ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, shutter=shutter, struck=struck,
                        acqtime=acqtime, numrois=numrois, i0map_show=i0map_show, itmap_show = itmap_show, 
                        dpc = dpc, align = align)
                        
                    batchlogf = open(batchlogfile, 'a')
                    batchlogf.write(line+' scan_id:'+ str(db[-1].start['scan_id'])+'\n')
                    batchlogf.close()
                    time.sleep(waittime)
    batchf.close()    


def hr2dxrf_top(*, xstart, xnumstep, xstepsize, 
            ystart, ynumstep, ystepsize, 
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime, numrois=1, i0map_show=True, itmap_show=True,
            energy=None, u_detune=None):

    '''
    input:
        xstart, xnumstep, xstepsize (float) unit: micron
        ystart, ynumstep, ystepsize (float) unit: micron
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
    md = get_stock_md()

    #setup the detector
    # TODO do this with configure
    current_preamp.exp_time.put(acqtime)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put((xnumstep+1)*(ynumstep+1))

         
    det = [current_preamp, xs]        

    #setup the live callbacks
    livecallbacks = []
    
    livetableitem = [tomo_stage.finex_top, tomo_stage.finey_top, 'current_preamp_ch0', 'current_preamp_ch2']

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
    print('xstop = '+str(xstop))  
    print('ystop = '+str(ystop)) 
    
    
    for roi_idx in range(numrois):
        roi_name = 'roi{:02}'.format(roi_idx+1)
        
        roi_key = getattr(xs.channel1.rois, roi_name).value.name
        livetableitem.append(roi_key)

        colormap = 'jet' #previous set = 'viridis'

        roimap = LiveGrid((ynumstep+1, xnumstep+1), roi_key, clim=None, cmap='jet', 
                            xlabel='x (um)', ylabel='y (um)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(roimap)


    if i0map_show is True:
        i0map = LiveGrid((ynumstep+1, xnumstep+1), 'current_preamp_ch2', clim=None, cmap='jet', 
                        xlabel='x (um)', ylabel='y (um)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(i0map)

    if itmap_show is True:
        itmap = LiveGrid((ynumstep+1, xnumstep+1), 'current_preamp_ch0', clim=None, cmap='jet', 
                        xlabel='x (um)', ylabel='y (um)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(itmap)

    livecallbacks.append(LiveTable(livetableitem)) 

    
    #setup the plan  
    #outer_product_scan(detectors, *args, pre_run=None, post_run=None)
    #outer_product_scan(detectors, motor1, start1, stop1, num1, motor2, start2, stop2, num2, snake2, pre_run=None, post_run=None)

    if energy is not None:
        if u_detune is not None:
            # TODO maybe do this with set
            energy.detune.put(u_detune)
        # TODO fix name shadowing
        yield from bp.abs_set(energy, energy, wait=True)
    

    #TO-DO: implement fast shutter control (open)
    #TO-DO: implement suspender for all shutters in genral start up script
    
    
    hr2dxrf_scanplan = outer_product_scan(det, tomo_stage.finey_top, ystart, ystop, ynumstep+1, tomo_stage.finex_top, xstart, xstop, xnumstep+1, True, md=md)
    hr2dxrf_scanplan = subs_wrapper(hr2dxrf_scanplan, livecallbacks)
    scaninfo = yield from hr2dxrf_scanplan

    #TO-DO: implement fast shutter control (close)    

    #write to scan log    
    logscan('2dxrf_hr_top')    
    
    return scaninfo
    
    
def hf2dxrf_xfm(*, xstart, xnumstep, xstepsize, 
            ystart, ynumstep, ystepsize, 
            shutter = True, struck = True, align = False,
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime, numrois=1, i0map_show=True, itmap_show=False,
            energy=None, u_detune=None,samplename=None):

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
    md = get_stock_md_xfm()
    md['sample']  = {'name': samplename}
    md['scaninfo']  = {'type': 'XRF', 'raster' : True}

    #setup the detector
    # TODO do this with configure
    # current_preamp.exp_time.put(acqtime)
    sclr1.preset_time.put(acqtime)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put((xnumstep+1)*(ynumstep+1))

    # det = [current_preamp, xs]
    det = [sclr1, xs]

    #setup the live callbacks
    livecallbacks = []
    
    # livetableitem = [stage.x, stage.y, 'current_preamp_ch0', 'current_preamp_ch2']
    livetableitem = [stage.x, stage.y, i0.name]

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
    print('xstop = '+str(xstop))  
    print('ystop = '+str(ystop)) 
    
    
    for roi_idx in range(numrois):
        roi_name = 'roi{:02}'.format(roi_idx+1)
        
        roi_key = getattr(xs.channel1.rois, roi_name).value.name
        livetableitem.append(roi_key)
        cscheme = 'inferno'
        
        roimap = LiveGrid((ynumstep+1, xnumstep+1), roi_key, clim=None, cmap=cscheme, 
                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(roimap)


    if i0map_show is True:
        # i0map = LiveGrid((ynumstep+1, xnumstep+1), 'current_preamp_ch2', clim=None, cmap='viridis', 
        #                 xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        i0map = LiveGrid((ynumstep+1, xnumstep+1), i0.name, clim=None, cmap='viridis', 
                                xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(i0map)

    if itmap_show is True:
        itmap = LiveGrid((ynumstep+1, xnumstep+1), 'current_preamp_ch0', clim=None, cmap='magma', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(itmap)

    livecallbacks.append(LiveTable(livetableitem)) 

    
    #setup the plan  

    if energy is not None:
        if u_detune is not None:
            # TODO maybe do this with set
            energy.detune.put(u_detune)
        # TODO fix name shadowing
        yield from bp.abs_set(energy, energy, wait=True)
    

    #TO-DO: implement fast shutter control (open)
    #TO-DO: implement suspender for all shutters in genral start up script
    # if shutter is True: 
    #     shut_b.put(1)
    if shutter is True:
       yield from abs_set(stage.x,xstart, wait = True)
       yield from abs_set(stage.y,ystart, wait = True)
       yield from mv(shut_b,'Open')

    # Snake option is set to False
    hf2dxrf_scanplan = outer_product_scan(det, stage.y, ystart, ystop, ynumstep+1, stage.x, xstart, xstop, xnumstep+1, False, md=md)
    hf2dxrf_scanplan = subs_wrapper( hf2dxrf_scanplan, livecallbacks)
    scaninfo = yield from hf2dxrf_scanplan

    #TO-DO: implement fast shutter control (close)    
    if shutter is True:
        yield from abs_set(stage.x, xstart, wait=True)
        yield from abs_set(stage.y, ystart, wait=True)
        yield from mv(shut_b,'Close') 

    #write to scan log    
    logscan('2dxrf_xfm')    
    
    return scaninfo
    
def hf2dxrf_xybatch_xfm(batch_dir = None, batch_filename = None, waittime = 5, repeat = 1):
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
        
    zstage_range = (-82.1, -70) #need to check
    xstage_range = (41, 61) #need to check
    ystage_range = (-140, 10) #need to check

    numpoints_range = (1, 1600000)
    
    stepsize_range = (0.0002, 10) 
    acqtime_range = (0.2, 5) 

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
        batchlogf.write('run number:'+str(run_num+1)+'\n')
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
                    hf_stage_z_gen = yield from list_scan([stage], stage.z, [zposition])
                    #hf_stage.z.move(zposition)                
                    
                    hf2dxrf_xfm_gen = yield from hf2dxrf_xfm(xstart=xstart, xnumstep=xnumstep, xstepsize=xstepsize, 
                        ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, 
                        acqtime=acqtime, numrois=numrois, i0map_show=False, itmap_show = False)
                        
                    batchlogf = open(batchlogfile, 'a')
                    batchlogf.write(line+' scan_id:'+ str(db[-1].start['scan_id'])+'\n')
                    batchlogf.close()
                    time.sleep(waittime)
    batchf.close()  

def hf2dxrf_ioc(waittime = 5, shutter=True, dpc = None, i0map_show=False,itmap_show=False, 
                     struck = True, align = False, numrois = 1,samplename=None):
    '''
    invokes hf2dxrf repeatedly with parameters provided separately.
        waittime                [sec]       time to wait between scans
        shutter                 [bool]      scan controls shutter
        dpc                     [bool]      use transmission area detector
        i0map_show              [bool]      show I_0 map
        itmap_show              [bool]      show I_t map
        struck                  [bool]      use scaler for I_0
        align                   [bool]      optimize beam location on each scan
        numrois                 [1,2,3]     number of rois to display on each scan
        
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
                    ystart=ystart, ynumstep=ynumstep, ystepsize=ystepsize, shutter=shutter, struck=struck,
                    acqtime=acqtime, numrois=numrois, i0map_show=i0map_show, itmap_show = itmap_show, 
                    dpc = dpc, align = align, samplename=samplename)
            if len(scanlist) is not 0:
                time.sleep(waittime)
    scanrecord.current_scan.put('')
