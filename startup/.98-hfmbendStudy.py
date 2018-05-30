# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:59:04 2016

@author: xf05id1
"""
from bluesky.plans import Count
from epics import PV

hfvlmAD.tiff.read_attrs.append('file_name')
hfvlmAD.read_attrs.append('cam')

studylogfilename = '/nfs/xf05id1/userdata/2016_cycle1/300388_Tchoubar-20160417/studylogfile.txt'
vlmcam_exptime = PV('XF:05IDD-BI:1{Mscp:1-Cam:1}AcquireTime')


def vlmimgacq():
    metadata_record()
    imgplan = Count([hfvlmAD], num=10)
    gs.RE(imgplan)
    logscan('hfvlmimg')
    
    studylogf = open(studylogfilename, 'a')
    studylogf.write('VLM images\n')
    studylogf.write('scan_id: '+str(db[-1].start['scan_id'])+'\n')
    studylogf.write('scan_uid: '+str(db[-1].start['uid'])+'\n')
    event = list(get_events(db[-1], stream_name='primary'))
    studylogf.write('hfvlm_cam_acquire_time: '+str(event[0]['data']['hfvlm_cam_acquire_time'])+'\n')
    studylogf.write('energy: '+str(db[-1].start['beamline_status']['energy'])+'\n') 
    studylogf.write('tiff_filename: '+event[0]['data']['hfvlm_tiff_file_name']+'\n')
    studylogf.write('ssa_v_gap: '+str(db[-1].start['ssa_slits']['v_gap'])+'\n\n')
        
    studylogf.close()
    
def vlmimg_xanes(hfmbend = None, waittime = 1):
    
    if hfmbend is not None:
        hfm.bend.move(hfmbend)
    else:
        hfmbend = hfm.bend.position
    time.sleep(waittime)
    
    studylogf = open(studylogfilename, 'a')
    studylogf.write('\n')
    studylogf.write(str(hfmbend)+'\n')
    studylogf.close()
    
    #move to scintillator
    hf_stage.x.move(30)
    hf_stage.y.move(20.06)
    time.sleep(waittime)
    
    #open b shutter
    shut_b.open_cmd.put(1)
    while (shut_b.close_status.get() == 1):
        epics.poll(.5)
        shut_b.open_cmd.put(1)    
    
    #acquire vlm iamge
    vlmimgacq()
    
    #move to Se foil
    hf_stage.x.move(8.4)
    hf_stage.y.move(21.995)
    
    #acquire xanes
    xanes(erange = [12630, 12640, 12680, 12700], 
            estep = [2, 0.25, 5],  
            harmonic = None,            
            acqtime=0.2, roinum=1, i0scale = 1e8, itscale = 1e8,samplename='Se foil',filename='Ptstripe'+str(hfmbend))
    studylogf = open(studylogfilename, 'a')
    studylogf.write('XANES scan\n')
    studylogf.write('scan_id: '+str(db[-1].start['scan_id'])+'\n')
    studylogf.write('scan_uid: '+str(db[-1].start['uid'])+'\n')   
    studylogf.close()
                        
def vlmimg_xanes_loop(hfmbend_range = [300000, 0], hfmbend_step = -30000):

    for hfmbend_target in range(hfmbend_range[0], hfmbend_range[1]+hfmbend_step, hfmbend_step):
        if hfmbend_target <=300000 and hfmbend_target >=266000:
            vlmcam_exptime.put(0.4)
        elif hfmbend_target <266000 and hfmbend_target >=212000:
            vlmcam_exptime.put(0.2)
        elif hfmbend_target <212000 and hfmbend_target >=180000:
            vlmcam_exptime.put(0.1)
        elif hfmbend_target <180000 and hfmbend_target >=136000:
            vlmcam_exptime.put(0.05)
        elif hfmbend_target <136000 and hfmbend_target >=104000:
            vlmcam_exptime.put(0.1)
        elif hfmbend_target <104000 and hfmbend_target >=32000:
            vlmcam_exptime.put(0.2)
        elif hfmbend_target <32000 and hfmbend_target >= 0:
            vlmcam_exptime.put(0.4)   
            
        vlmimg_xanes(hfmbend = hfmbend_target)
