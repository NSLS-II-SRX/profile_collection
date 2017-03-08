# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:32:39 2016
set up all user specific information

@author: xf05id1
"""
import os
import scanoutput
from databroker import DataBroker as db, get_events
import time

#user experiment will be put into the Start Document for every scan

proposal_num = None 
proposal_title = None
PI_lastname = None
saf_num = None

#proposal_num = 300766
#proposal_title = 'Trace Elements in Fluorite as a Window into Ore Forming Fluids and Igneous Petrogenesis'
#PI_lastname = 'Acerbo'
#saf_num = 300846

#proposal_num = 301566
#proposal_title = 'Study oxidation of nuclear cladding advanced steel alloys'
#PI_lastname = 'Elbakhshwan'
#saf_num = 300887

#proposal_num = 300269
#proposal_title = 'Laser Fabrication of single-crystal architecture in glass'
#PI_lastname = 'Jain'
#saf_num = 300828

#proposal_num = 300928
#proposal_title = 'Elemental Association and Chemical Speciation of Transition Metals in Submicrometer Atmospheric Particles'
#PI_lastname = 'Moffet'
#saf_num = 300947

#proposal_num = 301398
#proposal_title = 'Stoichiometry of detector-grate CdZnTe crystals and its influence on detectors performance'
#PI_lastname = 'Hossain'
#saf_num = 300953

#proposal_num = 301373
#proposal_title = 'Fe redox transformations during serpentinization: Implications for life and planetary-scale oxidation states'
#PI_lastname = 'Tutolo'
#saf_num = 300970

proposal_num = 301387 
proposal_title = 'Elemental concentration and size apportionment of combustion particles from wood-fired appliances'
PI_lastname = 'Gray-Georges'
saf_num = 300994

logfilename_postfix = str(saf_num)

cycle = '2017_cycle1'

gs.RE.md['proposal']  = {  'proposal_num': str(proposal_num), 
                         'proposal_title': str(proposal_title),
                            'PI_lastname': str(PI_lastname),
                                'saf_num': str(saf_num),
                                  'cycle': str(cycle)                            
                        }
                        

userdatadir = '/nfs/xf05id1/userdata/'+str(cycle)+'/'+str(saf_num)+'_'+str(PI_lastname)+'/'
scanoutput._DEFAULT_FILEDIR = userdatadir

try:    
    os.makedirs(userdatadir, exist_ok=True)
except Exception as e:
    print(e)
    print('cannot create directory:' + userdatadir)
    sys.exit()


userlogfile = userdatadir+'logfile'+logfilename_postfix+'.txt'

import os.path
if not os.path.exists(userlogfile):
    userlogf = open(userlogfile, 'w')
    userlogf.close()
    
def logscan(scantype):
    h=db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    userlogf = open(userlogfile, 'a')
    userlogf.write(str(scan_id)+'\t'+uid+'\t'+scantype+'\n')
    userlogf.close()
    
def logscan_event0info(scantype, event0info = []):
    h=db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    userlogf = open(userlogfile, 'a')
    userlogf.write(str(scan_id)+'\t'+uid+'\t'+scantype)
    events = list(get_events(h, stream_name='primary'))

    for item in event0info:      
        userlogf.write('\t'+item+'='+str(events[0]['data'][item])+'\t')
    userlogf.write('\n')
    userlogf.close()
    
def metadata_record():
    gs.RE.md['beamline_status']  = {'energy':  energy.energy.position 
                                #'slt_wb': str(slt_wb.position),
                                #'slt_ssa': str(slt_ssa.position)
                                }
                                
    gs.RE.md['initial_sample_position'] = {'hf_stage_x': hf_stage.x.position,
                                       'hf_stage_y': hf_stage.y.position,
                                       'hf_stage_z': hf_stage.z.position}
    gs.RE.md['wb_slits'] = {'v_gap' : slt_wb.v_gap.position,
                            'h_gap' : slt_wb.h_gap.position,
                            'v_cen' : slt_wb.v_cen.position,
                            'h_cen' : slt_wb.h_cen.position
                            }
    gs.RE.md['hfm'] = {'y' : hfm.y.position,
                               'bend' : hfm.bend.position} 
    gs.RE.md['ssa_slits'] = {'v_gap' : slt_ssa.v_gap.position,
                            'h_gap' : slt_ssa.h_gap.position,
                            'v_cen' : slt_ssa.v_cen.position,
                            'h_cen' : slt_ssa.h_cen.position                                      
                             }                                      
                                       
def logscan_detailed(scantype):
    h=db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    userlogf = open(userlogfile, 'a')
    userlogf.write(str(scan_id)+'\t'+uid+'\t'+scantype+'\t'+str(h['start']['scan_params'])+'\n')
    userlogf.close()

def scantime(scanid, printresults=True):
    '''
    input: scanid
    return: start and stop time stamps as strings 
    '''
    start_str = 'scan start: '+time.ctime(db[scanid].start['time'])
    stop_str  = 'scan stop : '+time.ctime(db[scanid].stop['time'])
    totaltime = db[scanid].stop['time'] - db[scanid].start['time']
    scannumpt = len(list(get_events(db[scanid])))
    
    if printresults is True:
        print(start_str)
        print(stop_str)
        print('total time:', totaltime, 's')
        print('number of points:', scannumpt)
        print('scan time per point:', totaltime/scannumpt, 's')
    return db[scanid].start['time'], db[scanid].stop['time'], start_str, stop_str

def timestamp_batchoutput(filename = 'timestamplog.text', initial_scanid = None, final_scanid = None):
    f = open(filename,'w')
    for scanid in range(initial_scanid, final_scanid+1):
        f.write(str(scanid)+'\n')
        try: 
            start_t, stop_t = scantime(scanid)
            f.write(start_t)
            f.write('\n')
            f.write(stop_t)
            f.write('\n')
        except:
            f.write('scan did no finish correctly.\n')
    f.close()

def scantime_batchoutput(filename = 'scantimelog.txt', scanlist = []):

    f = open(filename, 'w')
    f.write('scanid\tstartime(s)\tstoptime(s)\tstartime(date-time)\tstoptime(date-time)\n')
    for i in scanlist:
        starttime_s, endtime_s, starttime, endtime = scantime(i, printresults=False)
        f.write(str(i)+'\t'+str(starttime_s)+'\t'+str(endtime_s)+'\t'+starttime[12::]+'\t'+endtime[12::]+'\n')
    f.close()

