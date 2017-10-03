# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:32:39 2016
set up all user specific information

@author: xf05id1
"""
import os
import scanoutput
import time

#user experiment will be put into the Start Document for every scan

#proposal_num = None 
#proposal_title = None
#PI_lastname = None
#saf_num = None

proposal_num = 302100 
proposal_title = 'Understanding the nature of interface and surface characteristics of transition metal coated and extreme environment exposed silicon carbide'
PI_lastname = 'Vasudevamurthy'
saf_num = 301657

#proposal_num = 301918 
#proposal_title = 'u-EXAFS investigation of lithium insertion kinetics in Aggregated Fe3O4 electrodes'
#PI_lastname = 'Bock'
#saf_num = 301146

#proposal_num = 301726
#proposal_title = 'Evolution of Zn tolerance in symbiotic mycorrhizal fungi'
#PI_lastname = 'Liao'
#saf_num = 301188

#proposal_num = 301905
#proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
#PI_lastname = 'Thieme'
#saf_num = 301344

#proposal_num = 301815
#proposal_title = 'Determining the Provenance of Glacial Erratics found on the North Shore of Long Island Based on XRF Geochronology of the Mineral Monazite'
#PI_lastname = 'Mozer'
#saf_num = 301267

#proposal_num = 302001
#proposal_title = 'Spatial Statistical Modeling of the Heterogeneous Reactivity of Arsenic in Soil Matrices'
#PI_lastname = 'Sharma'
#saf_num = 301274

#proposal_num = 301826
#proposal_title = 'Localization and speciation of tungsten in spinal disks'
#PI_lastname = 'Bohle'
#saf_num = 301316

#proposal_num = 301039
#proposal_title = 'Elemental and chemical evolution of FES2 additives in Li-S battery for high energy density storage'
#PI_lastname = 'Chen-Wiegart'
#saf_num = 301349


logfilename_postfix = str(saf_num)

cycle = '2017_cycle3'

RE.md['proposal']  = {  'proposal_num': str(proposal_num), 
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
    events = list(db.get_events(h, stream_name='primary'))

    for item in event0info:      
        userlogf.write('\t'+item+'='+str(events[0]['data'][item])+'\t')
    userlogf.write('\n')
    userlogf.close()
    
def metadata_record():
    RE.md['beamline_status']  = {'energy':  energy.energy.position 
                                #'slt_wb': str(slt_wb.position),
                                #'slt_ssa': str(slt_ssa.position)
                                }
                                
    RE.md['initial_sample_position'] = {'hf_stage_x': hf_stage.x.position,
                                       'hf_stage_y': hf_stage.y.position,
                                       'hf_stage_z': hf_stage.z.position}
    RE.md['wb_slits'] = {'v_gap' : slt_wb.v_gap.position,
                            'h_gap' : slt_wb.h_gap.position,
                            'v_cen' : slt_wb.v_cen.position,
                            'h_cen' : slt_wb.h_cen.position
                            }
    RE.md['hfm'] = {'y' : hfm.y.position,
                               'bend' : hfm.bend.position} 
    RE.md['ssa_slits'] = {'v_gap' : slt_ssa.v_gap.position,
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
    scannumpt = len(list(db.get_events(db[scanid])))
    
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


