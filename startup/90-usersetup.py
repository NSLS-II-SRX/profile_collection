# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:32:39 2016
set up all user specific information

@author: xf05id1
"""
import os
import scanoutput
from databroker import DataBroker as db
import time

#user experiment will be put into the Start Document for every scan

#proposal_num = 300810
#proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
#PI_lastname = 'Thieme'
#saf_num = 300265

#proposal_num = 300802
#proposal_title = 'Elemental Segregation and Speciation in the Lead Carboxylate (Soap) Deterioration of Oil Paintings'
#PI_lastname = 'Centeno'
#saf_num = 300312
##avoid hitting undulator minimum gap during XANES scan
#energy.harmonic.put(7)

#proposal_num = 300546
#proposal_title = 'Simultaneous nanoscale X-ray ptychography and fluorescence measurements of heavy metal uptake in developing C. elegans embryos'
#PI_lastname = 'Jones'
#saf_num = 300329

#proposal_num = 300640
#proposal_title = 'Benchmarking and optimization of microscopic and spectroscopic performances of SRX beamline'
#PI_lastname = 'Tchoubar'
#saf_num = 300388
#logfilename_postfix = '20160406'

#proposal_num = 300605
#proposal_title = 'In-situ Spatial Resolution of the Electroactive Interface of iron bsed composite electrodes'
#PI_lastname = 'Takeuchi'
#saf_num = 300353
#logfilename_postfix = str(saf_num)

#proposal_num = 300178
#proposal_title = 'Nanoparticle Distribution in Biological Samples'
#PI_lastname = 'Woloschak'
#saf_num = 300358

#proposal_num = 300261
#proposal_title = 'Correlation between Processing Conditions, Chemical Heterogeneity, and Morphology in Nanofoams for Energy Applications'
#PI_lastname = 'Chen-Wiegart-ssfoam'
#saf_num = 300360

#proposal_num = 300126
#proposal_title = 'Process-Structure-Properties Correlation in Anti-corrosion Nano-coatings Fabricated via Environmentally Friendly Processes'
#PI_lastname = 'Chen-Wiegart-Henkel'
#saf_num = 300362

#proposal_num = 300381
#proposal_title = 'Spatial and temporal nano-mapping of comlexes in low-cost aqueousbattery materials for large-scale, high energy density storage'
#PI_lastname = 'Gallaway'
#saf_num = 300372

#proposal_num = 300658
#proposal_title = 'Transition Metal Oxyanions as specific inhibitors of sulfidogenesis'
#PI_lastname = 'Thieme'
#saf_num = 300375

#proposal_num = 300810
#proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
#PI_lastname = 'Chen-Wiegart-MIT'
#saf_num = 300337

#proposal_num = 300579
#proposal_title = 'Metal Uptake, Translocation and Accumulation in Plants Across Size'
#PI_lastname = 'Blaby'
#saf_num = 300397

#proposal_num = 300537
#proposal_title = 'Elemental and chemical evolution of Cu-S additives in Li-S battery for high energy density storage'
#PI_lastname = 'Chen-Wiegart-LiSbattery'
#saf_num = 300398

#proposal_num = 300810
#proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
#PI_lastname = 'Thieme-bandediron'
#saf_num = 300265

#proposal_num = 300626
#proposal_title = 'Micro and Nanoscale Reactivity Matrices'
#PI_lastname = 'Hesterberg'
#saf_num = 300393

#proposal_num = 300640
#proposal_title = 'Benchmarking and optimization of microscopic and spectroscopic performances of SRX beamline'
#PI_lastname = 'Tchoubar-20160417'
#saf_num = 300388
#logfilename_postfix = '20160406'

#proposal_num = 300810
#proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
#PI_lastname = 'Chen-Wiegart-3Dprint'
#saf_num = 300265

#proposal_num = 301229
#proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
#PI_lastname = 'Thieme-undulatorAlignment'
#saf_num = 300441

proposal_num = 301229
proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
PI_lastname = 'Thieme-beamlinecomissioning'
saf_num = 300441

proposal_num = 301229
proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
PI_lastname = 'Chubar-MachineCollectiveEffects'
saf_num = 300441

logfilename_postfix = str(saf_num)

cycle = '2016_cycle2'

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
    
    if printresults is True:
        print(start_str)
        print(stop_str)
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

