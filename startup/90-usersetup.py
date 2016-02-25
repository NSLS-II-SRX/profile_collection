# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:32:39 2016

@author: xf05id1
"""
import os
import scanoutput
from databroker import DataBroker as db

#user experiment will be put into the Start Document for every scan
proposal_num = 300810
proposal_title = 'Technical Commissioning of Beamline 5-ID (SRX)'
PI_lastname = 'Chubar'
saf_num = 300265

proposal_num = 300802
proposal_title = 'Elemental Segregation and Speciation in the Lead Carboxylate (Soap) Deterioration of Oil Paintings'
PI_lastname = 'Centeno'
saf_num = 300312


cycle = '2016_cycle1'

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


userlogfile = userdatadir+'logfile.txt'

import os.path
if not os.path.exists(userlogfile):
    userlogf = open(userlogfile, 'w')
    userlogf.close()
    
def logscan(scantype):
    h=db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']
    #scantype = '2dxrf'

    userlogf = open(userlogfile, 'a')
    userlogf.write(str(scan_id)+'\t'+uid+'\t'+scantype+'\n')
    userlogf.close()