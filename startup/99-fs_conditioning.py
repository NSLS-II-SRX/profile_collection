# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:17:33 2016

@author: xf05id1
"""

from epics import PV
import time

ttl1 = PV('XF:05IDD-ES:1{EVR:1-Out:FP1}Src:Scale-SP')
BT1p = PV('XF:05IDD-VA:1{BT:1-CCG:1}P-I')

def movefs():
#    ttl1.put(4)
#    time.sleep(2)
#    ttl1.put(3)
     shut_fast.high_cmd()
     time.sleep(5)
     shut_fast.low_cmd()

def fs_cycle(setp = 2E-6, open_wait = 1, close_wait = 5, duration = 24):
    '''
     open_wait, close_wait: wait time in min
     duration: in hours, how long to run the program as minimum time, can always Ctrl+C out
    '''
    
    num_cycle = int(duration*60/open_wait) #assuming no closing, minimum time to run    
    
    for i in range(num_cycle+1):
        print('open fast shutter')
        movefs()
        while(BT1p.get() < setp):
            print('pressure: '+str(BT1p.get()))
            print('wait '+str(open_wait)+' min')
            time.sleep(open_wait*60)
        print('pressure: '+str(BT1p.get()))        
        print('close fast shutter')
        movefs()
        print('wait '+str(close_wait)+' min')
        time.sleep(close_wait*60)
