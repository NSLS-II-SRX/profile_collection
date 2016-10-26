# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 19:08:01 2016

@author: xf05id1
#"""

from bluesky.plans import count, subs_wrapper
import copy

#

cryo_v19 = EpicsSignalRO('XF:05IDA-UT{Cryo:1-IV:19}Sts-Sts', name='cryo_v19')
cryo_lt19 = EpicsSignalRO('XF:05IDA-UT{Cryo:1}L:19-I', name='cryo_lt19')
cryo_pt1 = EpicsSignalRO('XF:05IDA-UT{Cryo:1}P:01-I', name='cryo_pt1')

hdcm_Si111_1stXtalrtd = EpicsSignalRO('XF:05IDA-OP:1{Mono:HDCM-C:111_1}T-I', name='hdcm_Si111_1stXtal_rtd')
hdcm_Si111_2ndXtal_rtd = EpicsSignalRO('XF:05IDA-OP:1{Mono:HDCM-C:111_2}T-I', name='hdcm_Si111_2ndXtal_rtd')
hdcm_1stXtal_ThermStab_rtd = EpicsSignalRO('XF:05IDA-OP:1{Mono:HDCM-C:111}T:TC-I', name='hdcm_1stXtal_ThermStab_rtd')   
hdcm_ln2out_rtd = EpicsSignalRO('XF:05IDA-OP:1{Mono:HDCM}T:LN2Out-I', name='hdcm_ln2out_rtd')
hdcm_water_rtd = EpicsSignalRO('XF:05IDA-OP:1{Mono:HDCM-H2O}T-I', name='hdcm_water_rtd')
    
dBPM_h = EpicsSignalRO('XF:05ID-BI:1{BPM:01}:PosX:MeanValue_RBV', name='dBPM_h')
dBPM_v = EpicsSignalRO('XF:05ID-BI:1{BPM:01}:PosY:MeanValue_RBV', name='dBPM_v')
dBPM_t = EpicsSignalRO('XF:05ID-BI:1{BPM:01}:Current1:MeanValue_RBV', name='dBPM_t')
dBPM_i = EpicsSignalRO('XF:05ID-BI:1{BPM:01}:Current2:MeanValue_RBV', name='dBPM_i')
dBPM_o = EpicsSignalRO('XF:05ID-BI:1{BPM:01}:Current3:MeanValue_RBV', name='dBPM_o')
dBPM_b = EpicsSignalRO('XF:05ID-BI:1{BPM:01}:Current4:MeanValue_RBV', name='dBPM_b')   

def cryo_test(numpt = 5, delay_time = 2, preamp_acqtime = 0.5):
     
        
    det = [current_preamp, cryo_v19, cryo_lt19, cryo_pt1, 
           hdcm_Si111_1stXtalrtd, hdcm_Si111_2ndXtal_rtd,  hdcm_1stXtal_ThermStab_rtd, hdcm_ln2out_rtd, hdcm_water_rtd,
           dBPM_h, dBPM_v, dBPM_t, dBPM_i, dBPM_o, dBPM_b]
           
    current_preamp.exp_time.put(preamp_acqtime)

    livecallbacks = []    
    livetableitem = [current_preamp.ch0, current_preamp.ch2, cryo_v19, cryo_lt19, cryo_pt1, 
           hdcm_Si111_1stXtalrtd, hdcm_Si111_2ndXtal_rtd,  hdcm_1stXtal_ThermStab_rtd, hdcm_ln2out_rtd, hdcm_water_rtd,
           dBPM_h, dBPM_v, dBPM_t, dBPM_i, dBPM_o, dBPM_b]
    livecallbacks.append(LiveTable(livetableitem))
    
    for det_item in livetableitem:
        liveploty = det_item
        liveplotfig = plt.figure(det_item.name)
        livecallbacks.append(LivePlot(liveploty, fig=liveplotfig))
    
    yield from subs_wrapper(count(det, num=numpt, delay = delay_time), livecallbacks)
    
