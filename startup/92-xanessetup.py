# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:16:52 2016

set up for 2D XRF scan for HR mode
@author: xf05id1
"""


from bluesky.plans import AbsListScanPlan
from bluesky.suspenders import PVSuspendFloor
import scanoutput
import numpy

ring_current_pv = 'SR:C03-BI{DCCT:1}I:Real-I'
cryo_v19_pv = 'XF:05IDA-UT{Cryo:1-IV:19}Sts-Sts'
i0_baseline = 1.53e-7
it_baseline = 9.34e-8

def xanes_afterscan(scanuid, roinum, filename, i0scale, itscale):

    headeritem = [] 
    h=db[-1]

    #datatable = get_table(h, ['energy_energy', 'saturn_mca_rois_roi'+str(roinum)+'_net_count', 'current_preamp_ch1'])        
    #energy_array = list(datatable['energy_energy'])   
    #if_array = list(datatable['saturn_mca_rois_roi'+str(roinum)+'_net_count'])
    #i0_array = list(datatable['current_preamp_ch1'])
    
    userheaderitem = {}
    userheaderitem['sample.name'] = h.start['sample']['name']
    
    
    #columnitem = ['energy_energy','saturn_mca_rois_roi'+str(roinum)+'_net_count', 'current_preamp_ch1']
    columnitem = ['energy_energy','saturn_mca_rois_roi'+str(roinum)+'_net_count','saturn_mca_rois_roi'+str(roinum)+'_count', 'current_preamp_ch0', 'current_preamp_ch1']    
    #columnitem = ['energy_energy','saturn_mca_rois_roi'+str(roinum)+'_count', 'current_preamp_ch1']    
    
    usercolumnitem = {}
 
    usercolumnnameitem = ['scaled_current_preamp_ch0', 'scaled_current_preamp_ch1']
    datatable = get_table(h, ['current_preamp_ch0', 'current_preamp_ch1'])        
    i0_array = abs(numpy.array(datatable['current_preamp_ch1']) - i0_baseline) * i0scale
    it_array = abs(numpy.array(datatable['current_preamp_ch0']) - it_baseline) * itscale
   
    usercolumnitem['scaled_current_preamp_ch1'] = i0_array
    usercolumnitem['scaled_current_preamp_ch0'] = it_array
    
    scanoutput.textout(header = headeritem, userheader = userheaderitem, column = columnitem, 
                       usercolumn = usercolumnitem, usercolumnname = usercolumnnameitem, output = False, filename_add = filename) 
    
    #ps.append(PeakStats(energy.energy.name, bpmAD.stats3.total.name))

def xanes(erange = [], estep = [],  
            harmonic = None, correct_c2_x=True,              
            acqtime=None, roinum=0, i0scale = 1e8, itscale = 1e8,
            samplename = '', filename = ''):
                
    '''
    roinum: setup the roi to be used to calculate the XANES spectrum

    '''                                
                
    #make sure user provided correct input

    if erange is []:
        raise Exception('erange = [], must specify energy ranges')
    if estep is []:
        raise Exception('estep = [], must specify energy step sizes')
    if len(erange)-len(estep) is not 1:
        raise Exception('must specify erange and estep correctly.'\
                         +'e.g. erange = [7000, 7100, 7150, 7500], estep = [2, 0.5, 5] ')

    if acqtime is None:
        raise Exception('acqtime = None, must specify an acqtime position')

    
    #convert erange and estep to numpy array
    erange = numpy.array(erange)
    estep = numpy.array(estep)

    #calculation for the energy points        
    ept = numpy.array([])
    for i in range(len(estep)):
        ept = numpy.append(ept, numpy.arange(erange[i], erange[i+1], estep[i]))
    ept = numpy.append(ept, numpy.array(erange[-1]))
    ept = ept/1000
   
    
    #setup the detector
    current_preamp.exp_time.put(acqtime)
    saturn.mca.preset_real_time.put(acqtime)
    #saturn.mca.preset_live_time.put(acqtime)

    saturn.read_attrs.append('mca.rois.roi'+str(roinum)+'.net_count')       
    saturn.read_attrs.append('mca.rois.roi'+str(roinum)+'.count') 
    det = [current_preamp, saturn, ring_current]        

    #setup the live callbacks
    livecallbacks = []    
    
    livetableitem = ['energy_energy', 'current_preamp_ch0', 'current_preamp_ch1']    
    livetableitem.append('saturn_mca_rois_roi'+str(roinum)+'_net_count')
    livetableitem.append('saturn_mca_rois_roi'+str(roinum)+'_count')
    livecallbacks.append(LiveTable(livetableitem, max_post_decimal = 4))

    liveploty = 'saturn_mca_rois_roi'+str(roinum)+'_net_count'
    #liveploty = 'saturn_mca_rois_roi'+str(roinum)+'_count'
    liveplotx = energy.energy.name
    liveplotfig = plt.figure()
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))
      

    if correct_c2_x is False:
        energy.move_c2_x.put(False)
    else:
    #this line sets the current xoffset to the energy axis, might want to see if that's what we want to do in the long run
        energy._xoffset = energy.crystal_gap()
        
    if harmonic is not None:        
        energy.harmonic.put(harmonic)

    #add user meta data
    gs.RE.md['sample']  = {'name': samplename}

    #setup the plan  
    xanes_scanplan = AbsListScanPlan(det, energy, ept)

    #run the plan
    scaninfo = gs.RE(xanes_scanplan, livecallbacks)

    #output the datafile
    xanes_afterscan(scaninfo, roinum, filename, i0scale, itscale)

    #clean up when the scan is done    
    energy.move_c2_x.put(True)
    energy.harmonic.put(None)
              
    return scaninfo