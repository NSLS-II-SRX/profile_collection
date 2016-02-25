## -*- coding: utf-8 -*-
#"""
#Created on Wed Feb 17 17:10:05 2016
#
#@author: xf05id1
#"""
#
##might want to put in configuration file
#import scanoutput
#from bluesky.plans import AbsScanPlan
#from bluesky.scientific_callbacks import PeakStats
#from bluesky.scientific_callbacks import plot_peak_stats
#from imp import reload
#import matplotlib.pylab as plt
#import sys
#
#bpmAD.cam.read_attrs = ['acquire_time']
#bpmAD.configuration_attrs = ['cam']
#
##sys.path.append('/nfs/xf05id1/src/srxtools/')
#
##Oleg specifically
#energy.move_u_gap.put(False)
#energy.move_c2_x.put(False)
#energy.u_gap.read_attrs.append('elevation')
##bpmAD.read_attrs.append('cam_acquire_time')
##olegplan=AbsScanPlan([bpmAD, pu, ring_current], energy, 7.792, 8.143, 20)   #testing
##olegplan=AbsScanPlan([bpmAD, pu, ring_current], energy, 7.792, 8.143, 176)  #for 5th harmonic at ugap = 6.8
#olegplan=AbsScanPlan([bpmAD, pu, ring_current], energy, 14.1, 14.6, 251)     #for 9th harmonic at ugap = 6.8
#
#livetableitem = [energy.energy, bpmAD.stats1.total, bpmAD.stats3.total, ring_current]
#liveploty = bpmAD.stats3.total.name
#liveplotx = energy.energy.name
#liveplotfig = plt.figure()
#
#        
##ps.append(PeakStats(energy.energy.name, bpmAD.stats3.total.name))
#
##for executing the current plan
##RE(olegplan, [LiveTable(livetableitem), LivePlot(liveploty, x=liveplotx, fig=liveplotfig), ps[-1]])
##figureofmerit=oleg_afterscan(ps)
#
#
##def ud_crab_plan(pu, us_u, us_l, ds_u, ds_l, other_dets):
#
#
################################################
#
##from bluesky.callbacks import CallbackBase
##
##ps=[]  
##
### custom peakstats factory
##def oleg_peakstats(scan):
##    "Set up peakstats"
##    cur_ps = PeakStats(energy.energy.name, bpmAD.stats3.total.name)
##    ps.append(cur_ps)
##    return cur_ps
##
### manage running on run stop
##class OlegPostScan(CallbackBase):
##
##    def stop(self, doc):
##        cur_ps = ps[-1]
##        plt.figure()
##        oleg_afterscan(cur_ps)
##
##
##post_oleg = OlegPostScan()
##
##olegplan.subs['all'].append(post_oleg)
##olegplan.subs['all'].append(LiveTable(livetableitem))
##olegplan.subs['all'].append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))
#
#def oleg_afterscan(cur_ps):
#    plot_peak_stats(cur_ps) 
#
#    headeritem = ['pu_us_upper_readback','pu_us_lower_readback','pu_ds_upper_readback','pu_ds_lower_readback',\
#                  'energy_u_gap_elevation_ct_us','energy_u_gap_elevation_offset_us',\
#                  'pu_elevation_avg_elevation', 'pu_elevation_ds_elevation', 'pu_elevation_readback', \
#                  'energy_u_gap_readback'] 
#    maxenergy = cur_ps.max[0]
#    maxintensity = cur_ps.max[1]
#    fwhm = cur_ps.fwhm
#    
#    #extract the ring current for the max intensity point
##    ring_current_array = []
##    energy_array = []
#    h=db[-1]
##    events=list(get_events(h))
##    for event in events:
##        ring_current_array.append(event['data']['ring_current'])
##        energy_array.append(event['data']['energy_energy'])
#    datatable = get_table(h, ['energy_energy', 'ring_current'])        
#    energy_array = list(datatable.energy_energy)   
#    ring_current_array = list(datatable.ring_current)    
#    maxcurrent = ring_current_array[energy_array.index(maxenergy)]  
#    
#    userheaderitem = {}
#    userheaderitem['maxenergy'] = maxenergy
#    userheaderitem['maxintensity'] = maxintensity
#    userheaderitem['fwhm'] = fwhm
#    userheaderitem['maxcurrent'] = maxcurrent
#    userheaderitem['scaled_maxintensity'] = maxintensity/maxcurrent
#
#    columnitem = ['energy_energy', 'energy_bragg_user_readback', 'bpmAD_stats1_total', 'bpmAD_stats2_total', 'bpmAD_stats3_total', 'ring_current']
#
#    scanoutput.textout(header = headeritem, userheader = userheaderitem, column = columnitem, output = False) 
#    #ps.append(PeakStats(energy.energy.name, bpmAD.stats3.total.name))
#
#
#
#
#    return maxenergy, maxintensity, fwhm , maxcurrent
#
##def u_opt():
##    for target in [3.415, 3.42]:
##        yield from ud_crab_plan(pu, target, target, target, target, [ut])
##        cur_ps = PeakStats(energy.energy.name, bpmAD.stats3.total.name)
##        ps.append(cur_ps)
##        token = yield Msg('subscribe', None, 'all', cur_ps)
##        yield from olegplan
##        yield Msg('unsubscribe', None, token) 
#
##def run_oleg():
##
##    # make a local plan
##    local_plan = u_opt()
##    # make a local callback list
##    #cbs = [LiveTable(livetableitem),
##    #       LivePlot(liveploty, x=liveplotx, fig=liveplotfig)]
##    # actually run the plan with the callbacks
##    #gs.RE(local_plan, cbs)
##    gs.RE(local_plan)
#
#
#################################################3
#   
##Some Global Variables can be defined here
##E.g. some acceptable limits of the parameters to be optimized, etc.
#
#
#def undSpecKPP(_tilt_microrad, _taper_microm, _elev_microm): 
#    
#    #Some validation of _tilt_microrad, _taper_microm, _elev_microm can be placed here ?
#    
#    ylim = 0.099
#    
#    #Getting re-calculated direct settings for IVU axes:
#    yUU_mm, yUL_mm, yDU_mm, yDL_mm, elev_mm = und_combined_motion(_tilt_microrad, _taper_microm, _elev_microm)
#    if abs(yUU_mm-yDU_mm) > ylim:
#        err_msg = 'y upper axes difference exceeds the limit '+str(ylim)
#        raise Exception(err_msg)
#
#    if abs(yUL_mm-yDL_mm) > ylim:
#        err_msg = 'y lower axes difference exceeds the limit '+str(ylim)
#        raise Exception(err_msg)
#        
#    print('we can move undulator')
#    
#    #move undulator
#    uplan_taper_tilt=ud_crab_plan(pu, yUU_mm, yUL_mm, yDU_mm, yDL_mm, [])
#    gs.RE(uplan_taper_tilt)
#    pu.elevation.set(elev_mm)
#        
#    #mono scan
#    ps = PeakStats(energy.energy.name, bpmAD.stats3.total.name)    
#    #current_pre = ring_current.get()
#    gs.RE.md['undulator_setup']  = {'tilt': _tilt_microrad,  'taper': _taper_microm, 'elevation': _elev_microm}
#    gs.RE(olegplan, [LiveTable(livetableitem), LivePlot(liveploty, x=liveplotx, fig=liveplotfig, legend_keys=['undulator_setup']), ps])    
#    #current_post = ring_current.get()
#    
#    maxenergy, maxintensity, fwhm, maxcurrent = oleg_afterscan(ps)
#    #current_avg = (current_pre + current_post)/2
#
#    #to-dos:
#        #improve key performance paramters calculation
#        #scale results by ring current
#
#    print ('scaled maxintensity: '+str(maxintensity/maxcurrent)+'\n')
#    print ('fwhm               : '+str(fwhm)+'\n')
#    print ('maxintensity       : '+str(maxintensity)+'\n')
#    print ('maxcurrent         : '+str(maxcurrent)+'\n')
#    print ('maxenergy          : '+str(maxenergy)+'\n')
#    
#    return maxintensity/maxcurrent, fwhm, maxenergy, maxcurrent
#
#        
# 
