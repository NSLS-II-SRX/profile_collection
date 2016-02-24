# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:10:05 2016

@author: xf05id1
"""

#might want to put in configuration file
import scanoutput
from bluesky.plans import AbsScanPlan
from bluesky.scientific_callbacks import PeakStats
from bluesky.scientific_callbacks import plot_peak_stats
from imp import reload
import matplotlib.pylab as plt
import sys
#sys.path.append('/nfs/xf05id1/src/srxtools/')

#Oleg specifically
energy.move_u_gap.put(False)
energy.move_c2_x.put(False)
energy.u_gap.read_attrs.append('elevation')
olegplan=AbsScanPlan([bpmAD, pu], energy, 8.143, 7.792, 20)
#olegplan=AbsScanPlan([bpmAD, pu], energy, 8.143, 7.792, 176)
livetableitem = [energy.energy, bpmAD.stats1.total, bpmAD.stats3.total]
liveploty = bpmAD.stats3.total.name
liveplotx = energy.energy.name
liveplotfig = plt.figure()

ps=[]          
ps.append(PeakStats(energy.energy.name, bpmAD.stats3.total.name))

#for executing the current plan
#RE(olegplan, [LiveTable(livetableitem), LivePlot(liveploty, x=liveplotx, fig=liveplotfig), ps[-1]])
#figureofmerit=oleg_afterscan(ps)


#def ud_crab_plan(pu, us_u, us_l, ds_u, ds_l, other_dets):

def u_opt():
    psresults =[]
    for target in [3.415, 3.42]:
        #yield from ud_crab_plan(pu, target, target, target, target, [ut])
        yield from olegplan
        psout = oleg_afterscan(ps)
        psresults.append(psout)      
        

# gs.RE(play())
#RE(u_opt(), [LiveTable(livetableitem), LivePlot(liveploty, x=liveplotx, fig=liveplotfig), ps[-1]])

# gs.RE(play())
# gs.RE(play())
# gs.RE(play())

        
def oleg_afterscan(ps):
    plot_peak_stats(ps[-1])   

    headeritem =  ['pu_us_upper_readback','pu_us_lower_readback','pu_ds_upper_readback','pu_ds_lower_readback',\
                               'energy_u_gap_elevation_ct_us','energy_u_gap_elevation_offset_us',\
                               'energy_u_gap_readback'] 
  
    maxenergy = ps[-1].max[0]
    maxintensity = ps[-1].max[1]    

    userheaderitem = {}
    userheaderitem['maxenergy'] = maxenergy
    userheaderitem['maxintensity'] = maxintensity

    columnitem = ['energy_energy', 'bpmAD_stats1_total', 'bpmAD_stats2_total', 'bpmAD_stats3_total']

    scanoutput.textout(header = headeritem, userheader = userheaderitem, column = columnitem)    
    ps.append(PeakStats(energy.energy.name, bpmAD.stats3.total.name))
    return userheaderitem   
