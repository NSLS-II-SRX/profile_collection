import numpy
import string
from matplotlib import pyplot
import subprocess 
import scipy as sp
import scipy.optimize
import x3toAthenaSetup as xa
import srxpeak

'''
    
This program provides functionality to calibrate hdcm energy:
    With provided xanes scan rstuls, it will calculate their edge DCM location
    It will then fit the E vs Bragg RBV with four values, provide fitting results: dtheta, dlatticeSpace
    
#1. collected xanes at 3-5 different energies - e.g. Ti(5 keV), Fe(7 keV), Cu (9 keV), Se (12 keV)
    They can be in xrf mode or in transmission mode; note the scan id in bluesky
#2. setup the scadid in scanlogDic dictionary
    scanlogDic = {'Fe': 264, 'Ti': 265, 'Cr':267, 'Cu':271, 'Se': 273}
#3. pass scanlogDic to braggcalib()
    braggcalib(scanlogDic = scanlogDic, use_xrf = True)
    currently, the xanes needs to be collected on roi1 in xrf mode
'''

from databroker import DataBroker as db, get_table, get_images, get_events

def scanderive(xaxis,yaxis): 
    
    length=len(xaxis)
    dxaxis=xaxis[0:-1]
    dyaxis=yaxis[0:-1]
    
    for i in range(0,length-1):
        dxaxis[i]=(xaxis[i]+xaxis[i+1])/2.
        dyaxis[i]=(yaxis[i+1]-yaxis[i])/(xaxis[i+1]-xaxis[i])
		#print "Deriv. max value is ",dyaxis.max()," at ", dxaxis[dyaxis.argmax()]
		#print "Deriv. min value is ",dyaxis.min()," at ", dxaxis[dyaxis.argmin()]
		#pyplot.plot(dxaxis,dyaxis,'+')
    p=pyplot.plot(dxaxis,dyaxis*(-1),'-')
    #make the useoffset = False
    ax = pyplot.gca()
    ax.ticklabel_format(useOffset=False)
    edge = dxaxis[dyaxis.argmin()]
    #edge = dxaxis[dyaxis.argmax()]

    return p, dxaxis,dyaxis, edge

def find_edge(scanid = -1, use_xrf = False):
    #baseline = -8.5e-10
    baseline_it = 4e-9
    table = get_table(db[scanid], stream_name='primary')
    
    braggpoints = table.energy_bragg

    if use_xrf is False:
        it = table.current_preamp_ch0
        #i0 = table.current_preamp_ch2        
        #normliazedit = -numpy.log(numpy.array(it[1::])/abs(numpy.array((i0[1::])-baseline)))
        mu = -numpy.log(abs(numpy.array(it[1::])-baseline_it))

    else:
        mu = table.xs_channel2_rois_roi01_value_sum
        
    p, xaxis, yaxis, edge = scanderive(numpy.array(braggpoints[1::]), numpy.array(mu[1::]))

    return p, xaxis, yaxis, edge

def braggcalib(scanlogDic = {}, use_xrf = False):
#    
    #2016-2 July
    #scanlogDic = {'Fe': 264, 'Ti': 265, 'Cr':267, 'Cu':271, 'Se': 273}    
    #scanlogDic = {'Fe': 264, 'Ti': 265, 'Cr':266}

    #2016-2 Aug 15, after cryo tripped due to water intervention on power dip on 8/14/2016
    #scanlogDic = {'Fe': 1982, 'Cu':1975, 'Cr': 1984, 'Ti': 1985, 'Se':1986}

    fitfunc = lambda pa, x: 12.3984/(2*pa[0]*numpy.sin((x+pa[1])*numpy.pi/180))  
    errfunc = lambda pa, x, y: fitfunc(pa,x) - y

    energyDic={'Cu':8.979, 'Se': 12.658, 'Zr':17.998, 'Nb':18.986, 'Fe':7.112, 
               'Ti':4.966, 'Cr': 5.989, 'Co': 7.709}
    BraggRBVDic={}
    fitBragg=[]
    fitEnergy=[]

    for element in scanlogDic:
        print(scanlogDic[element])
        current_scanid = scanlogDic[element]
        p, xaxis, yaxis, edge = find_edge(scanid = current_scanid, use_xrf = use_xrf)
            
        BraggRBVDic[element] = round(edge,3)
        print('Edge position is at Braggg RBV', BraggRBVDic[element])
        pyplot.show(p)
        
        fitBragg.append(BraggRBVDic[element])
        fitEnergy.append(energyDic[element])
    
    fitEnergy=numpy.sort(fitEnergy)
    fitBragg=numpy.sort(fitBragg)[-1::-1]
    
    guess = [3.1356, 0.32]
    fitted_dcm, success = sp.optimize.leastsq(errfunc, guess, args = (fitBragg, fitEnergy))
    
    print('(111) d spacing:', fitted_dcm[0])
    print('Bragg RBV offset:', fitted_dcm[1])
    print('success:', success)
    
    
    newEnergy=fitfunc(fitted_dcm, fitBragg)
    
    print(fitBragg)
    print(newEnergy)
    
    pyplot.figure(1)    
    pyplot.plot(fitBragg, fitEnergy,'b^', label = 'raw scan')
    bragg = numpy.linspace(fitBragg[0], fitBragg[-1], 200)
    pyplot.plot(bragg, fitfunc(fitted_dcm, bragg), 'k-', label = 'fitting')
    pyplot.legend()
    pyplot.xlabel('Bragg RBV (deg)')
    pyplot.ylabel('Energy(keV)')
    
    pyplot.show() 
    print('(111) d spacing:', fitted_dcm[0])
    print('Bragg RBV offset:', fitted_dcm[1])


