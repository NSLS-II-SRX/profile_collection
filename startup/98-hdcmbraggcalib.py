import numpy
import string
from matplotlib import pyplot
import subprocess 
import scipy as sp
import scipy.optimize
# import x3toAthenaSetup as xa
# import srxpeak

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


def scanderive(xaxis, yaxis): 
    
    # length=len(xaxis)
    # dxaxis=xaxis[0:-1]
    # dyaxis=yaxis[0:-1]
    
    # for i in range(0,length-1):
    #     dxaxis[i]=(xaxis[i]+xaxis[i+1])/2.
    #     dyaxis[i]=(yaxis[i+1]-yaxis[i])/(xaxis[i+1]-xaxis[i])
		#print "Deriv. max value is ",dyaxis.max()," at ", dxaxis[dyaxis.argmax()]
		#print "Deriv. min value is ",dyaxis.min()," at ", dxaxis[dyaxis.argmin()]
		#pyplot.plot(dxaxis,dyaxis,'+')
    dyaxis = numpy.gradient(yaxis, xaxis)
    p = pyplot.plot(xaxis, dyaxis, '-')
    # p=pyplot.plot(dxaxis,dyaxis*(-1),'-')
    #make the useoffset = False
    ax = pyplot.gca()
    ax.ticklabel_format(useOffset=False)
    edge = xaxis[dyaxis.argmin()]
    p = pyplot.plot(edge, dyaxis[dyaxis.argmin()], '*r', markersize=25)
    #edge = dxaxis[dyaxis.argmax()]

    return p, xaxis,dyaxis, edge

def find_edge(scanid = -1, use_xrf = False, element = ''):
    #baseline = -8.5e-10
    baseline_it = 4e-9
    table = db.get_table(db[scanid], stream_name='primary')
    #bluesky.preprocessors
    braggpoints = table.energy_bragg

    if use_xrf is False:
        #it = table.current_preamp_ch0
        it = table.sclr_it
        #i0 = table.current_preamp_ch2        
        #normliazedit = -numpy.log(numpy.array(it[1::])/abs(numpy.array((i0[1::])-baseline)))
        mu = -numpy.log(abs(numpy.array(it[1::])-baseline_it))

    else:
        #mu = table.xs_channel2_rois_roi01_value_sum
        # mu = table[table.keys()[12]]
        # mu = table[table.keys()[10]]
        if (element is ''):
            print('Please send the element name')
        else:
            ch_name = 'Det1_' + element + '_ka1'
            mu = table[ch_name]
            ch_name = 'Det2_' + element + '_ka1'
            mu = mu + table[ch_name]
            ch_name = 'Det3_' + element + '_ka1'       
            mu = mu + table[ch_name]
        
    p, xaxis, yaxis, edge = scanderive(numpy.array(braggpoints), numpy.array(mu))

    return p, xaxis, yaxis, edge

def braggcalib(scanlogDic = {}, use_xrf = False):
#    
    #2016-2 July
    #scanlogDic = {'Fe': 264, 'Ti': 265, 'Cr':267, 'Cu':271, 'Se': 273}    
    #scanlogDic = {'Fe': 264, 'Ti': 265, 'Cr':266}

    #2016-2 Aug 15, after cryo tripped due to water intervention on power dip on 8/14/2016
    #scanlogDic = {'Fe': 1982, 'Cu':1975, 'Cr': 1984, 'Ti': 1985, 'Se':1986}

    #2016-3 Oct 3
    #scanlogDic = {'Se':20}

    #2018-1 Jan 26
    #scanlogDic = {'Fe': 11256, 'Cu':11254, , 'Ti': 11260, 'Se':11251}
    # 2018-1 Feb 24
    # scanlogDic = {'Ti': 12195, 'Fe': 12194, 'Se':12187}

    # 2018-2 Jun 5
    # scanlogDic = {'Fe' : 14476,
    #               'V'  : 14477,
    #               'Cr' : 14478,
    #               'Cu' : 14480,
    #               'Se' : 14481,
    #               'Zr' : 14482}

    # 2018-3 Oct 2
    # if (scanlogDic == {}):
    #     scanlogDic = {'V'  : 18037,
    #                   'Cr' : 18040,
    #                   'Fe' : 18043,
    #                   'Cu' : 18046,
    #                   'Se' : 18049,
    #                   'Zr' : 18052}

    # 2019-1 Feb 5
    # if (scanlogDic == {}):
    #     scanlogDic = {'V'  : 21828,
    #                   'Cr' : 21830,
    #                   'Fe' : 21833,
    #                   'Cu' : 21835,
    #                   'Se' : 21838,
    #                   'Zr' : 21843}

    # 2019-1 Apr 23 
    if (scanlogDic == {}):
        scanlogDic = {'V'  : 26058,
                      'Cr' : 26059,
                      'Se' : 26060,
                      'Zr' : 26061}
    fitfunc = lambda pa, x: 12.3984/(2*pa[0]*numpy.sin((x+pa[1])*numpy.pi/180))  
    errfunc = lambda pa, x, y: fitfunc(pa,x) - y

    energyDic={'Cu':8.979, 'Se': 12.658, 'Zr':17.998, 'Nb':18.986, 'Fe':7.112, 
               'Ti':4.966, 'Cr': 5.989, 'Co': 7.709, 'V': 5.465, 'Mn':6.539,
               'Ni':8.333}
    BraggRBVDic={}
    fitBragg=[]
    fitEnergy=[]

    for element in scanlogDic:
        print(scanlogDic[element])
        
        current_scanid = scanlogDic[element]
        p, xaxis, yaxis, edge = find_edge(scanid = current_scanid, use_xrf = use_xrf, element = element)
            
        BraggRBVDic[element] = round(edge, 6)
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


# braggcalib(use_xrf=True)

