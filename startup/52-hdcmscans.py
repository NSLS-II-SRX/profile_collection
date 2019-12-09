print(f'Loading {__file__}...')

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
# ------------------------------------------------------------------- #
"""
Created on Wed Jun 17 17:03:46 2015
new script for doing HDCM C1Roll and C2X calibration automatically
it requires the HDCM Bragg is calibrated and the d111 and dBragg in SRXenergy script are up-to-date

converting to be compatible with bluesky, still editing
"""
import SRXenergy  # Not used in this file
from epics import caget  # caput/get should not be used
from epics import caput
from epics import PV  # PV should probably be changed to EpicsSignalRO
import time  # time.sleep should be changed to bps.sleep if used in plan
import string
from matplotlib import pyplot
import subprocess 
import scipy as sp
import scipy.optimize
import math
import numpy as np
import srxbpm  # Not used in this file

def hdcm_c1roll_c2x_calibration():
    onlyplot = False
    #startTi = False
    usecamera = True
    endstation = False
    
    numAvg = 10
    
    
    print(energy._d_111)
    print(energy._delta_bragg)
    
    if endstation == False:  #default BPM1
        q=38690.42-36449.42 #distance of observing point to DCM; here observing at BPM1
        camPixel=0.006 #mm
        expotimePV = 'XF:05IDA-BI:1{BPM:1-Cam:1}AcquireTime'
    else:
        q=(62487.5+280)-36449.42 #distance of observing point to DCM; here observing at 28 cm downstream of M3. M3 is at 62.4875m from source
        camPixel=0.00121 #mm
        expotimePV = 'XF:05IDD-BI:1{Mscp:1-Cam:1}AcquireTime'
    
    
    
    if onlyplot == False:    
    
        if endstation == True:
            cenxPV= 'XF:05IDD-BI:1{Mscp:1-Cam:1}Stats1:CentroidX_RBV'
            cenyPV= 'XF:05IDD-BI:1{Mscp:1-Cam:1}Stats1:CentroidY_RBV'
        else:        
            cenxPV= 'XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:CentroidX_RBV'
            cenyPV= 'XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:CentroidY_RBV'
    
        bragg_rbv = PV('XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr.RBV')
        bragg_val = PV('XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr.VAL')
    
    
        ctmax = PV('XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:MaxValue_RBV')
        expo_time = PV('XF:05IDA-BI:1{BPM:1-Cam:1}AcquireTime_RBV')
        
        umot_go = PV('SR:C5-ID:G1{IVU21:1-Mtr:2}Sw:Go')
        
        #know which edges to go to
        #if startTi == True:    
        #    elementList=['Ti', 'Fe', 'Cu', 'Se']
        #else:
        #    elementList=['Se', 'Cu', 'Fe', 'Ti']
      
        if endstation == False:
            #if dcm_bragg.position > 15:   
            if bragg_rbv.get() > 15:
                #elementList=['Ti', 'Cr', 'Fe', 'Cu', 'Se']  
                #Ti requires exposure times that would require resetting the 
                #threshold in the stats record
                elementList=['Cr', 'Fe', 'Cu', 'Se']  
            else:
                #elementList=['Se', 'Cu', 'Fe', 'Cr', 'Ti']
                elementList=['Se', 'Cu', 'Fe', 'Cr']
        else:
            if bragg_rbv.get() > 13:
            #if dcm_bragg.position > 13:     
                elementList=['Ti', 'Cr', 'Fe', 'Cu', 'Se']  
            else:
                    elementList=['Se', 'Cu', 'Fe', 'Cr', 'Ti']
        
             
        energyDic={'Cu':8.979, 'Se': 12.658, 'Fe':7.112, 'Ti':4.966, 'Cr':5.989}
        harmonicDic={'Cu':5, 'Se': 5, 'Fe':3, 'Ti':3, 'Cr':3}            #150 mA, 20151007
        
        #use for camera option
        expotime={'Cu':0.003, 'Fe':0.004, 'Se':0.005, 'Ti':0.015, 'Cr':0.006}  #250 mA, 20161118, BPM1
        #expotime={'Cu':0.005, 'Fe':0.008, 'Se':0.01, 'Ti':0.03, 'Cr':0.0012}  #150 mA, 20151110, BPM1
        #expotime={'Cu':0.1, 'Fe':0.2, 'Se':0.2, 'Cr': 0.3}  #150 mA, 20151007, end-station
        
        #use for bpm option    
        foilDic={'Cu':25.0, 'Se': 0.0, 'Fe':25.0, 'Ti':25}
        
        centroidX={}
        centroidY={}
        
        theoryBragg=[]
        dx=[]
        dy=[]
    
        
        C2Xval=caget('XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr.VAL')
        C1Rval=caget('XF:05IDA-OP:1{Mono:HDCM-Ax:R1}Mtr.VAL')
        
        
        #dBragg=SRXenergy.whdBragg()
        dBragg = energy._delta_bragg
        
        for element in elementList:
            centroidXSample=[]
            centroidYSample=[]
    
            print(element)
            E=energyDic[element]
            print('Edge:', E)
            
            
            energy.move_c2_x.put(False)
            energy.move(E,wait=True)
#            energy.set(E)
#
#            while abs(energy.energy.position - E) > 0.001 :        
#                time.sleep(1)

            print('done moving energy')            
            #BraggRBV, C2X, ugap=SRXenergy.EtoAll(E, harmonic = harmonicDic[element])
            
            #print BraggRBV
            #print ugap
            #print C2X, '\n' 
        
            #go to the edge
    
            #ugap_set=PV('SR:C5-ID:G1{IVU21:1-Mtr:2}Inp:Pos')
            #ugap_rbv=PV('SR:C5-ID:G1{IVU21:1-LEnc}Gap')
            
#            print 'move undulator gap to:', ugap
            #ivu1_gap.move(ugap)    
#            ugap_set.put(ugap, wait=True)
#            umot_go.put(0)
#            time.sleep(10)
    
#            while (ugap_rbv.get() - ugap) >=0.01 :
#                time.sleep(5)                        
#            time.sleep(2)
            
#            print 'move Bragg to:', BraggRBV
#            bragg_val.put(BraggRBV, wait= True)
#            while (bragg_rbv.get() - BraggRBV) >=0.01 :
#                time.sleep(5)
            #dcm_bragg.move(BraggRBV)
#            time.sleep(2)
     
            if usecamera == True:
                caput(expotimePV, expotime[element]) 
                while ctmax.get() <= 200:
                    caput(expotimePV, expo_time.get()+0.001)
                    print('increasing exposuring time.')
                    time.sleep(0.6)
                while ctmax.get() >= 180:
                    caput(expotimePV, expo_time.get()-0.001) 
                    print('decreasing exposuring time.')
                    time.sleep(0.6)    
                print('final exposure time =' + str(expo_time.get()))
                print('final max count =' + str(ctmax.get()))
                
                
                #record the centroids on BPM1 camera            
                print('collecting positions with', numAvg, 'averaging...')
                for i in range(numAvg):
                    centroidXSample.append(caget(cenxPV))
                    centroidYSample.append(caget(cenyPV))
                    time.sleep(2)
                if endstation == False:    
                    centroidX[element] = sum(centroidXSample)/len(centroidXSample)
                else:
                    #centroidX[element] = 2452-sum(centroidXSample)/len(centroidXSample)
                    centroidX[element] = sum(centroidXSample)/len(centroidXSample)
    
                centroidY[element] = sum(centroidYSample)/len(centroidYSample)
                  
                print(centroidXSample)
                print(centroidYSample)
                #print centroidX, centroidY
                
                #centroidX[element]=caget(cenxPV)
                #centroidY[element]=caget(cenyPV)
                dx.append(centroidX[element]*camPixel)
                dy.append(centroidY[element]*camPixel)
            
                print(centroidX)
                print(centroidY, '\n')
            #raw_input("press enter to continue...")
            
    #        else:      
    #            
    #            bpm1_y.move(foilDic[element])
    #            time.sleep(2)
    #            position=bpm1.Pavg(Nsamp=numAvg)
    #            dx.append(position['H'])
    #            dy.append(position['V'])   
    #            print dx
    #            print dy
            
            theoryBragg.append(energy.bragg.position+dBragg)
                        
        #fitting
            
            #fit centroid x to determine C1roll
            #fit centroid y to determine C2X
    
    
        if endstation == True:        
            temp=dx
            dx=dy
            dy=temp
    
        print('C2Xval=', C2Xval)
        print('C1Rval=', C1Rval)
        print('dx=', dx)
        print('dy=', dy)
        print('theoryBragg=', theoryBragg)
    
    else:
        C1Rval=caget('XF:05IDA-OP:1{Mono:HDCM-Ax:R1}Mtr.VAL')
        C2Xval=caget('XF:05IDA-OP:1{Mono:HDCM-Ax:X2}Mtr.VAL')
    
    fitfunc = lambda pa, x: pa[1]*x+pa[0]  
    errfunc = lambda pa, x, y: fitfunc(pa,x) - y
    
    pi=math.pi
    sinBragg=np.sin(np.array(theoryBragg)*pi/180)
    sin2Bragg=np.sin(np.array(theoryBragg)*2*pi/180)
    print('sinBragg=', sinBragg)
    print('sin2Bragg=', sin2Bragg)
    
    
    guess = [dx[0], (dx[-1]-dx[0])/(sinBragg[-1]-sinBragg[0])]
    fitted_dx, success = sp.optimize.leastsq(errfunc, guess, args = (sinBragg, dx))
    print('dx=', fitted_dx[1], '*singBragg +', fitted_dx[0])
    
    droll=fitted_dx[1]/2/q*1000 #in mrad
    print('current C1Roll:', C1Rval)
    print('current C1Roll is off:', -droll)
    print('calibrated C1Roll:', C1Rval + droll, '\n')
    
    sin2divBragg = sin2Bragg/sinBragg
    print('sin2divBragg=', sin2divBragg)
    
    
    guess = [dy[0], (dy[-1]-dy[0])/(sin2divBragg[-1]-sin2divBragg[0])]
    fitted_dy, success = sp.optimize.leastsq(errfunc, guess, args = (sin2divBragg, dy))
    print('dy=', fitted_dy[1], '*(sin2Bragg/sinBragg) +', fitted_dy[0])
    print('current C2X:', C2Xval)
    print('current C2X corresponds to crystal gap:', fitted_dy[1])
    
    pyplot.figure(1)
    pyplot.plot(sinBragg, dx, 'b+')
    pyplot.plot(sinBragg, sinBragg*fitted_dx[1]+fitted_dx[0], 'k-')
    pyplot.title('C1Roll calibration')
    pyplot.xlabel('sin(Bragg)')
    if endstation == False:
        pyplot.ylabel('dx at BPM1 (mm)')
    else:
        pyplot.ylabel('dx at endstation (mm)')
    pyplot.show()        
    
    pyplot.figure(2)
    pyplot.plot(sin2divBragg, dy, 'b+')
    pyplot.plot(sin2divBragg, sin2divBragg*fitted_dy[1]+fitted_dy[0], 'k-')
    pyplot.title('C2X calibration')
    pyplot.xlabel('sin(2*Bragg)/sin(Bragg)')
    if endstation == False:
        pyplot.ylabel('dy at BPM1 (mm)')
    else:
        pyplot.ylabel('dy at endstation (mm)')
    pyplot.show()
