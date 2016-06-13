# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:03:46 2015
new script for doing HDCM C1Roll and C2X calibration automatically
it requires the HDCM Bragg is calibrated and the d111 and dBragg in SRXenergy script are up-to-date

converting to be compatible with bluesky, still editing

@author: xf05id1
"""
import SRXenergy
from epics import caget
from epics import caput
from epics import PV
import time
import string
from matplotlib import pyplot
import subprocess 
import scipy as sp
import scipy.optimize
import math
import numpy as np
import srxbpm

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
                elementList=['Ti', 'Cr', 'Fe', 'Cu', 'Se']  
            else:
                    elementList=['Se', 'Cu', 'Fe', 'Cr', 'Ti']
        else:
            if bragg_rbv.get() > 13:
            #if dcm_bragg.position > 13:     
                elementList=['Ti', 'Cr', 'Fe', 'Cu', 'Se']  
            else:
                    elementList=['Se', 'Cu', 'Fe', 'Cr', 'Ti']
        
             
        energyDic={'Cu':8.979, 'Se': 12.658, 'Fe':7.112, 'Ti':4.966, 'Cr':5.989}
        harmonicDic={'Cu':5, 'Se': 5, 'Fe':3, 'Ti':3, 'Cr':3}            #150 mA, 20151007
        
        #use for camera option
        expotime={'Cu':0.005, 'Fe':0.008, 'Se':0.01, 'Ti':0.03, 'Cr':0.0012}  #150 mA, 20151110, BPM1
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
            energy.set(E)

            while abs(energy.energy.position - E) > 0.001 :        
                time.sleep(1)

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