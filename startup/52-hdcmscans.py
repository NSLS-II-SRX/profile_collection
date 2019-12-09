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


# Simple Gaussian
def gaussian(x, A, sigma, x0):
    return A*np.exp(-(x - x0)**2/(2 * sigma**2))

# More complete Gaussian with offset and slope
def f_gauss(x, A, sigma, x0, y0, m):
    return y0 + m*x + A*np.exp(-(x - x0)**2/(2 * sigma**2))

# Integral of the Gaussian function with slope and offset
def f_int_gauss(x, A, sigma, x0, y0, m):
    x_star = (x - x0) / sigma
    return A * erf(x_star / np.sqrt(2)) + y0 + m*x

def peakup_dcm(correct_roll=True, plot=False, shutter=True, use_calib=False):
    """

    Scan the HDCM fine pitch and, optionally, roll against the ion chamber in the D Hutch

    correct_roll    <Bool>      If True, align the beam in the vertical (roll)
    plot            <Bool>      If True, plot the intensity as a function of pitch/roll
    shutter         <Bool>      If True, the shutter will be automatically opened/closed
    use_calib       <Bool>      If True, use a previous calibration as an initial guess

    """

    e_value=energy.energy.get()[1]
    pitch_old = dcm.c2_pitch.position
    roll_old = dcm.c1_roll.position

    det = [sclr1]

    ps = PeakStats(dcm.c2_pitch.name, im.name)
    ps1 = PeakStats(dcm.c1_roll.name, im.name)

    if (shutter ==  True):
        RE(mv(shut_b,'Open'))

    # Turn off the ePID loop for the pitch motor
    # 'XF:05IDD-CT{FbPid:02}PID:on'
    c2_pid=EpicsSignal("XF:05IDD-CT{FbPid:02}PID:on")
    c2_pid.put(0)  # Turn off the ePID loop
    c2_V=EpicsSignal("XF:05ID-BI{EM:BPM1}DAC1")
    c2_V.put(3.0)  # Reset the piezo voltage to 3 V

    # pitch_lim = (-19.320, -19.370)
    # pitch_num = 51
    # roll_lim = (-4.9, -5.14)
    # roll_num = 45

    # pitch_lim = (-19.375, -19.425)
    # pitch_num = 51
    # roll_lim = (-4.9, -5.6)
    # roll_num = 51

    pitch_lim = (pitch_old-0.025, pitch_old+0.025)
    roll_lim = (roll_old-0.2, roll_old+0.2)

    pitch_num = 51
    roll_num = 51

    if (use_calib):
        # Factor to convert eV to keV
        K = 1
        if (e_value > 1000):
            K = 1 / 1000
        # Pitch calibration
        pitch_guess = -0.00055357 * K * e_value - 19.39382381
        dcm.c2_pitch.move(pitch_guess, wait=True)
        # Roll calibration
        roll_guess  = -0.01124286 * K * e_value - 4.93568571
        dcm.c1_roll.move(roll_guess, wait=True)
        # Output guess
        print('\nMoving to guess:')
        print('\tC2 Pitch: %f' % (pitch_guess))
        print('\tC1 Roll:  %f\n' % (roll_guess))

    #if e_value < 10.:
    #    sclr1.preset_time.put(0.1)
    #    RE(scan([sclr1], dcm.c2_pitch, -19.335, -19.305, 31), [ps])
    #else:
    #    sclr1.preset_time.put(1.)
    #    RE(scan([sclr1], dcm.c2_pitch, -19.355, -19.310, 46), [ps])
    if e_value < 14.:
        # sclr1.preset_time.put(0.1)
        sclr1.preset_time.put(1.0)
    else:
        sclr1.preset_time.put(1.0)

    if (plot == True):
        sclr1.preset_time.put(1.0)  # Let's collect a longer scan since we're plotting it
        RE(scan(det, dcm.c2_pitch, pitch_lim[0], pitch_lim[1], pitch_num), [ps])
        print('Pitch: Centroid at %f\n\n' % (ps.cen))
        plt.figure()
        plt.plot(ps.x_data, ps.y_data, label='Data')
        plt.plot((ps.cen, ps.cen), (np.amin(ps.y_data), np.amax(ps.y_data)), '--k', label='Centroid')
        plt.xlabel('HDCM C2 PITCH')
        plt.ylabel('Counts')
        plt.legend()
    else:
        RE(scan(det, dcm.c2_pitch, pitch_lim[0], pitch_lim[1], pitch_num), [ps])

    time.sleep(0.5)
    dcm.c2_pitch.move(ps.cen, wait=True)
    time.sleep(0.5)
    if (np.abs(dcm.c2_pitch.position - ps.cen) > 0.001):
        print('The pitch motor did not move on the first try. Trying again...', end='')
        dcm.c2_pitch.move(ps.cen, wait=True)
        if (np.abs(dcm.c2_pitch.position - ps.cen) > 0.001):
            print('FAIL! Check motor location.\n')
        else:
            print('OK\n')
    logscan('peakup_pitch')
    c2pitch_kill.put(1)

    if correct_roll == True:
        if (plot == True):
            sclr1.preset_time.put(1.0)  # If we are debugging, let's collect a longer scan
            RE(scan(det, dcm.c1_roll, roll_lim[0], roll_lim[1], roll_num), [ps1])
            # print('Roll: Maximum flux at %f' % (ps1.max[0]))
            print('Roll: Centroid at %f\n\n' % (ps1.cen))
            plt.figure()
            plt.plot(ps1.x_data, ps1.y_data, label='Data')
            plt.plot((ps1.cen, ps1.cen), (np.amin(ps1.y_data), np.amax(ps1.y_data)), '--k', label='Centroid')
            plt.xlabel('HDCM ROLL')
            plt.ylabel('Counts')
            plt.legend()
        else:
            RE(scan(det, dcm.c1_roll, roll_lim[0], roll_lim[1], roll_num), [ps1])

        time.sleep(0.5)
        dcm.c1_roll.move(ps1.cen,wait=True)
        time.sleep(0.5)
        if (np.abs(dcm.c1_roll.position - ps1.cen) > 0.001):
            print('The roll motor did not move on the first try. Trying again...', end='')
            dcm.c1_roll.move(ps1.cen, wait=True)
            if (np.abs(dcm.c1_roll.position - ps1.cen) > 0.001):
                print('FAIL! Check motor location.\n')
            else:
                print('OK\n')
        logscan('peakup_roll')

    # Output old/new values
    print('Old pitch value:\t%f' % pitch_old)
    print('New pitch value:\t%f' % ps.cen)
    print('Current pitch value:\t%f' % dcm.c2_pitch.position)
    print('Old roll value: \t%f' % roll_old)
    print('New roll value: \t%f' % ps1.cen)
    print('Current roll value: \t%f\n' % dcm.c1_roll.position)

    if (shutter == True):
        RE(mv(shut_b,'Close'))

    #for some reason we now need to kill the pitch motion to keep it from overheating.  6/8/17
    #this need has disappeared mysteriously after the shutdown - gjw 2018/01/19
    # This has now reappeared - amk 2018/06/06
    time.sleep(1)
    c2pitch_kill.put(1)
    c2_pid.put(1)  # Turn on the ePID loop


from scipy.optimize import curve_fit

def peakup_fine(scaler='sclr_i0', plot=True, shutter=True, use_calib=True,
                fix_roll=True, fix_pitch=True):
    """

    Scan the HDCM C2 Piezo Motor to optimize the beam.

    scaler      <String>    Define which scaler channel to maximize on ('sclr_i0', 'sclr_im')
    plot        <Bool>      If True, plot the results
    shutter     <Bool>      If True, the shutter is automatically opened/closed
    use_calib   <Bool>      If True, use lookup table as an initial guess
    fix_roll    <Bool>      If True, peakup C1 roll piezo
    fix_pitch   <Bool>      If True, peakup C2 pitch piezo

    """

    # Get the energy in eV
    E = energy.energy.get()[1]
    if (E < 1000):
        E = E * 1000

    # Define the detector
    det = [sclr1, dcm.c1_roll, dcm.c2_pitch]

    # Set the roll piezo to its default value (3.0)
    # and return the roll to its original value
    rf1_default = 3.0
    total_roll = dcm.c1_roll.position
    # yield from bps.mov(dcm.c1_fine, rf1_default)
    # yield from bps.mov(dcm.c1_roll, total_roll)

    # Set limits
    roll_lim = (2.5, 3.5)
    roll_num = 51

    # Turn off the ePIC loop for the pitch motor
    # c2_pid = EpicsSignal('XF:05IDD-CT{FbPid:02}PID:on')
    # yield from bps.mov(c2_pid, 0)
    yield from bps.mov(dcm.c2_fine.pid_enabled, 0)
    yield from dcm.c2_fine.reset_pid()

    # Set the pitch piezo to its default value (3.0)
    # and return the pitch to its original value
    pf2_default = 3.0
    total_pitch = dcm.c2_pitch.position
    yield from bps.mov(dcm.c2_fine, pf2_default)
    yield from bps.mov(dcm.c2_pitch, total_pitch)
    yield from bps.sleep(1)
    yield from bps.mov(dcm.c2_pitch_kill, 1.0)

    # Set limits
    pitch_lim = (2.5, 3.5)
    pitch_num = 51

    # Use calibration
    if (use_calib):
        # 2018-06-28
        # roll_guess = -0.01124286 * (E/1000) - 4.93568571
        # 2019-02-14
        # roll_guess = -0.00850813 * (E/1000) - 5.01098505
        # 2019-02-14
        # roll_guess = -0.01661758 * (E/1000) - 5.09654066
        # 2019-08-29
        # roll_guess = 0.000
        # 2019-08-29
        # roll_guess = 0.00921 * (E/1000) - 0.380612
        # 2019-11-12
        roll_guess = -0.295
        yield from bps.mov(dcm.c1_roll, roll_guess)
        # 2019-02-14
        # pitch_guess = -0.00106066 * (E/1000) - 19.37338813
        # 2019-04-24
        # pitch_guess = -0.00202462 * (E/1000) - 17.57951692
        # 2019-11-12
        B = energy.energy_to_positions((E/1000), 3, 0)[0]
        pitch_guess = 0.000611 * B + 0.002945
        yield from bps.mov(dcm.c2_pitch, pitch_guess)
        yield from bps.mov(dcm.c2_pitch_kill, 1.0)

    # Set counting time
    sclr1.preset_time.put(1.0)

    # Open the shutter
    if (shutter == True):
        yield from bps.mov(shut_b, 'Open')

    # Setup LiveCallbacks
    plt.figure('Peakup')
    plt.clf()
    livecallbacks = [LivePlot(scaler, dcm.c2_pitch.name,
                              linestyle='', marker='*', color='C0',
                              label='raw',
                              fig=plt.figure('Peakup'))]

    # Run the C2 pitch fine scan
    @subs_decorator(livecallbacks)
    def myplan():
        yield from scan(det, dcm.c2_fine, pitch_lim[0], pitch_lim[1], pitch_num)
    yield from myplan()

    # Close the shutter
    if (shutter == True):
        yield from bps.mov(shut_b, 'Close')

    # Add scan to scanlog
    logscan('peakup_fine_pitch')

    # Collect the data
    h = db[-1]
    # x = h.table()['c2_fine_readback'].values
    x = h.table()['dcm_c2_pitch'].values
    y = h.table()[scaler].values

    # Fit the data
    # gaussian(x, A, sigma, x0):
    y_min = np.amin(y)
    y_max = np.amax(y)
    x_loc = np.argmax(y)
    try:
        popt, _ = curve_fit(f_gauss, x, y, p0=[y_max, 0.001, x[x_loc], 0, 0])
        pitch_new = popt[2]
        print('Maximum flux found at %.4f' % (pitch_new))
    except RuntimeError:
        print('No optimized parameters found.')
        print('Scanning a larger range.')

        # Move total pitch to its original value
        yield from bps.mov(dcm.c2_fine, pf2_default)
        yield from bps.mov(dcm.c2_pitch, total_pitch)
        yield from bps.mov(dcm.c2_pitch_kill, 1.0)

        # Set extended pitch limits
        pitch_lim = (2.0, 4.0)
        pitch_num = 101

        # Set counting time
        sclr1.preset_time.put(1.0)

        # Open the shutter
        if (shutter == True):
            yield from bps.mov(shut_b, 'Open')

        # Run the C2 pitch fine scan
        yield from scan(det, dcm.c2_fine, pitch_lim[0], pitch_lim[1], pitch_num)

        # Close the shutter
        if (shutter == True):
            yield from bps.mov(shut_b, 'Close')

        # Add scan to scanlog
        logscan('peakup_fine_pitch')

        # Collect the data
        h = db[-1]
        # x = h.table()['c2_fine_readback'].values
        x = h.table()['dcm_c2_pitch'].values
        y = h.table()[scaler].values

        # Fit the data
        # gaussian(x, A, sigma, x0):
        y_min = np.amin(y)
        y_max = np.amax(y)
        x_loc = np.argmax(y)

        try:
            popt, _ = curve_fit(f_gauss, x, y, p0=[y_max, 0.001, x[x_loc], 0, 0])
            pitch_new = popt[2]
            print('Maximum flux found at %.4f' % (pitch_new))
        except RuntimeError:
            print('No optimized parameters found.')
            print('Returning to default.')

            pitch_new = total_pitch
            plot = False

    # Move to the maximum
    yield from bps.mov(dcm.c2_fine, pf2_default)
    yield from bps.sleep(1.0)
    ind = 0
    while (np.abs(dcm.c2_pitch.position - pitch_new) > 0.0005):
        yield from bps.mov(dcm.c2_pitch, pitch_new)
        yield from bps.sleep(1.0)
        ind = ind + 1
        if (ind > 5):
            print('Warning: C2 Fine motor might not be in correct location.')
            break

    # Get the new position and set the ePID to that
    yield from bps.mov(dcm.c2_pitch_kill, 1.0)

    # Reset the ePID-I value
    yield from dcm.c2_fine.reset_pid()
    yield from bps.sleep(1.0)
    yield from bps.mov(dcm.c2_fine.pid_enabled, 1)

    # Plot the results
    if (plot == True):
        plt.figure('Peakup')
        # plt.clf()
        # plt.xlabel('C2 Pitch [mrad]')
        # plt.ylabel(scaler + ' [cts]')
        # plt.plot(x, y, 'C0*', label='raw')
        x_plot = np.linspace(x[0], x[-1], num=101)
        y_plot = f_gauss(x_plot, *popt)
        plt.plot(x_plot, y_plot, 'C0--', label='fit')
        plt.plot((pitch_new, pitch_new), (y_min, y_max), '--k', label='max')
        plt.legend()


def ic_energy_batch(estart,estop,npts):
    ion_chamber_fp=open('/home/xf05id1/current_user_data/ionchamber_readings_'+time.strftime('%Y%m%d%H%M%S')+'.csv','w')
    ion_chamber_fp.write('#energy,I premirror,I sample,I transmittedi\n')
    for i in np.linspace(estart,estop,npts):
        energy.move(i)
        time.sleep(5)
        peakup_dcm()
        time.sleep(5)
        ion_chamber_fp.write('%8.0f,%d,%d,%d\n'%(i,im.get(),i0.get(),it.get()))
    ion_chamber_fp.close()



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
