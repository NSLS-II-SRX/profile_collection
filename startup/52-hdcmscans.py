print(f'Loading {__file__}...')

import numpy as np
import string
import matplotlib.pyplot as plt
import subprocess
import scipy as sp
import scipy.optimize
import lmfit
import threading
from scipy.optimize import curve_fit

from bluesky.callbacks.mpl_plotting import QtAwareCallback


'''

This program provides functionality to calibrate HDCM energy:
    With provided XANES scan rstuls, it will calculate their edge DCM location
    It will then fit the E vs Bragg RBV with four values,
    provide fitting results: dtheta, dlatticeSpace

#1. collected xanes at 3-5 different energies
    e.g. Ti(5 keV), Fe(7 keV), Cu (9 keV), Se (12 keV)
    They can be in XRF mode or in transmission mode;
    note the scan id in bluesky
#2. setup the scadid in scanlogDic dictionary
    scanlogDic = {'Fe': 264, 'Ti': 265, 'Cr':267, 'Cu':271, 'Se': 273}
#3. pass scanlogDic to braggcalib()
    braggcalib(scanlogDic = scanlogDic, use_xrf = True)
    currently, the xanes needs to be collected on roi1 in xrf mode
'''

def mono_calib(Element, acqtime=1.0):
    """
    SRX mono_calib(Element)

    Go to the edge of the specified element, do a peakup, setroi, and automatic perform a xanes scan (+-50eV) on the specified element

    Parameters
    ----------
    Element : str

    Returns
    -------
    None

    Examples
    --------
    >>> mono_calib('V')

    """

    getemissionE(Element)
    EnergyX = getbindingE(Element)
    energy.move(EnergyX)
    setroi(1,Element)
    yield from bps.sleep(5)
    yield from peakup_fine(use_calib=False)
    yield from xanes_plan(erange=[EnergyX-50,EnergyX+50],estep=[1.0], samplename=f'{Element}Foil',filename=f'{Element}Foilstd',acqtime=acqtime, shutter=True)

def scanderive(xaxis, yaxis):
    dyaxis = np.gradient(yaxis, xaxis)
    edge = xaxis[dyaxis.argmin()]

    fig, ax = plt.subplots()
    # p = plt.plot(xaxis, dyaxis, '-')
    ax.plot(xaxis, dyaxis, '-')
    # ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    p = ax.plot(edge, dyaxis[dyaxis.argmin()], '*r', markersize=25)

    return p, xaxis, dyaxis, edge


def find_edge(scanid=-1, use_xrf=True, element=''):
    tbl = db.get_table(db[scanid], stream_name='primary')
    braggpoints = np.array(tbl['energy_bragg'])
    energypoints = np.array(tbl['energy_energy_setpoint'])

    if use_xrf is False:
        it = np.array(tbl['sclr_it'])
        i0 = np.array(tbl['sclr_i0'])
        tau = it / i0
        norm_tau = (tau - tau[0]) / (tau[-1] - tau[0])
        mu = -1 * np.log(np.abs(norm_tau))
    else:
        if (element == ''):
            print('Please send the element name')
        else:
            try:
                ch_name = 'Det1_' + element + '_ka1'
                mu = tbl[ch_name]
                ch_name = 'Det2_' + element + '_ka1'
                mu = mu + tbl[ch_name]
                ch_name = 'Det3_' + element + '_ka1'
                mu = mu + tbl[ch_name]
                ch_name = 'Det4_' + element + '_ka1'
                mu = mu + tbl[ch_name]
                mu = np.array(mu)
            except Exception:
                ch_name = 'ROI_01'
                mu = tbl[ch_name]
                ch_name = 'ROI_02'
                mu = mu + tbl[ch_name]
                ch_name = 'ROI_03'
                mu = mu + tbl[ch_name]
                ch_name = 'ROI_04'
                mu = mu + tbl[ch_name]
                mu = np.array(mu)

    p, xaxis, yaxis, edge = scanderive(braggpoints, mu)
    Ep, Exaxis, Eyaxis, Eedge = scanderive(energypoints, mu)

    return p, xaxis, yaxis, edge, Eedge


def braggcalib(scanlogDic={}, use_xrf=True, man_correction={}):
    # If scanlogDic is empty, we will use this hard coded dictionary
    # 2019-1 Apr 23

    if (scanlogDic == {}):
        scanlogDic = {'V':  26058,
                      'Cr': 26059,
                      'Se': 26060,
                      'Zr': 26061}

    fitfunc = lambda pa, x: (12.3984 /
                             (2 * pa[0] * np.sin((x + pa[1]) * np.pi / 180)))
    errfunc = lambda pa, x, y: fitfunc(pa, x) - y

    energyDic = {'Cu': 8.979, 'Se': 12.658, 'Zr': 17.998, 'Nb': 18.986,
                 'Ti': 4.966, 'Cr': 5.989, 'Co': 7.709, 'V': 5.465,
                 'Ni': 8.333, 'Fe': 7.112, 'Mn': 6.539}
    BraggRBVDic = {}
    EnergyRBVDic = {}
    fitBragg = []
    fitEnergy = []

    for element in scanlogDic:
        print(scanlogDic[element])
        current_scanid = scanlogDic[element]

        p, xaxis, yaxis, edge, Eedge = find_edge(scanid=current_scanid,
                                          use_xrf=use_xrf,
                                          element=element)

        BraggRBVDic[element] = round(edge, 6)
        EnergyRBVDic[element] = round(Eedge, 6)

        # plt.show(p)
        fitBragg.append(BraggRBVDic[element])
        fitEnergy.append(energyDic[element])
        
        if element in man_correction:
            fitBragg[-1] = man_correction[element]
            man_correction.pop(element)
        
        # print('Edge position is at Bragg RBV: \t ', BraggRBVDic[element])
        print('Edge position is at Bragg RBV: \t ', fitBragg[-1])
        # print('Edge position is at Energy RBV (not calibrated): \t ', EnergyRBVDic[element])
        # print('Difference in Energy in Edge position: \t', EnergyRBVDic[element]-energyDic[element]*1000)


    fitEnergy = np.sort(fitEnergy)
    fitBragg = np.sort(fitBragg)[-1::-1]

    guess = [3.1356, 0.32]
    fitted_dcm, success = sp.optimize.leastsq(errfunc,
                                              guess,
                                              args=(fitBragg, fitEnergy))

    print('(111) d spacing:\t', fitted_dcm[0])
    print('Bragg RBV offset:\t', fitted_dcm[1])
    print('Success:\t', success)

    newEnergy = fitfunc(fitted_dcm, fitBragg)

    print(fitBragg)
    print(newEnergy)

    plt.figure(1)
    plt.plot(fitBragg, fitEnergy, 'b^', label='raw scan')
    bragg = np.linspace(fitBragg[0], fitBragg[-1], 200)
    plt.plot(bragg, fitfunc(fitted_dcm, bragg), 'k-', label='fitting')
    plt.legend()
    plt.xlabel('Bragg RBV (deg)')
    plt.ylabel('Energy (keV)')

    pyplot.show()
    print('(111) d spacing:', fitted_dcm[0])
    print('Bragg RBV offset:', fitted_dcm[1])


class PairedCallback(QtAwareCallback):
    def __init__(self, scaler, dcm_c2_pitch_name, pitch_guess, *args, **kwargs):
        super().__init__(use_teleporter=kwargs.pop('use_teleporter', None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        def setup():
            fig, ax = plt.subplots()
            self.ax = ax
            fig.canvas.set_window_title('Peakup')
            self.ax.clear()
            # Setup LiveCallbacks
            self.live_plot = LivePlot(scaler, dcm_c2_pitch_name,
                                      linestyle='', marker='*', color='C0',
                                      label='raw',
                                      ax=self.ax,
                                      use_teleporter=False)

            # Setup LiveFit
            # f_gauss(x, A, sigma, x0, y0, m)
            model = lmfit.Model(f_gauss, ['x'])
            init_guess = {'A': lmfit.Parameter('A', 100000, min=0),
                          'sigma': lmfit.Parameter('sigma', 0.001, min=0),
                          'x0': lmfit.Parameter('x0', pitch_guess),
                          'y0': lmfit.Parameter('y0', 0, min=0),
                          'm': lmfit.Parameter('m', 0, vary=False)}
            self.lf = LiveFit(model, scaler, {'x': dcm_c2_pitch_name}, init_guess)
            self.lpf = LiveFitPlot(self.lf, ax=self.ax, color='r', use_teleporter=False, label='Gaussian fit')

        self.__setup = setup

    def start(self, doc):
        self.__setup()
        self.live_plot.start(doc)
        self.lpf.start(doc)
        super().start(doc)

    def descriptor(self, doc):
        self.live_plot.descriptor(doc)
        self.lpf.descriptor(doc)
        super().descriptor(doc)

    def event(self, doc):
        self.live_plot.event(doc)
        self.lpf.event(doc)
        super().event(doc)

    def event_page(self, doc):
        self.live_plot.event_page(doc)
        self.lpf.event_page(doc)
        super().event_page(doc)

    def stop(self, doc):
        self.live_plot.stop(doc)
        self.lpf.stop(doc)
        super().stop(doc)


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
    det = [sclr1, bpm4, dcm.c1_roll, dcm.c2_pitch]

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
    pitch_lim = (2.0, 4.0)
    pitch_num = 51

    # Find approximate values
    # 2020-02-03
    roll_guess = 0.071  # For getting X-rays to nanoKB
    # 2020-07-20
    roll_guess = 0.121
    # 2020-10-26
    roll_guess = 0.351
    # 2020-02-03
    B = energy.energy_to_positions((E/1000), 3, 0)[0]
    pitch_guess = 0.0009145473*B + 0.0141488665
    # 2020-10-26
    B = energy.energy_to_positions((E/1000), 3, 0)[0]
    pitch_guess = 0.0010913788*B - 0.0139213806


    # Use calibration
    if (use_calib):
        if (fix_roll):
           yield from bps.mov(dcm.c1_roll, roll_guess)
        if (fix_pitch):
            yield from bps.mov(dcm.c2_pitch, pitch_guess)
            yield from bps.mov(dcm.c2_pitch_kill, 1.0)

    # Set counting time
    sclr1.preset_time.put(1.0)

    # Open the shutter
    # if (shutter == True):
    #     yield from bps.mov(shut_b, 'Open')
    yield from check_shutters(shutter, 'Open')

    paired_callback = PairedCallback(scaler, dcm.c2_pitch.name, pitch_guess)

    # Run the C2 pitch fine scan
    # @subs_decorator(livecallbacks)
    # @subs_decorator(lpf)
    @subs_decorator(paired_callback)
    def myplan():
        return (
            # yield from scan(det,
            #                 dcm.c2_fine,
            #                 pitch_lim[0],
            #                 pitch_lim[1],
            #                 pitch_num)
            yield from adaptive_scan(det, 'sclr_i0', dcm.c2_fine,
                                     pitch_lim[0], pitch_lim[1],
                                     0.01, 0.1, 10000, True)
        )
    uid = yield from myplan()

    # Close the shutter
    # if (shutter is True):
    #     yield from bps.mov(shut_b, 'Close')
    yield from check_shutters(shutter, 'Close')

    # Add scan to scanlog
    logscan('peakup_fine_pitch')

    # Display results of livefit
    print(paired_callback.lf.result.values)

    # Collect the data
    # h = db[-1]
    h = db[uid]
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
        print('No optimized parameters found. Try scanning a larger range.')
        print('Returning to original values.')

        # Return to original values
        yield from bps.mov(dcm.c1_fine, rf1_default)
        yield from bps.mov(dcm.c1_roll, total_roll)
        yield from bps.mov(dcm.c2_fine, pf2_default)
        yield from bps.mov(dcm.c2_pitch, total_pitch)
        yield from bps.mov(dcm.c2_pitch_kill, 1.0)

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
    if (plot is True):
        # plt.figure('Peakup')
        x_plot = np.linspace(x[0], x[-1], num=101)
        y_plot = f_gauss(x_plot, *popt)
        paired_callback.ax.plot(x_plot, y_plot, 'C0--', label='fit')
        paired_callback.ax.plot((pitch_new, pitch_new), (y_min, y_max), '--k', label='max')
        paired_callback.ax.legend()


def ic_energy_batch(estart, estop, npts,
                    acqtime=1.0, count_pts=50, outfile=None):
    if (outfile is not None):
        outfile = '%s_SRXionchamber_readings.csv' % (datetime.datetime.now().strftime('%Y%m%d'))
        outdir = '/home/xf05id1/current_user_data/'
        ion_chamber_fp = open(outdir + outfile, 'w')
        ion_chamber_fp.write('# Energy, ICM, IC0, ICT\n')

    # Setup scaler and open shutter
    yield from abs_set(sclr1.preset_time, acqtime)
    yield from bps.mov(shut_b, 'Open')

    for i in np.linspace(estart, estop, num=npts):
        yield from mv(energy, i)
        yield from bps.sleep(10)
        yield from peakup_fine(shutter=False)
        yield from bps.sleep(10)
        yield from count([sclr1], num=count_pts)

        if (outfile is not None):
            tbl = db[-1].table()
            icm_mean = tbl['sclr_im'].mean
            ic0_mean = tbl['sclr_i0'].mean
            ict_mean = tbl['sclr_it'].mean
            ion_chamber_fp.write('%8.0f, %d, %d, %d\n' % (i, icm_mean, ic0_mean, ict_mean))

    # Close the shutter
    yield from bps.mov(shut_b, 'Close')

    if (outfile is not None):
        ion_chamber_fp.close()


def hdcm_bragg_temperature(erange, estep, dwell, N, dt=0):
    # Loop to test the bragg temperature during a XANES scan

    # Convert erange and estep to numpy array
    ept = np.array([])
    erange = np.array(erange)
    estep = np.array(estep)
    # Calculation for the energy points
    for i in range(len(estep)):
        ept = np.append(ept, np.arange(erange[i], erange[i+1], estep[i]))
    ept = np.append(ept, np.array(erange[-1]))

    dets = [dcm.temp_pitch, energy.energy, dcm.bragg, dcm.c1_roll, dcm.c2_pitch, bpm4]
    dets_by_name = [d.name
                    for d in dets]
    if ('bpm4' in dets_by_name):
        dets_by_name.append('bpm4_x')
        dets_by_name.append('bpm4_y')

    livecallbacks = LiveTable(dets_by_name)

    def custom_perstep(detectors, motor, step):
        def move():
            grp = _short_uid('set')
            yield Msg('checkpoint')
            yield Msg('set', motor, step, group=grp)
            yield Msg('wait', None, group=grp)

        yield from move()
        yield from bps.sleep(dwell)
        yield from trigger_and_read(list(detectors) + [motor])

    @subs_decorator({'all' : livecallbacks})
    def myscan():
        yield from list_scan(dets, energy, list(ept), per_step=custom_perstep)

    for i in range(N):
        yield from myscan()
        yield from bps.sleep(dt)

    return


# braggcalib(use_xrf=True)
# ------------------------------------------------------------------- #
"""
Created on Wed Jun 17 17:03:46 2015
new script for doing HDCM C1Roll and C2X calibration automatically
it requires the HDCM Bragg is calibrated and the d111 and dBragg in SRXenergy script are up-to-date

converting to be compatible with bluesky, still editing
"""
# import SRXenergy  # Not used in this file
import time  # time.sleep should be changed to bps.sleep if used in plan
import string
from matplotlib import pyplot
import subprocess
import scipy as sp
import scipy.optimize
import math
import numpy as np
# import srxbpm  # Not used in this file

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
        # expotimePV = 'XF:05IDA-BI:1{BPM:1-Cam:1}AcquireTime'
        expotimePV = bpmAD.cam.acquire_time
    else:
        q=(62487.5+280)-36449.42 #distance of observing point to DCM; here observing at 28 cm downstream of M3. M3 is at 62.4875m from source
        camPixel=0.00121 #mm
        # expotimePV = 'XF:05IDD-BI:1{Mscp:1-Cam:1}AcquireTime'
        expotimePV = hfvlmAD.cam.acquire_time



    if onlyplot == False:

        if endstation == True:
            # cenxPV= 'XF:05IDD-BI:1{Mscp:1-Cam:1}Stats1:CentroidX_RBV'
            # cenyPV= 'XF:05IDD-BI:1{Mscp:1-Cam:1}Stats1:CentroidY_RBV'
            cenxPV = hfvlmAD.stats1.centroid.x
            cenyPV = hfvlmAD.stats1.centroid.y
        else:
            # cenxPV= 'XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:CentroidX_RBV'
            # cenyPV= 'XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:CentroidY_RBV'
            cenxPV = bpmAD.stats1.centroid.x
            cenyPV = bpmAD.stats1.centroid.y

        # bragg_rbv = PV('XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr.RBV')
        # bragg_val = PV('XF:05IDA-OP:1{Mono:HDCM-Ax:P}Mtr.VAL')
        bragg_rbv = dcm.bragg.user_readback
        bragg_val = dcm.bragg.user_setpoint


        # ctmax = PV('XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:MaxValue_RBV')
        ctmax = bpmAD.stats1.max_value
        # expo_time = PV('XF:05IDA-BI:1{BPM:1-Cam:1}AcquireTime_RBV')
        expo_time = bpmAD.cam.acquire_time

        # PV not found
        # only usage is commented out
        # umot_go = PV('SR:C5-ID:G1{IVU21:1-Mtr:2}Sw:Go')

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


        C2Xval = dcm.c2_x.user_setpoint.get()
        C1Rval = dcm.c1_roll.user_setpoint.get()


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
                expotimePV.put(expotime[element])
                while ctmax.get() <= 200:
                    expotimePV.put(expo_time.get() + 0.001)
                    print('increasing exposuring time.')
                    time.sleep(0.6)
                while ctmax.get() >= 180:
                    expotimePV.put(expo_time.get() - 0.001)
                    print('decreasing exposuring time.')
                    time.sleep(0.6)
                print('final exposure time =' + str(expo_time.get()))
                print('final max count =' + str(ctmax.get()))


                #record the centroids on BPM1 camera
                print('collecting positions with', numAvg, 'averaging...')
                for i in range(numAvg):
                    centroidXSample.append(cenxPV.get())
                    centroidYSample.append(cenyPV.get())
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
        # Is this different from definition above?
        C2Xval = dcm.c2_x.user_setpoint.get()
        C1Rval = dcm.c1_roll.user_setpoint.get()

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
