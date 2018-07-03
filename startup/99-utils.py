import os
import numpy as np
from ophyd import EpicsSignal
from bluesky.plans import relative_scan
from bluesky.callbacks import LiveFit,LiveFitPlot
from bluesky.callbacks.fitting import PeakStats
from bluesky.plan_stubs import mv
import lmfit
import time

def cryofill(wait_time_after_v19_claose = 60*10):
    cryo_v19_possp = EpicsSignal('XF:05IDA-UT{Cryo:1-IV:19}Pos-SP', name='cryov19_possp')
    cryo_v19_possp.set(100)
    while abs(cryo_v19.get() - 1) > 0.05:
        cryo_v19_possp.set(100)
        time.sleep(2)

    time.sleep(5)
    while (cryo_v19.get() - 0) > 0.05:
        print('cryo cooler still refilling')
        time.sleep(5)
    cryo_v19_possp.set(0)
    print('waiting for', wait_time_after_v19_claose, 's', 'before taking data...')
    time.sleep(wait_time_after_v19_claose)

def breakdown(batch_dir=None, batch_filename=None,xstart=None,ystart=None,\
    xsteps=None,ysteps=None,xstepsize=None,ystepsize=None,zposition=None,\
    acqtime=None,numrois=None,xbasestep=39,ybasestep=39):
    '''
    helper function for hf2dxrf_xybath
    takes a large range with uniform step size and breaks it into chunks

    batch_dir (string): directory for the input batch file
    batch_filename (string): text file name that defines the set points for batch scans
    xstart (float): starting x position
    ystart (float): starting y position
    xsteps (int): steps in X
    ysteps (int): steps in Y
    xstepsize (float): scan step in X
    ystepsize (float): scan step in Y
    zposition (float or list of floats): position(s) in z
    acqtime (float): acquisition time
    numrois (int): number or ROIs
    xbasestep (int): number of X steps in each atomic sub-scan
    ybasestep (int): number of Y steps in each atomic sub-scan
    '''
    xchunks=np.ceil((xsteps+1)/(xbasestep+1))
    ychunks=np.ceil((ysteps+1)/(ybasestep+1))
    xoverflow=np.mod((xsteps+1),(xbasestep+1))-1
    yoverflow=np.mod((ysteps+1),(ybasestep+1))-1
    print('xdimension = '+str(xchunks))
    print('ydimension = '+str(ychunks))
    print('xoverflow = '+str(xoverflow))
    print('yoverflow = '+str(yoverflow))

    if zposition is None:
        zposition=[hf_stage.z.position]
    if zposition.__class__ is not list:
        zposition=[zposition]
    mylist=list()
    for k in zposition:
        for j in range(0,int(ychunks),1):
            for i in range(0,int(xchunks),1):
                xs= xstart+(xbasestep+1)*i*xstepsize
                ys= ystart+(ybasestep+1)*j*ystepsize
                if (ychunks > 1):
                    if ((j==ychunks-1) and (yoverflow >= 0)):
                        ysteps=yoverflow
                    else:
                        ysteps=ybasestep
                if (xchunks>1):
                    if((i==xchunks-1) and (xoverflow >= 0)):
                        xsteps=xoverflow
                    else:
                        xsteps=xbasestep

                mylist.append([k,xs,xsteps,xstepsize,ys,ysteps,ystepsize,\
                acqtime,numrois])
    if batch_dir is None:
        batch_dir = os.getcwd()
        print("No batch_dir was assigned, using the current directory")
    else:
        if not os.path.isdir(batch_dir):
            raise Exception(\
            "Please provide a valid batch_dir for the batch file path.")
    if batch_filename is None:
        raise Exception(\
        "Please provide a batch file name, e.g. batch_file = 'xrf_batch_test.txt'.")
    batchfile = batch_dir+'/'+batch_filename

    with open(batchfile, 'w') as batchf:
        for item in mylist:
            for entry in item:
                batchf.write('%s '%entry)
            batchf.write('\n')
    return mylist

def xybatch_grid(xstart, xstepsize, xnumstep, ystart, ystepsize, ynumstep):
    xylist = []
    for j in np.linspace(ystart, ystart+ystepsize*ynumstep, ynumstep+1):
        for i in np.linspace(xstart, xstart+xstepsize*xnumstep, xnumstep+1):
            xylist.append([i, j])
    return xylist

def gaussian(x, A, sigma, x0):
    return A*np.exp(-(x - x0)**2/(2 * sigma**2))

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

    ps = PeakStats(dcm.c2_pitch.name, i0.name)
    ps1 = PeakStats(dcm.c1_roll.name, i0.name)

    if (shutter ==  True):
        RE(mv(shut_b,'Open'))
    c2pitch_kill=EpicsSignal("XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Cmd:Kill-Cmd")

    # pitch_lim = (-19.320, -19.370)
    # pitch_num = 51
    # roll_lim = (-4.9, -5.14)
    # roll_num = 45

    pitch_lim = (-19.375, -19.425)
    pitch_num = 51
    roll_lim = (-4.9, -5.6)
    roll_num = 51

    if (use_calib):
        # Pitch calibration
        pitch_guess = -0.00055357 * e_value - 19.39382381
        dcm.c2_pitch.move(pitch_guess, wait=True)
        # Roll calibration
        roll_guess  = -0.01124286 * e_value - 4.93568571
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
    print('New roll value: \t%f\n' % ps1.cen)
    print('Current roll value: \t%f\n' % dcm.c1_roll.position)

    if (shutter == True):
        RE(mv(shut_b,'Close'))

    #for some reason we now need to kill the pitch motion to keep it from overheating.  6/8/17
    #this need has disappeared mysteriously after the shutdown - gjw 2018/01/19
    # This has now reappeared - amk 2018/06/06
    time.sleep(1)
    c2pitch_kill.put(1)


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


def retune_undulator():
    energy.detune.put(0.)
    energy.move(energy.energy.get()[0])

import skbeam.core.constants.xrf as xrfC

interestinglist = ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

elements = dict()
element_edges = ['ka1','ka2','kb1','la1','la2','lb1','lb2','lg1','ma1']
element_transitions = ['k', 'l1', 'l2', 'l3', 'm1', 'm2', 'm3', 'm4', 'm5']
for i in interestinglist:
    elements[i] = xrfC.XrfElement(i)

def setroi(roinum, element, edge=None, det=None):
    '''
    Set energy ROIs for Vortex SDD.  Selects elemental edge given current energy if not provided.
    roinum      [1,2,3]     ROI number
    element     <symbol>    element symbol for target energy
    edge                    optional:  ['ka1','ka2','kb1','la1','la2','lb1','lb2','lg1','ma1']
    '''
    cur_element = xrfC.XrfElement(element)
    if edge == None:
        for e in ['ka1','ka2','kb1','la1','la2','lb1','lb2','lg1','ma1']:
            if cur_element.emission_line[e] < energy.energy.get()[1]:
                edge = 'e'
                break
    else:
        e = edge

    e_ch = int(cur_element.emission_line[e] * 1000)
    if det is not None:
        det.channel1.set_roi(roinum, e_ch-100, e_ch+100, name=element + '_' + e)
    else:
        for d in [xs.channel1,xs.channel2,xs.channel3]:
            d.set_roi(roinum,e_ch-100,e_ch+100,name=element+'_'+e)
    print("ROI{} set for {}-{} edge.".format(roinum,element,e))


def clearroi(roinum=None):
    if roinum == None:
        roinum = [1, 2, 3]
    else:
        roinum = [roinum]

    # xs.channel1.rois.roi01.clear
    for roi in roinum:
        for d in [xs.channel1.rois, xs.channel2.rois, xs.channel3.rois]:
            if (1 in roinum):
                d.roi01.clear()  # set_roi(roi, 0, 0, name='')
            if (2 in roinum):
                d.roi02.clear()
            if (3 in roinum):
                d.roi03.clear()


def getemissionE(element,edge = None):
    cur_element = xrfC.XrfElement(element)
    if edge == None:
        print("edge\tenergy [keV]")
        for e in element_edges:
            if cur_element.emission_line[e] < 25. and cur_element.emission_line[e] > 1.:
                print("{0:s}\t{1:8.2f}".format(e,cur_element.emission_line[e]))
    else:
        return cur_element.emission_line[edge]


def getbindingE(element,edge=None):
    '''
    Return edge energy in eV if edge is specified, otherwise return K and L edge energies and yields
    element     <symbol>        element symbol for target
    edge        ['k','l1','l2','l3']    return binding energy of this edge
    '''
    if edge == None:
        y = [0.,'k']
        print("edge\tenergy [eV]\tyield")
        for i in ['k','l1','l2','l3']:
            print("{0:s}\t{1:8.2f}\t{2:5.3}".format(i,xrfC.XrayLibWrap(elements[element].Z,'binding_e')[i]*1000.,
                                                  xrfC.XrayLibWrap(elements[element].Z,'yield')[i]))
            if (y[0] < xrfC.XrayLibWrap(elements[element].Z,'yield')[i]
             and xrfC.XrayLibWrap(elements[element].Z,'binding_e')[i] < 25.):
                y[0] = xrfC.XrayLibWrap(elements[element].Z,'yield')[i]
                y[1] = i
        return xrfC.XrayLibWrap(elements[element].Z,'binding_e')[y[1]]*1000.
    else:
       return xrfC.XrayLibWrap(elements[element].Z,'binding_e')[edge]*1000.
def copyscanparam(src_num,dest_num):
    '''
    Copy all scan paramaters from scan src_num to scan dest_num
    wrapper for cp method in python scanrecord object
    '''
    src = 'scan{}'.format(src_num-1)
    dest = 'scan{}'.format(dest_num-1)
    scanrecord.cp(src,dest)

def printfig():
    plt.savefig('/home/xf05id1/tmp/temp.png', bbox_inches='tight',
    pad_inches=4)
    os.system("lp -d HXN-printer-1 /home/xf05id1/tmp/temp.png")

def estimate_scan_duration(xnum, ynum, dwell, scantype=None, event_delay=None):
    '''
    xnum    int     number of steps or points as entered on the command line for the scan in X
    ynum    int     number of steps or points as entered on the command line for the scan in Y
    dwell   float   exposure time in seconds as entered on the command line
    scantype    string  one of [XRF,XRF_fly,XANES]
    '''
    overhead={'xrf':0.7,'xrf_fly':3.8,'xanes':1.6}
    if event_delay == None:
        try:
            delay =  overhead[scantype.casefold()]
        except KeyError:
            print("Warning:  scantype is not supported")
            delay = 0.
    else:
        delay = event_delay

    if scantype.casefold() == 'xrf_fly':
        if delay is not 0.:
            delay = delay/xnum
        xnum = xnum - 1
        ynum = ynum - 1

    result = ( (xnum + 1) * (ynum + 1) ) * ( dwell + delay )
    div,rem = divmod(result,3600)
    print("Estimated duration is {0:d} hr {1:.1f} min ({2:.1f} sec).".format(int(div),rem/60,result))

    return result
