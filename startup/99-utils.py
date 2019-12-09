print(f'Loading {__file__}...')

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from ophyd import EpicsSignal
from ophyd.utils import make_dir_tree
from bluesky.plans import relative_scan
from bluesky.callbacks import LiveFit,LiveFitPlot
from bluesky.callbacks.fitting import PeakStats
from bluesky.plan_stubs import mv
import lmfit
import time

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

# Run a knife-edge scan
def knife_edge(motor, start, stop, stepsize, acqtime,
               fly=True, high2low=False, use_trans=True):
    """
    motor       motor   motor used for scan
    start       float   starting position
    stop        float   stopping position
    stepsize    float   distance between data points
    acqtime     float   counting time per step
    fly         bool    if the motor can fly, then fly that motor
    high2low    bool    scan from high transmission to low transmission
                        ex. start will full beam and then block with object (knife/wire)
    """

    # Set detectors
    det = [sclr1]
    if (use_trans == False):
        det.append(xs)

    # Set counting time
    sclr1.preset_time.put(1.0)
    if (use_trans == False):
        xs.settings.acquire_time.put(1.0)

    # Need to convert stepsize to number of points
    num = np.round((stop - start) / stepsize) + 1

    # Run the scan
    if (motor.name == 'hf_stage_y'):
        if fly:
            yield from y_scan_and_fly(start, stop, num,
                                      hf_stage.x.position, hf_stage.x.position+0.001, 1,
                                      acqtime)
        else:
            yield from scan(det, motor, start, stop, num)
    else:
        if fly:
            yield from scan_and_fly(start, stop, num,
                                    hf_stage.y.position, hf_stage.y.position+0.001, 1,
                                    acqtime)
        else:
            # table = LiveTable([motor])
            # @subs_decorator(table)
            # LiveTable([motor])
            yield from scan(det, motor, start, stop, num)

    # Get the information from the previous scan
    haz_data = False
    loop_counter = 0
    MAX_LOOP_COUNTER = 15
    print('Waiting for data...', end='', flush=True)
    while (loop_counter < MAX_LOOP_COUNTER):
        try:
            tbl = db[-1].table('stream0', fill=True)
            haz_data = True
            print('done')
            break
        except:
            loop_counter += 1
            time.sleep(1)

    # Check if we haz data
    if (not haz_data):
        print('Data collection timed out!')
        return
    
    # Get the position information
    if fly:
        pos = 'enc1'
    else:
        pos = motor.name
    # if (motor == hf_stage.y):
    #     pos = 'hf_stage_y'
    # elif (motor == hf_stage.x):
    #     pos = 'hf_stage_x'
    # else:
    #     pos = 'pos'

    # Get the data
    if (use_trans == True):
        y = tbl['it'].values[0] / tbl['im'].values[0]
    else:
        y = np.sum(np.array(tbl['fluor'])[0][:, :, 794:814], axis=(1, 2))
        y = y / np.array(tbl['i0'])[0]
    x = np.array(tbl[pos])[0]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    dydx = np.gradient(y, x)

    # Fit the raw data
    # def f_int_gauss(x, A, sigma, x0, y0, m)
    p_guess = [0.5*np.amax(y),
               0.001,
               0.5*(x[0] + x[-1]),
               np.amin(y) + 0.5*np.amax(y),
               0.001]
    if high2low:
        p_guess[0] = -0.5 * np.amin(y)
    try:
        popt, _ = curve_fit(f_int_gauss, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_int_gauss(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Display fit of raw data
    plt.figure('Raw')
    plt.clf()
    plt.plot(x, y, '*', label='Raw Data')
    plt.plot(x_plot, f_int_gauss(x_plot, *p_guess), '-', label='Guess fit')
    plt.plot(x_plot, y_plot, '-', label='Final fit')
    plt.legend()

    # Use the fitted raw data to fit a Gaussian
    # def f_gauss(x, A, sigma, x0, y0, m):
    try:
        if (high2low == True):
            p_guess = [np.amin(dydx_plot), popt[1], popt[2], 0, 0]
        else:
            p_guess = [np.amax(dydx_plot), popt[1], popt[2], 0, 0]

        popt2, _ = curve_fit(f_gauss, x_plot, dydx_plot, p0=p_guess)
        # popt2, _ = curve_fit(f_gauss, x, dydx, p0=p_guess)
    except:
        print('Fit failed.')
        popt2 = p_guess


    # Plot the fit
    plt.figure('Derivative')
    plt.clf()
    plt.plot(x, dydx, '*', label='dydx raw')
    plt.plot(x_plot, dydx_plot, '-', label='dydx fit')
    plt.plot(x_plot, f_gauss(x_plot, *p_guess), '-', label='Guess')
    plt.plot(x_plot, f_gauss(x_plot, *popt2), '-', label='Fit')
    plt.legend()

    # Report findings
    C = 2 * np.sqrt(2 * np.log(2))
    print('\nThe beam size is %f um' % (1000 * C * popt2[1]))
    print('The edge is at %.4f mm\n' % (popt2[2]))


def mv_position(pos = []):
    """
    Move to predefined positions of phosphor paper(pos1,default),
    the schitillator(pos2), or the Cu wire(pos3).

    pos     <list> 1 = [22.7, 25.66, 52.596] # phosphor paper
                   2 = [29.18, 19.33, 51.82] # scintillator
                   3 = [27.166, 25.217, 45.859] # Cu wire
                   [x, y, z] # any positions defined
    """
    print('To go to position 1,2,3; Assuming topx, topz at 0; rotation at 45 deg.')

    # Check current positions
    if (pos == []):
        print('You are now in this position: %f, %f, %f. No new positions given.Exiting...' % (hf_stage.x.position, hf_stage.y.position, hf_stage.z.position))
        return

    # Check positions and go there
    if (pos == 1):
        print('Will go to phosphor paper position.')
        pos = [22.7, 25.66, 52.596]
        yield from mv(hf_stage.x, pos[0], hf_stage.y, pos[1], hf_stage.z, pos[2])
    elif (pos == 2):
        print('Will go to scintillator position.')
        pos = [29.18, 19.33, 51.82]
        yield from mv(hf_stage.x, pos[0], hf_stage.y, pos[1], hf_stage.z, pos[2])
    elif (pos == 3):
        print('Will go to Cu horizontal wire position.')
        pos = [27.166, 25.217, 45.859]
        yield from mv(hf_stage.x, pos[0], hf_stage.y, pos[1], hf_stage.z, pos[2])
    elif (len(pos) > 2):
        print('You will move to the defined positions now.')
        yield from mv(hf_stage.x, pos[0], hf_stage.y, pos[1], hf_stage.z, pos[2])
    else:
        print('Not a position, exiting...')
        return


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

