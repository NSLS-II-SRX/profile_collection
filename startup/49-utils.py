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
from bluesky.utils import short_uid
from bluesky.plan_stubs import checkpoint, abs_set, wait, trigger_and_read
import lmfit
import time

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

# Error function with offset
def f_offset_erf(x, A, sigma, x0, y0):
    x_star = (x - x0) / sigma
    return A * erf(x_star / np.sqrt(2)) + y0

# Let's fit two error functions
def f_two_erfs(x, A1, sigma1, x1, y1,
                  A2, sigma2, x2, y2):
    x1_star = (x - x1) / sigma1
    x2_star = (x - x2) / sigma2

    f_combo = f_offset_erf(x, A1, sigma1, x1, y1) + f_offset_erf(x, A2, sigma2, x2, y2)
    return f_combo

def mv_position(pos = []):
    """
    Move to predefined positions of phosphor paper(pos1,default),
    the schitillator(pos2), or the Cu wire(pos3).

    pos     <list> 1 = [22.7, 25.66, 52.596] # phosphor paper
                   2 = [32.11,17.646,61.374] # scintillator
                   3 = [[30.114,22.464,53.874] # ANT Siemens star
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
        pos = [32.11, 17.064, 61.454]
        yield from mv(hf_stage.x, pos[0], hf_stage.y, pos[1], hf_stage.z, pos[2])
    elif (pos == 3):
        print('Will go to Simens Star position.')
        pos = [30.114,22.464,53.874]
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


def custom_one_nd_step(detectors, step, pos_cache):
    """
    Inner loop of an N-dimensional step scan

    This is the default function for ``per_step`` param in ND plans.

    Parameters
    ----------
    detectors : iterable
        devices to read
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions
    """
    def move():
        yield from checkpoint()
        grp = short_uid('set')
        for motor, pos in step.items():
            if pos == pos_cache[motor]:
                # This step does not move this motor.
                continue
            yield from abs_set(motor, pos, group=grp)
            pos_cache[motor] = pos
        yield from wait(group=grp)

    motors = step.keys()
    yield from move()

    # Here is the custom part, add a 1 second delay
    yield from bps.sleep(1)

    yield from trigger_and_read(list(detectors) + list(motors))