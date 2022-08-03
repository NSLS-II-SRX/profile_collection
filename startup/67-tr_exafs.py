print(f'Loading {__file__}...')

import numpy as np
import time as ttime
import matplotlib.pyplot as plt
from itertools import product
import logging
import os

#a lot of the these are currently useless...
import bluesky.plans as bp
from bluesky.plan_stubs import (mov, movr)

# Notes from Andy
'''User directory is 
/home/xf05id1/current_user_data
    with open(userlogfile, 'a') as userlogf:
        userlogf.write(str(scan_id) + '\t' + uid + '\t' + scantype + '\n')'''

# Setting up a logging file
def start_logging():
    logfile = ttime.strftime("%y-%m-%d_T%H%M%S",ttime.localtime(ttime.time())) + '_logfile.txt'
    logdir = '/home/xf05id1/current_user_data/log_files/'
    os.makedirs(logdir, exist_ok=True)

    logging.basicConfig(filename=logdir+logfile, 
                        level=logging.DEBUG,
                        format='%(asctime)s| %(name)-4s: %(levelname)-12s %(message)s',
                        datefmt='%m-%d %H:%M')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    main_logger = logging.getLogger('main')
    
    note('Log file start.')

    return main_logger

# Log function to print to console and log file. Replaces print function.
def log(message):
    print(message)
    main_logger.info(message)

# Log function to print only to log file
def note(message):
    print(message)
    main_logger.debug(message)


# Changing VLM from production to laser VLM
nano_flying_zebra_laser = SRXFlyer1Axis(
    list(xs for xs in [xs] if xs is not None), sclr1, nanoZebra, name="nano_flying_zebra_laser"
)
set_flyer_zebra_stage_sigs(nano_flying_zebra_laser, 'time')

# something with nano_flying_zebra_laser._mode(SRXmode.fly)


# Laser controller as defined in pyEPICS
#can I set this up as a single ophyd object??
# laser_signal = epics.PV('XF:05IDD-ES:1{Dev:Zebra1}:SOFT_IN:B0') #zebra 1 or 2??
# laser_signal = EpicsSignal('XF:05IDD-ES:1{Dev:Zebra1}:SOFT_IN:B0', name='laser_signal')
# laser_power = epics.PV('XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_NGATE')
# laser_hold = epics.PV('XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_STEP')
# laser_ramp = epics.PV('XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_WID')
# laser_signal.put(0), laser_power.put(0), laser_hold.put(30), laser_ramp.put(5)

class SRXLaser(Device):
    signal = Cpt(EpicsSignal, 'XF:05IDD-ES:1{Dev:Zebra1}:SOFT_IN:B0')
    power = Cpt(EpicsSignal, 'XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_NGATE')
    hold = Cpt(EpicsSignal, 'XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_STEP')
    ramp = Cpt(EpicsSignal, 'XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_WID')

laser = SRXLaser('', name='laser')

laser.signal.set(0)
laser.power.set(0)
laser.hold.set(0)
laser.ramp.set(0)


# Defining a new scaler (sclr2) with photodiode channel included
# All of the following code will use sclr2 even when ip is not needed
# Will haveing two objects with the possibility to read from the same detectors going to cause problems??
'''sclr2 = SRXScaler("XF:05IDD-ES:1{Sclr:1}", name="sclr2")
sclr2.read_attrs = ["channels.chan2", "channels.chan3", "channels.chan4", "channels.chan5"]
i0_channel = getattr(sclr2.channels, "chan2")
i0_channel.name = "sclr_i0"
it_channel = getattr(sclr2.channels, "chan4")
it_channel.name = "sclr_it"
im_channel = getattr(sclr2.channels, "chan3")
im_channel.name = "sclr_im"
# How to confifure as voltage measurement??
vp_channel = getattr(sclr2.channels, "chan5")
vp_channel.name = "sclr_vp"
i0 = sclr2.channels.chan2
it = sclr2.channels.chan4
im = sclr2.channels.chan3
vp = sclr2.channels.chan5'''


#######################################
###     Main Utitiity Functions     ###
#######################################


def gen_xye_pos(erange = [11817, 11862, 11917, 12267], estep = [2, 0.5, 5], filedir='', filename='', start=[], end=[], spacing=10, replicates=1):

    '''
    erange      (array) energy range for XANES/EXAFS in eV. e.g., [11867-50, 11867-20, 11867+50, 11867+400]
    estep       (array) energy step size for each energy range in eV. e.g., = [2, 0.5, 5]
    filedir     (str)   file directory for where to save xye position data
    filename    (str)   file name for where to save xye position data
    start       (array) [x, y] coordinates for starting corner of rectangular ROI. nano_stage_x
    end         (array) [x, y] coordinates for ending corner of rectangular ROI. nano_stage_y
    spacing     (float) Spacing in microns between individual event coordinates. Large enough to avoid overlap
    replicates  (int)   Number of replicates at each energy of interest
    '''

    # Make sure user provided correct input
    if (erange is []):
        raise AttributeError("An energy range must be provided in a list by means of the 'erange' keyword.")
    if (estep is []):
        raise AttributeError("A list of energy steps must be provided by means of the 'esteps' keyword.")
    if (not isinstance(erange,list)) or (not isinstance(estep,list)):
        raise TypeError("The keywords 'estep' and 'erange' must be lists.")
    if (len(erange)-len(estep) != 1):
        raise ValueError("The 'erange' and 'estep' lists are inconsistent; " \
                         + 'c.f., erange = [7000, 7100, 7150, 7500], estep = [2, 0.5, 5] ')
    if (filedir == '') or (filename == ''):
        log("File name and directory blank. Generating defaults.")
        # Setting up a logging file
        filename = ttime.strftime("%y-%m-%d_T%H%M%S",ttime.localtime(ttime.time())) + '_coordinates.txt'
        filedir = '/home/xf05id1/current_user_data/xye_data/'
        os.makedirs(filedir, exist_ok=True)
    if (start is []) or (end is []):
        raise AttributeError("Please make sure start and end positions are defined!")

    # Convert erange and estep to numpy array
    ept = np.array([])
    erange = np.array(erange)
    estep = np.array(estep)
    # Calculation for the energy points
    for i in range(len(estep)):
        ept = np.append(ept, np.arange(erange[i], erange[i+1], estep[i]))
    ept = np.append(ept, np.array(erange[-1]))
    # Add replicate energy measurements
    ept = np.repeat(ept, replicates)

    # Map out points for each event
    x_ind = np.arange(start[0], end[0]+spacing, spacing)
    y_ind = np.arange(start[0], end[0]+spacing, spacing)
    xy_pos = np.array(list(product(x_ind, y_ind))) #numpy meshgrid may also work??

    # Are there enough points for the energies of interest?
    if len(ept) > len(xy_pos):
        print(f"{len(ept)=}\n{len(xy_pos)=}")
        raise ValueError("Not enough points for the number of events desired. Refine spacing or select a larger area.")

    # Combine first points with energies of interest
    xye_pos = []
    for i in range(len(ept)):
        xye_pos.append([xy_pos[i][0], xy_pos[i][1], ept[i]])
    xye_pos = np.asarray(xye_pos)

    # Save the positions and energies with 
    np.savetxt(filedir+filename, xye_pos, delimiter=',', fmt='%1.4f')
    note('xye_pos saved to ' + filedir+filename)


def read_xye_pos(filedir, filename):

    '''
    filedir     (str)   file directory for where to load xye position data
    filename    (str)   file name for where to load xye position data
    '''

    # Checking for filedir and filename. Loading most recent if not
    if (filedir == '') or (filename == ''):
        log("No file location information given. Loading most recent.")
        # Setting up a logging file
        filedir = '/home/xf05id1/current_user_data/xye_data/'
        filename =  os.listdir(calibdir)[-1]

    # Reading data into an array
    xye_pos = np.genfromtxt(filedir+filename, delimiter=',')
    return xye_pos


def laser_on(power, hold, ramp=5, delay=0):

    '''
    All variables sans delay are passed to laser_controller.py via laser object.
    power       (float) Target laser power in mW
    hold        (float) Hold time at target laser power in sec. Not currently used
    ramp        (float) Ramp time to target laser power in sec
    delay       (float) Delay befor triggering laser in sec
    '''

    # Make sure user provided correct input
    if any((power < 0), (hold < 0), (ramp < 0), (delay < 0)):
        raise ValueError("Values must be positive floats.")

    # Log some info
    note('Laser startup!')
    note(f'{power} mW power, {hold} sec hold, {ramp} sec ramp.')
    
    # Set up variables. Settle_time to not overwhelm zebra
    yield from abs_set(laser.power, power, settle_time=0.010)
    yield from abs_set(laser.hold, hold, settle_time=0.010)
    yield from abs_set(laser.ramp, ramp, settle_time=0.010)

    # Trigger laser after delay
    if delay > 0:
        yield from bps.sleep(delay)
    yield from abs_set(laser.signal, 1)
    note('Laser on!')


def laser_off():
    note('Laser off!')
    # Turn laser off
    yield from abs_set(laser.signal, 0)


def beam_knife_edge_scan(beam, direction, edge, distance, stepsize, 
                         acqtime=1.0, shutter=True):

    '''
    beam        (str)   'x-ray', 'laser', or 'both' Specifies appropriate detectors
    direction   (str)   Scan direction. x for vertical edge, y for horizontal.
    edge        (list)  [x,y,z] location of edge
    distance    (float) Distance in µm to either side of feature to scan across
    stepsize    (float) Step size in µm of scans
    acqtime     (float) Acquisition time of detectors
    shutter     (bool)  Use X-rays or not
    '''
    
    # Checking direction inputs
    variables = ['x', 'y']
    if not any(direction in variables for direction in variables):
        raise ValueError("Incorrect direction assignment. Please specify 'x' or 'y' for direction.")

    # Defining up the motors
    motors = [nano_stage.x, nano_stage.y, nano_stage.z]
    scan_motor = motors[variables.index(direction)]

    # Which beams are used
    vlaser_on = (beam == 'laser') or (beam == 'both')
    xray_on = ((beam == 'x-ray') or (beam == 'both')) & shutter

    # Setting up the detectors 
    det = [sclr1]
    # sclr1.stage_sigs['preset_time'] = acqtime 
    if beam in ['x-ray', 'both']:
        det.append(xs)
        # xs.stage_sigs['preset_time'] = acqtime
        # xs.cam.stage_sigs['acqtime'] = acqtime #whcih is appropriate??
    elif beam != 'laser':
        raise ValueError("Incorrect beam assignment. Please specify 'x-ray', 'laser', or 'both'.")

    # Convert stepsize to number of points
    num = np.round(((2 * distance) / stepsize) + 1)

    # Move sample stage to center position of features
    yield from mov(motors[0], edge[0],
                   motors[1], edge[1],
                   motors[2], edge[2])
    note(f'Moving motors to {edge} coordinates.')

    # Perform scan with stage to be adjusted
    #yield from rel_scan(det, scan_motor, -distance, distance, num) #depricated
    plotme = LivePlot('')
    @subs_decorator(plotme)
    def _plan(distance_1):
        log(f'Running {beam} beam(s) knife edge scan along {direction} at {edge}.')
        
        # Turn laser on if used        
        if vlaser_on:
            yield from laser_on(4, 200, ramp=5, delay=0) #arbitrary 4 mW power
            yield from bps.sleep(5) # let laser reach full power

        # Specifying scan type and *args
        if direction == 'x':
            xstart, xstop, xnum = edge[0] - distance_1, edge[0] + distance_1, num
            ystart, ystop, ynum = edge[1], edge[1], 1 #should ynum be zero???
            yield from coarse_scan_and_fly(xstart, xstop, xnum,
                                           ystart, ystop, ynum, acqtime,
                                           flying_zebra=nano_flying_zebra_coarse,
                                           shutter=xray_on, plot=False)
        elif direction == 'y':
            xstart, xstop, xnum = edge[0], edge[0], 1 #should xnum be zero???
            ystart, ystop, ynum = edge[1] - distance_1, edge[1] + distance_1, num
            yield from coarse_y_scan_and_fly(ystart, ystop, ynum,
                                           xstart, xstop, xnum, acqtime,
                                           flying_zebra=nano_flying_zebra_coarse,
                                           shutter=xray_on, plot=False)
        # Turn laser off if used
        if vlaser_on:
            yield from laser_off()


    # Perform the actual scan
    yield from _plan(distance_1=distance)


    # Plot and process the data
    beam_param = []
    ext_scan = False 
    for beam_1 in ['laser', 'x-ray']:

        # If laser was not used, skips to checking x-ray signal
        if beam not in ['laser', 'both']: #seeing if the original beam used the laser
            continue #if not skips to only fitting x-ray data

        # When checking x-rays, see if any x-rays were avialable
        if (beam_1 == 'x-ray') & (not shutter):
            log('No x-rays to properly perform knife edge scan.')
            cent_position = edge[variables.index(direction)]
            fwhm = -1 # this way we obviously know something is wrong
            continue
        
        # Try to find edge
        try:
            cent_position, fwhm = beam_knife_edge_plot(beam=beam_1, scan_motor=scan_motor, plotme=plotme)

        # RuntimeErrors for nonideal fitting. Trying to fix by extended scan range    
        except RuntimeError: 
            if not ext_scan:
                try:
                    log('Doubling scan range.')
                    yield from _plan(distance_1 = 2 * distance)
                    ext_scan = True
                    cent_position, fwhm = beam_knife_edge_plot(beam=beam_1, scan_motor=scan_motor, plotme=plotme)
            
                except RuntimeError:
                    log('Knife edge scan failed to find position.')
                    raise RuntimeError()
           
            else:
                log('Knife edge scan failed to find position.')
                raise RuntimeError()
        
        beam_param.append(cent_position, fwhm)

    return beam_param # Cent_position then fwhm. Laser parameters first if 'both'


def beam_knife_edge_plot(beam, scan_motor, scanid=-1, plot_guess=True, 
                         bin_low=934, bin_high=954, plotme=None):

    '''
    beam        (str)   'x-ray' or 'laser' Specifies appropriate detectors, motors, and curve shape
    scan_motor  (motor) 
    scanid      (int)   ID of previous scan. Default is -1
    plot_guess  (bool)  If true, plot guess function. Helps if curve fitting is poor
    bin_low     (int)   Start bin of table data #number for foil of interest
    bin_high    (int)   End bin of table data
    plotme      (ax)    Pyplot axis. Creates one if None.
    '''

    # Get the scanid
    h = db[int(scanid)]
    start_doc = h.start
    id_str = start_doc['scan_id']
    note(f'Trying to determine beam parameters from {scanid} scan.')

    motors = [nano_stage.x, nano_stage.y, nano_stage.z]
    pos = start_doc['scan']['fast_axis']['motor_name']
    variables = ['x', 'y']

    # Get the information from the previous scan
    haz_data = False
    loop_counter = 0
    MAX_LOOP_COUNTER = 30
    print('Waiting for data...', end='', flush=True)
    while (loop_counter < MAX_LOOP_COUNTER):
        try:
            tbl = db[-1].table('stream0', fill=True)
            haz_data = True
            log('done')
            break
        except:
            loop_counter += 1
            ttime.sleep(1) #are we not supposed to use sleep inside the run engine??

    # Check if we haz data
    if (not haz_data):
        log('Data collection timed out!')
        raise OSError()

    # Get the data from either x3 or the photodiode
    if (beam == 'x-ray'):
        y = np.sum(np.array(tbl['fluor'])[0][:, :, bin_low:bin_high], axis=(1, 2)) #these number change depending on material
    elif (beam == 'laser'):
        y = tbl['it'].values[0] #redefined for when channel 4 is using the photodiode

    # Get position data
    # x = np.array(tbl[pos])[0]
    # if (x.size ==1):
    #    x = np.array(tbl[pos])
    ## Need to interpolate x values
    xstart = start_doc['scan']['scan_input'][0]
    xstop = start_doc['scan']['scan_input'][1]
    xnum = start_doc['scan']['scan_input'][2]
    x = np.linspace(xstart, xstop, xnum)
    x, y = x.astype(np.float64), y.astype(np.float64)
    note(f'Data acquired for {beam}! Now fitting...')

    # Guessing the function and fitting the raw data
    p_guess = [np.amax(y),
                0.5,
                x[np.argmax(np.abs(np.gradient(y,x)))],
                np.amin(y) + 0.5*np.amax(y),
                0.1]
    if np.mean(y[:3]) > np.mean(y[-3:]):
        p_guess[0] = -np.amax(y)

    # Fitting and useful information
    try:
        popt, pcov = curve_fit(f_int_gauss, x, y, p0=p_guess)
    except:
        log('Raw fit failed.')
        popt, pcov = p_guess, 2*p_guess #guaranteed to raise an error later
    cent_position = popt[2]
    fwhm = 2 * np.sqrt(2 * np.log(2))*popt[1]
    perr = np.sqrt(np.diag(pcov))
    frac_err = np.abs(perr/popt)

    # Report useful data
    log(f'The beam center is at {cent_position:.4f} µm along ' + variables[i] + '.')
    log(f'The beam fwhm is {fwhm:.4f} µm along ' + variables[i] + '.')

    # Set plotting variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)

    # Display the fit of the raw data
    if (plotme is None):
        fig, ax = plt.subplots(1, 1)
    else:
        ax = plotme.ax

    #is it worth just saving these to a designated folder?
    # Display fit of raw data
    ax.cla()
    ax.plot(x, y, '+', label='Raw Data', c='k')
    if plot_guess:
        ax.plot(x_plot, f_int_gauss(x_plot, *p_guess), '--', label='Guess Fit', c='0.5')
    ax.plot(x_plot, f_int_gauss(x_plot, *popt), '-', label='Erf Fit', c='r')
    ax.set_title(f'Scan {id_str} of ' + beam)
    ax.set_xlabel(pos)
    ax.set_ylabel('ROI Counts')
    ax.legend()
    plt.savefig(f'/home/xf05id1/current_user_data/{id_str}_erf_{beam}_{direction}.png')

    # Display the fit derivative
    ax.cla()
    ax.plot(x, np.gradient(y, x), '+', label='Derivative Data', c='k')
    if plot_guess:
        ax.plot(x_plot, np.gradient(f_int_gauss(x_plot, *p_guess), x_plot), '--', label='Guess Fit', c='0.5')
    ax.plot(x_plot, np.gradient(f_int_gauss(x_plot, *popt), x_plot), '-', label='Erf Fit', c='r')
    ax.set_title(f'Scan {id_str} of ' + beam)
    ax.set_xlabel(pos)
    ax.set_ylabel('Derivative ROI Counts')
    ax.legend()
    plt.savefig(f'/home/xf05id1/current_user_data/{id_str}_gauss_{beam}_{direction}.png')
    plt.close()
    
    # Check the quality of the fit and to see if the edge is mostly within range
    # For the failure, rerun the scan outside of this function
    # After plotting, so there is way to guage fit quality visually
    if np.abs(cent_position) > 0.8*np.abs(distance):
        log('Edge position barely within FOV.')
        raise RuntimeError()
    if any(frac_err > 1):
        log('Poor fitting. Coefficient error exceeds predicted values.')
        raise RuntimeError()

    return cent_position, fwhm


def auto_beam_alignment(v_edge, h_edge, distance, stepsize, acqtime=1.0,
                        shutter=True, check=False):

    '''
    v_edge          (list)  [x,y,z] location of vertical line/edge. Scan across for x position
    h_edge          (list)  [x,y,z] location of horizontal line/edge. Scan across fory position
    distance        (float) Distance in µm to either side of feature to scan across
    stepsize        (float) Step size in µm of scans
    acqtime         (float) Acquisition time of detectors
    shutter         (bool)  Use X-rays or not
    check           (bool)  If True, double check the laser adjustment and correspondence between sample and vlm stages
    '''


    # Setting up label variables
    motors = [nano_stage.x, nano_stage.y, nano_stage.z]
    variables = ['x','y']
    FOV = [500, 500] # FOV of VLM in um. What is this? Laser spot should start within VLM image.
    vlm_motors = [nano_vlm_stage.x, nano_vlm_stage.y, nano_vlm_stage.z]
    xray_pos, xray_sizes = [], []
    laser_pos, laser_sizes = [], []
    off_adj = []

    # Alignment
    note('Running auto beam alignment')
    for i, j in enumerate([v_edge, h_edge]):

        # Determine beam positions along variable
        beam_param = yield from beam_knife_edge_scan('both', variables[i], j, distance=distance, stepsize=stepsize, 
                                                                acqtime=acqtime, shutter=shutter )
        
        # Adjust VLM position
        offset = beam_param[2] - beam_param[0]
        if np.abs(offset) > 0.5*FOV[i]:
            raise RuntimeError("Trying to adjust stage by more than 50% of FOV. Retry beam alignment.")
        yield from movr(vlm_motors[i], (offset * 0.001)) # vlm motors in mm not um
        log(f'Offset VLM by {offset:.4f} µm along ' + variables[i] + '-axis.')

        # Confirm adjustment
        adjustment = 0
        if check:
            log('Checking VLM stage correspondence. Determining new laser position.')

            # Re-determine laser position along variable
            new_laser_pos, new_laser_size = yield from beam_knife_edge_scan('laser', variables[i], j, distance=distance, stepsize=stepsize, 
                                                                  acqtime=acqtime, shutter=shutter )
            
            # Adjust VLM position
            new_offset = beam_param[2] - new_laser_pos
            adjustment = offset-new_offset
            if np.abs(new_offset) > np.abs(offset):
                raise RuntimeError("Stage correspondence issue. Beam alignment will not converge.")
            yield from movr(vlm_motors[i], new_offset)
            log(f'Offset VLM by {new_offset:.4f} µm along ' + variables[i] + '-axis.')
            log(f'Sample and VLM stage correlation off by {adjustment:.4f} µm along ' + variables[i] + '-axis.')

        # Record information
        laser_pos.append(new_laser_pos), laser_sizes.append(new_laser_size)
        xray_pos.append(beam_param[2]), xray_sizes.append(beam_param[3])
        off_adj.append(offset, adjustment)

        return xray_pos, xray_sizes, laser_pos, laser_sizes, off_adj


def laser_time_series(power, hold, ramp=5, dets=[xs, merlin, nano_vlm], 
                      acqtime=0.001, shutter=True):
    
    '''
    power
    hold
    ramp
    dets        (list) detectors used to collect time-resolved data
    total_time  (float) Total acquisition time. Defualt is 60 seconds
    acqtime     (float) Acquisition/integration time. Defualt is 0.001 seconds
    shutter     (bool)  Use X-rays or not
    '''

    # Record relevant meta data in the Start document, define in 90-usersetup.py
    # Add user meta data
    note('Setting up time series collection...')
    note(f'{dets_by_name} recording for {total_time} sec at {acqtime} intervals.')
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'XAS_TIME' #Should this be something different?
    scan_md['scan']['acquisition'] = total_time #Can I make up new entries like this?
    scan_md['scan']['dwell'] = acqtime

    # Register the detectors
    dets = [ring_current, sclr1] + dets #what is xbpm2???
    dets_by_name = {d.name : d for d in dets}

    # Setup scaler
    if (acqtime < 0.001):
        acqtime = 0.001 #limits the time resolution. Why???
    sclr1.stage_sigs['external_trig'] = True #how to define this value

    # Total number of time steps
    total_time = hold + ramp
    N_tot = total_time/acqtime #how does this incorporate dead time??
    #define the N_tot as a function of TTL pulses
    #how to trigger off of theses pulses and for how long?

    # Setup xspress3
    if ('xs' in dets_by_name):
        xs.stage_sigs['external_trig'] = True # how to define this
        xs.cam.stage_sigs['acquire_time'] = acqtime
        xs.cam.stage_sigs['acquire_period'] = acqtime + 0.005 #too match the area detectors
        xs.stage_sigs['total_points'] = N_tot
 
    # Setup Merlin area detector
    if ('merlin' in dets_by_name):
        merlin.cam.stage_sigs['tigger_mode'] = 0
        merlin.cam.stage_sigs['acquire_time'] = acqtime
        merlin.cam.stage_sigs['acquire_period'] = acqtime + 0.005 #can I implement binning via the stage_sigs??
        merlin.cam.stage_sigs['num_images'] = N_tot #this is not supposed to be one
        merlin.hdf5.stage_sigs['num_capture'] = N_tot
        merlin._mod = SRXMode.step #what does this do???
        merlin.stage_sigs['total_points'] = N_tot

    # Setup VLM camera
    if('nano_vlm' in dets_by_name):
        # I cannot find anything about the acquisistion rate of this camera. Was that not put in the code since it is not really used??
        # I am not sure if the nano_vlm is actually setup to record data or not.
        # Do I need to implement binning on this one as well?? Also built on CamBase like Merlin
        nano_vlm.cam.stage_sigs['tigger_mode'] = 0
        nano_vlm.cam.stage_sigs['acquire_time'] = acqtime
        nano_vlm.cam.stage_sigs['acquire_period'] = acqtime + 0.005
        nano_vlm.cam.stage_sigs['num_images'] = N_tot
        #does nano_vlm have a hdf5 thing??
        nano_vlm.hdf5.stage_sigs['num_capture'] = N_tot
        nano_vlm._mod = SRXMode.step #what does this do???
        nano_vlm.stage_sigs['total_points'] = N_tot

    # Setup Dexela area detector
    #if('dexela' in dets_by_name):
        #add the important things for the dexela. Including dark frame??

    # Check shutter
    yield from check_shutter(shutter, 'Open')

    # Turn on laser and start counting!
    yield from laser_on(power, hold, ramp, delay=0) #if laser is in an opyd object, can it also be triggered at same time as everythin else??
    yield from count(dets, num=N_tot, md=scan_md) #not actually counting
    yield from laser_off()

    # Close shutter
    yield from check_shutter(shutter, 'Close')

    # Plotting data
    #will be useful, but maybe after the acqusition and not live. Does this make it easier??
    #plot each series after they have been acquired??
    note('Time series acquired!') #How to add scan ID information??


def tr_xanes_plan(xye_pos, power, hold, v_edge, h_edge, distance, stepsize, N_start=0, z_pos=[], ramp=5,
                  dets=[xs, merlin, nano_vlm], acqtime=0.001,
                  waittime=5, peakup_N=15, align_N=15, shutter=True):

    '''
    xye_pos     (list)  x and y positions and energies to to acquire time series
    power       (float) Target laser power. Controlled by calibration curve
    hold        (float) Hold time at target laser power. If -1, then holds indefinitely
    v_edge      (list)  [x,y] location of vertical line/edge. Scan across for X-ray x position
    h_edge      (list)  [x,y] location of horizontal line/edge. Scan across for X-ray y position
    distance    (float) Distance in µm to either side of feature to scan across
    stepsize    (float) Step size in µm of scans
    n_start     (int)   Start index of batch. Used to pick up failed batches.
    z_pos       (float) z position for focused laser (i.e., sample plane)
    ramp        (float) Time for ramp up to target laser power
    dets        (list)  detectors used to collect time-resolved data
    acqtime     (float) Acquisition/integration time. Defualt is 0.001 seconds
    waittime    (float) Wait time between collecting times series
    peakup_N    (int)   Run a peakup every peakup_N time series. Consider the number of replicate energies
    align_N     (int)   Run auto beam alignment align_N time series
    shutter     (bool)  Use X-rays or not
    '''

    # Check positions
    if (xye_pos == []):
        raise AttributeError("You need to enter spatial and energy positions.")

    #Number of total events
    N = len(xye_pos)

    # Check N_start
    if any((N_start < 0), (N_start <= N), (not isinstance(erange,int))):
        raise ValueError("N_start must be a positive integer within the number of events.")
    
    # Checking for improper hold time input
    if hold < 0:
        raise ValueError("Hold times cannot be negative nor indefinite for batched time series collection.")

    # Define total_time from laser parameters
    total_time = ramp + hold

    # Define quality variables
    xray_pos_lst, laser_pos_lst = [], []
    xray_size_lst, laser_size_lst = [], []
    offsets_lst, time_lst = [], []

    # Log batch information...
    if N_start == 0:
        log('Starting TR_XANES batch...')
    else:
        log('Re-starting TR_XANES batch...')
    note('Target paramters are:')
    note(f'{N} events. {total_time} sec acquire periods. {acqtime} sec acquisition rate.')
    note(f'{power} mW laser power. {ramp} sec ramp with {hold} sec hold.')
    note(f'Alignment every {align_N} events. Peakup every {peakup_N} events.')
    note(f'Edges at: vertical {v_edge}, horizontal {h_edge}')

    # Move z-stage to sample plane if given
    if (z_pos != []):
        log(f'Moving to:')
        log(f'\tz = {z_pos}')
        yield from mov(nano_stage.z, z_pos)
    else:
        log('No z-coordinate given. Assuming position already at sample plane.') #how to record current z_pos
        note(f'Current z_pos is {z_pos}.')

    # Timining statistics
    N_time = N-N_start
    num_peakup = int((N_time+peakup_N-1)/peakup_N)
    peakup_count = 0
    num_align = int((N_time+align_N-1)/align_N)
    align_count = 0
    t_elap_p, t_elap_a, t_elap_e = 0, 0, 0

    # Loop through xye_pos positions and energies
    for i in range(N):
        # Skipping any previouly performed events
        if i < N_start:
            continue

        # Periodically perform peakup to maximize signal
        if (i % peakup_N == 0) or (i == N_start):
            t0_p = ttime.time()
            log('Performing peakup...')
            yield from peakup_fine(shutter=shutter) #shutter=shutter necessary?
            t_elap_p += ttime.time()-t0_p

        # Periodically perform auto_align to confirm laser and x-ray coincidence
        if (i % align_N == 0) or (i == N_start):
            t0_a = ttime.time()
            log('Performing auto beam alignment...')
            xray_pos, laser_pos, xray_size, laser_size, offsets = yield from auto_beam_alignment(v_edge, h_edge, distance, 
                                                                                                 stepsize, acqtime=1.0, shutter=shutter)
            
            # Append quality variables/write to scan log file
            xray_pos_lst.append(xray_pos), laser_pos_lst.append(laser_pos)
            xray_size_lst.append(xray_size), laser_size_lst.append(laser_size)
            offsets_lst.append(offsets), time_lst.append(ttime.ctime())
            # All this information is currently in the log file, but maybe save somewhere else??

            # Update the v_edge and h_edge positions
            # Will these positions drift along direction of edge??
            v_edge = [xray_pos[0], v_edge[1]] #update x-position
            h_edge = [h_edge[0], xray_pos[1]] #updated y-position
            note(f'New edge positions: vertical {v_edge} and horizontal {h_edge}')
            t_elap_a += ttime.time()-t0_a
        
        # Move to positions and energy
        t0_e = ttime.time()
        log(f'Scanning though event {i} of {N} events.')
        log('Moving to:')
        log(f'\tx = {xye_pos[i][0]}')
        log(f'\ty = {xye_pos[i][1]}')
        log(f'\te = {xye_pos[i][2]}')
        yield from mov(nano_stage.x, xye_pos[i][0], nano_stage.y, xye_pos[i][1], energy, xye_pos[i][2])

        # Trigger laser and collect time series data
        yield from laser_time_series(power, hold, ramp, xye_pos[i][2], dets=dets, total_time=total_time, acqtime=acqtime, shutter=shutter)

        # Time estimates
        t_elap_e += ttime.time()-t0_e
        t_rem_p = (num_peakup-peakup_count)*(t_elap_p/num_peakup)
        t_rem_a = (num_align-align_count)*(t_elap_a/num_align)
        t_rem_e = (N_time-i+1)*(t_elap_e/N_time)
        t_rem_tot = np.sum(t_rem_p, t_rem_a, t_rem_e)
        if t_rem_tot < 86400:
            str_rem = ttime.strftime("%#H:%M:%S",ttime.gmtime(t_rem_tot))
        elif t_rem_tot >= 86400:
            str_rem = ttime.strftime("%-d day and %#H:%M:%S",ttime.gmtime(t_rem_tot))
        elif t_rem_tot > 86400 * 2: #this really is just to have pural days...
            str_rem = ttime.strftime("%-d days and %#H:%M:%S",ttime.gmtime(t_rem_tot))
        str_comp = ttime.strftime("%a %b %#d %#H:%M:%S",ttime.localtime(ttime.time() + t_rem_tot))

        log(f'Finished event {i} of {N}.')
        log(f'Estimated {str_rem} remaining.')
        log(f'Predicted completion at {str_comp}.')
        
        # Wait
        if (i != (N-1)):
            log(f'Waiting {waittime} seconds until next event starts.')
            yield from bps.sleep(waittime)

    # Log end of batch
    log('Batch is complete!!!')
