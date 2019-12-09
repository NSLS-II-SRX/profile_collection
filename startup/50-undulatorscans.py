print(f'Loading {__file__}...')


import numpy as np
import matplotlib.pyplot as plt
from bluesky.callbacks.fitting import PeakStats
from bluesky.callbacks.mpl_plotting import plot_peak_stats


def bpmAD_exposuretime_adjust():  
    '''
    Adjust the exposure time of the BPM Camera
    '''
    maxct = bpmAD.stats1.max_value.get()
    if (maxct < 150):
        while(bpmAD.stats1.max_value.get() <= 170):
            current_exptime = bpmAD.cam.acquire_time.value
            yield from abs_set(bpmAD.cam.acquire_time, current_exptime + 0.0005, wait=True)
            yield from bps.sleep(0.5)
    elif (maxct > 170):
        while(bpmAD.stats1.max_value.get() >= 150):
            current_exptime = bpmAD.cam.acquire_time.value
            yield from abs_set(bpmAD.cam.acquire_time, current_exptime - 0.0005, wait=True)
            yield from bps.sleep(0.5)


def undulator_calibration(outfile=None,
                          UCalibDir = '/nsls2/xf05id1/shared/config/undulator_calibration/',
                          u_gap_start=6500, u_gap_end=12000, u_gap_step = 500):
    '''
    outfile  string   filename for a txt file for the lookup table
                      desirable to name it with the date of the calibration
                      e.g. SRXUgapCalibration20161225.txt
    undulator gap set point range are defined in:
        u_gap_start  float
        u_gap_end    float
        u_gap_step   float
    '''  
    
    bpmAD.cam.read_attrs = ['acquire_time']
    bpmAD.configuration_attrs = ['cam']

    # Format a default filename
    if (outfile is None):
        outfile = '%s_SRXUgapCalibration.txt' % (datetime.datetime.now().strftime('%Y%m%d'))

    # Check if the file exists
    if (not os.path.exists(UCalibDir + outfile)):
        f = open(UCalibDir + outfile, 'w')
        f.write('Undulator_gap\tFundemental_energy\n')
        f.close()
    
    # Bragg scan setup default
    energy_res = 0.002     # keV
    bragg_scanwidth = 0.1  # keV
    bragg_scanpoint = int(bragg_scanwidth * 2 / energy_res + 1)
    harmonic = 3

    yield from abs_set(energy.harmonic, harmonic)
    
    # Generate lookup table by scanning Bragg at each undulator gap set point
    for u_gap_setpoint in np.arange(u_gap_start, u_gap_end+u_gap_step, u_gap_step):
        # Look up the energy from the previous lookup table
        # Right now, the lookup table is in mm, not um!
        # A new lookup table should be created with the correct units
        energy_setpoint = float(energy.utoelookup(u_gap_setpoint / 1000)) * harmonic
        print('Move u_gap to:\t', u_gap_setpoint)
        print('Move Bragg energy to:\t', energy_setpoint)
        
        yield from abs_set(energy.move_c2_x, False, wait=True)
        yield from abs_set(energy.move_u_gap, True, wait=True)
        yield from bps.sleep(0.2)    
        yield from mv(energy, energy_setpoint)

        yield from bpmAD_exposuretime_adjust()    
        yield from abs_set(energy.move_u_gap, False, wait=True)

        # Setup LiveCallbacks
        liveplotfig1 = plt.figure()
        liveploty = bpmAD.stats1.total.name
        livetableitem = [energy.energy, bpmAD.stats1.total, ring_current]
        liveplotx = energy.energy.name
        ps = PeakStats(energy.energy.name, bpmAD.stats1.total.name)
        livecallbacks = [LiveTable(livetableitem),
                         LivePlot(liveploty, x=liveplotx, fig=liveplotfig1), ps]

        # Setup the scan
        @subs_decorator(livecallbacks)
        def braggscan():
            yield from scan([bpmAD, pu, ring_current],
                            energy,
                            energy_setpoint-bragg_scanwidth,
                            energy_setpoint+bragg_scanwidth,
                            bragg_scanpoint)

        # Run the scan
        yield from braggscan()

        # Find the maximum and output result to file
        maxenergy = ps.max[0]
        maxintensity = ps.max[1]
        fwhm = ps.fwhm
        print('Max energy is:\t', maxenergy)
        print('Fundemental energy:\t', maxenergy / harmonic)
        
        f = open(UCalibDir + outfile, 'a')
        f.write(str(energy.u_gap.position) + '\t' + str(maxenergy / harmonic) + '\n')
        f.close()

