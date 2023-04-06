print(f'Loading {__file__}...')


import os
import datetime
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
            bpmAD.cam.acquire_time.put(current_exptime + 0.0005)
            yield from bps.sleep(0.5)
    elif (maxct > 170):
        while(bpmAD.stats1.max_value.get() >= 150):
            current_exptime = bpmAD.cam.acquire_time.value
            bpmAD.cam.acquire_time.put(current_exptime - 0.0005)
            yield from bps.sleep(0.5)


def undulator_calibration(
    outfile=None,
    UCalibDir='/home/xf05id1/current_user_data/',
    u_gap_start=6500,
    u_gap_end=12000,
    u_gap_step=500,
    harmonic=3,
):
    '''
    outfile  string   filename for a txt file for the lookup table
                      desirable to name it with the date of the calibration
                      e.g. SRXUgapCalibration20161225.txt
    undulator gap set point range are defined in:
        u_gap_start  float
        u_gap_end    float
        u_gap_step   float
    '''

    # Format a default filename
    if (outfile is None):
        outfile = f"{datetime.datetime.now().strftime('%Y%m%d')}_SRXUgapCalibration.txt"

    # Check if the file exists
    if (not os.path.exists(UCalibDir + outfile)):
        with open(UCalibDir + outfile, 'w') as f:
            f.write('Undulator_gap\tFundemental_energy\n')

    # Bragg scan setup default
    energy_res = 0.002     # keV
    bragg_scanwidth = 0.25  # keV +/- this value
    bragg_scanpoint = (np.floor((2 * bragg_scanwidth) / (energy_res)) + 1).astype('int')

    energy.harmonic.put(harmonic)

    # Generate lookup table by scanning Bragg at each undulator gap set point
    for u_gap_setpoint in np.arange(u_gap_start,
                                    u_gap_end + u_gap_step,
                                    u_gap_step):
        # Look up the energy from the previous lookup table
        # Right now, the lookup table is in mm, not um!
        # A new lookup table should be created with the correct units
        energy_setpoint = (float(energy.utoelookup(u_gap_setpoint / 1000))
                           * harmonic)
        print('Move u_gap to:\t', u_gap_setpoint)
        print('Move Bragg energy to:\t', energy_setpoint)

        # energy.move_c2_x.put(False)
        energy.move_u_gap.put(True)
        yield from mv(energy, energy_setpoint)
        energy.move_u_gap.put(False)

        # Setup LiveCallbacks
        # liveplotfig1 = plt.figure()
        liveplotx = energy.energy.name
        liveploty = bpm3.total_current.name
        livetableitem = [energy.energy.name, ring_current.name, bpm3.total_current.name]
        ps = PeakStats(energy.energy.name, bpm3.total_current.name)
        livecallbacks = [LiveTable(livetableitem),
                         LivePlot(liveploty, x=liveplotx),
                         ps]

        # Setup the scan
        @subs_decorator(livecallbacks)
        def braggscan():
            yield from scan([bpm3, bpm4, ring_current],
                            energy,
                            energy_setpoint - bragg_scanwidth,
                            energy_setpoint + bragg_scanwidth,
                            bragg_scanpoint)

        # Run the scan
        yield from braggscan()

        # Find the maximum and output result to file
        maxenergy = ps.max[0]
        maxintensity = ps.max[1]
        fwhm = ps.fwhm
        print('Max energy is:\t', maxenergy)
        print('Fundemental energy:\t', maxenergy / harmonic)

        with open(UCalibDir + outfile, 'a') as f:
            f.write(f"{energy.u_gap.position / 1000:.6f}\t{(maxenergy / harmonic):.8f}\n")
    
    # Return moving u_gap
    energy.move_u_gap.put(True)

