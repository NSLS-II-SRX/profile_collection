def xanes_map(erange=[], estep=[],
              xstart=-5, xstop=5, xnum=21,
              ystart=-5, ystop=5, ynum=21, dwell=0.050,
              *, shutter=True, harmonic=1, align=False, align_at=None):

    # Convert erange and estep to numpy array
    ept = np.array([])
    erange = np.array(erange)
    estep = np.array(estep)
    # Calculation for the energy points
    for i in range(len(estep)):
        ept = np.append(ept, np.arange(erange[i], erange[i+1], estep[i]))
    ept = np.append(ept, np.array(erange[-1]))


    # Record relevant meta data in the Start document, defined in 90-usersetup.py
    # Add user meta data
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'XAS_MAP'
    scan_md['scan']['ROI'] = 1
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['energy_input'] = str(np.around(erange, 2)) + ', ' + str(np.around(estep, 2))
    # scan_md['scan']['energy'] = ept
    scan_md['scan']['map_input'] = [xstart, xstop, xnum, ystart, ystop, ynum, dwell]
    scan_md['scan']['detectors'] = [d.name for d in detectors]
    scan_md['scan']['fast_axis'] = {'motor_name' : xmotor.name,
                                    'units' : xmotor.motor_egu.get()}
    scan_md['scan']['slow_axis'] = {'motor_name' : ymotor.name,
                                    'units' : ymotor.motor_egu.get()}
    scan_md['scan']['theta'] = {'val' : nano_stage.th.user_readback.get(),
                                'units' : nano_stage.th.motor_egu.get()}
    scan_md['scan']['delta'] = {'val' : delta,
                                'units' : xmotor.motor_egu.get()}
    scan_md['scan']['snake'] = snake
    scan_md['scan']['shape'] = (xnum, ynum)


    # Setup DCM/energy options
    if (harmonic != 1):
        yield from abs_set(energy.harmonic, harmonic)


    # Prepare to peak up DCM at middle scan point
    if (align_at is not None):
        align = True
    if (align is True):
        if (align_at is None):
            align_at = 0.5*(ept[0] + ept[-1])
            print("Aligning at ", align_at)
            yield from abs_set(energy, align_at, wait=True)
        else:
            print("Aligning at ", align_at)
            yield from abs_set(energy, float(align_at), wait=True)
        yield from peakup(shutter=shutter)


    # Loop through energies and run 2D maps
    for e in ept:
        print(f"  Moving to energy, {e} eV...")
        yield from mov(energy, e)
        yield from bps.sleep(1)

        print(f"  Running map...")
        yield from nano_scan_and_fly(xstart, xstop, xnum,
                                     ystart, ystop, ynum, dwell,
                                     shutter=shutter)



