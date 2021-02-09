print(f'Loading {__file__}...')
import h5py
import time as ttime

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
        if motor.name == 'hf_stage_x':
            pos = 'enc2'
        elif motor.name == 'hf_stage_y':
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
        y = np.sum(np.array(tbl['fluor'])[0][:, :, 961:981], axis=(1, 2))
        #y = y / np.array(tbl['i0'])[0]
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
        # popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
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
    # plt.plot(x_plot, f_two_erfs(x_plot, *p_guess), '-', label='Guess fit')
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

# Run a knife-edge scan
def nano_knife_edge(motor, start, stop, stepsize, acqtime,
                    normalize=True, use_trans=False,
                    scan_only=False, shutter=True, plot=True, plot_guess=False):
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

    # Need to convert stepsize to number of points
    num = np.round((stop - start) / stepsize) + 1

    # Run the scan
    if (motor.name == 'nano_stage_sx'):
        fly = True
        pos = 'enc1'
        fluor_key = 'fluor'
        y0 = nano_stage.sy.user_readback.get()
        plotme = LivePlot('')
        @subs_decorator(plotme)
        def _plan():
            yield from nano_scan_and_fly(start, stop, num,
                                         y0, y0, 1, acqtime,
                                         shutter=shutter)
        yield from _plan()
    elif (motor.name == 'nano_stage_sy'):
        fly = True
        pos = 'enc2'
        fluor_key = 'fluor'
        x0 = nano_stage.sx.user_readback.get()
        plotme = LivePlot('')
        @subs_decorator(plotme)
        def _plan():
            yield from nano_y_scan_and_fly(start, stop, num,
                                           x0, x0, 1, acqtime,
                                           shutter=shutter)
        yield from _plan()
    elif (motor.name == 'nano_stage_x'):
        fly = False
        pos = motor.name
        fluor_key = 'xs2_channel1'
        y0 = nano_stage.y.user_readback.get()
        dets = [xs2, sclr1]
        yield from abs_set(xs2.total_points, num)
        livecallbacks = [LiveTable([motor.name,
                                    xs2.channel1.rois.roi01.value.name])]
        livecallbacks.append(LivePlot(xs2.channel1.rois.roi01.value.name,
                                      motor.name))
        if (shutter):
            yield from mov(shut_b, 'Open')
        yield from subs_wrapper(scan(dets, motor, start, stop, num),
                                {'all' : livecallbacks})
        if (shutter):
            yield from mov(shut_b, 'Close')
    elif (motor.name == 'nano_stage_y'):
        fly = False
        pos = motor.name
        fluor_key = 'xs2_channel1'
        x0 = nano_stage.x.user_readback.get()
        dets = [xs2, sclr1]
        yield from abs_set(xs2.total_points, num)
        livecallbacks = [LiveTable([motor.name,
                                    xs2.channel1.rois.roi01.value.name])]
        livecallbacks.append(LivePlot(xs2.channel1.rois.roi01.value.name,
                                      motor.name))
        if (shutter):
            yield from mov(shut_b, 'Open')
        yield from subs_wrapper(scan(dets, motor, start, stop, num),
                                {'all' : livecallbacks})
        if (shutter):
            yield from mov(shut_b, 'Close')
    else:
        print(f'{motor.name} is not implemented in this scan.')
        return

    # Do not do fitting, only do the scan
    if (scan_only):
        return

    # Get the scanid
    id_str = db[-1].start['scan_id']

    # Get the information from the previous scan
    haz_data = False
    loop_counter = 0
    MAX_LOOP_COUNTER = 30
    print('Waiting for data...', end='', flush=True)
    while (loop_counter < MAX_LOOP_COUNTER):
        try:
            if (fly):
                tbl = db[int(id_str)].table('stream0', fill=True)
            else:
                tbl = db[int(id_str)].table(fill=True)
            haz_data = True
            print('done')
            break
        except:
            loop_counter += 1
            yield from bps.sleep(1)

    # Check if we haz data
    if (not haz_data):
        print('Data collection timed out!')
        return
    
    # Get the data
    if (use_trans == True):
        y = tbl['it'].values[0] / tbl['im'].values[0]
    else:
        bin_low = xs.channel1.rois.roi01.bin_low.get()
        bin_high = xs.channel1.rois.roi01.bin_high.get()
        d = np.array(tbl[fluor_key])[0]
        if (d.ndim == 1):
            d = np.array(tbl[fluor_key])
        d = np.stack(d)
        if (d.ndim == 2):
            d = np.sum(d[:, bin_low:bin_high], axis=1)
        elif (d.ndim == 3):
            d = np.sum(d[:, :, bin_low:bin_high], axis=(1, 2))
        try:
            I0 = np.array(tbl['i0'])[0]
        except KeyError:
            I0 = np.array(tbl['sclr_i0'])
        if (normalize):
            y = d / I0
        else:
            y = d
    x = np.array(tbl[pos])[0]
    if (x.size == 1):
        x = np.array(tbl[pos])
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    dydx = np.gradient(y, x)
    try:
        with h5py.File('/home/xf05id1/current_user_data/nano_knife_edge_scan.h5', 'a') as hf:
            tmp_str = f'dataset_{id_str}'
            hf.create_dataset(tmp_str, data=[d, y, x, y]) #raw_cts,norm_cts,x_pos,y_pos
        #ftxt = open('/home/xf05id1/current_user_data/nano_knife_edge_scan.txt','a')
        #ftxt.write(data=[d,y,x,y])
        #ftxt.close()
    except:
        pass

    # Fit the raw data
    # def f_int_gauss(x, A, sigma, x0, y0, m)
    # def f_offset_erf(x, A, sigma, x0, y0):
    # def f_two_erfs(x, A1, sigma1, x1, y1,
    #                   A2, sigma2, x2, y2):
    p_guess = [0.5*np.amax(y),
               1.000,
               0.5*(x[0] + x[-1]) - 1.0,
               np.amin(y) + 0.5*np.amax(y),
               -0.5*np.amax(y),
               1.000,
               0.5*(x[0] + x[-1]) + 1.0,
               np.amin(y) + 0.5*np.amax(y)]
    try:
        # popt, _ = curve_fit(f_offset_erf, x, y, p0=p_guess)
        popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    C = 2 * np.sqrt(2 * np.log(2))
    print(f'\nThe beam size is {C * popt[1]:.4f} um')
    print(f'The beam size is {C * popt[5]:.4f} um')
    #print(f'\nThe left edge is at\t{popt[2]:.4f}.')
    #print(f'The right edge is at\t{popt[6]:.4f}.')
    print(f'The center is at\t{(popt[2]+popt[6])/2:.4f}.\n')

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_two_erfs(x_plot, *popt)
    # y_plot = f_offset_erf(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Display fit of raw data
    if (plot and 'plotme' in locals()):
        plotme.ax.cla()
        plotme.ax.plot(x, y, '*', label='Raw Data')
        if (plot_guess):
            plotme.ax.plot(x_plot, f_two_erfs(x_plot, *p_guess), '--', label='Guess fit')
        plotme.ax.plot(x_plot, y_plot, '-', label='Final fit')
        plotme.ax.set_title(f'Scan {id_str}')
        plotme.ax.set_xlabel(motor.name)
        if (normalize):
            plotme.ax.set_ylabel('Normalized ROI Counts')
        else:
            plotme.ax.set_ylabel('ROI Counts')
        plotme.ax.legend()

    # Use the fitted raw data to fit a Gaussian
    # def f_gauss(x, A, sigma, x0, y0, m):
    # try:
    #     if (high2low == True):
    #         p_guess = [np.amin(dydx_plot), popt[1], popt[2], 0, 0]
    #     else:
    #         p_guess = [np.amax(dydx_plot), popt[1], popt[2], 0, 0]

    #     popt2, _ = curve_fit(f_gauss, x_plot, dydx_plot, p0=p_guess)
    #     # popt2, _ = curve_fit(f_gauss, x, dydx, p0=p_guess)
    # except:
    #     print('Fit failed.')
    #     popt2 = p_guess
    # C = 2 * np.sqrt(2 * np.log(2))
    # try:
    #     p_guess = [np.amin(dydx), 1, x[np.argmin(dydx)], 0, 0]
    #     popt2, _ = curve_fit(f_gauss, x, dydx, p0=p_guess)
    #     print('beamsize =f' % (C*popt2[1]))
    # except:
    #     print('fail')
    #     popt2 = p_guess
    #     pass
    # try:
    #     p_guess = [np.amax(dydx), 1, x[np.argmax(dydx)], 0, 0]
    #     popt3, _ = curve_fit(f_gauss, x, dydx, p0=p_guess)
    #     print('beamsize =f' % (C*popt3[1]))
    # except:
    #     print('fail')
    #     popt3 = p_guess
    #     pass


    # # Plot the fit
    # plt.figure('Derivative')
    # plt.clf()
    # plt.plot(x, dydx, '*', label='dydx raw')
    # plt.plot(x_plot, dydx_plot, '-', label='dydx fit')
    # #plt.plot(x_plot, f_gauss(x_plot, *p_guess), '-', label='Guess')
    # plt.plot(x_plot, f_gauss(x_plot, *popt2), '-', label='Fit')
    # plt.plot(x_plot, f_gauss(x_plot, *popt3), '-', label='Fit')
    # plt.title('Scans' % (id_str))
    # plt.legend()

    # # Report findings
    # C = 2 * np.sqrt(2 * np.log(2))
    # print('\nThe beam size isf um' % (C * popt2[1]))
    # print('The edge is at.4f mm\n' % (popt2[2]))

# Written quickly
def plot_knife_edge(scanid=-1, fluor_key='fluor', use_trans=False, normalize=True, plot_guess=False,
                    bin_low=None, bin_high=None):
    # Get the scanid
    h = db[int(scanid)]
    id_str = h.start['scan_id']

    try:
        if (h.start['scaninfo']['fast_axis'] == 'NANOHOR'):
            pos = 'enc1'
        else:
            pos = 'enc2'
    except:
        print('Not a knife-edge scan')
        return

    # Get the information from the previous scan
    haz_data = False
    loop_counter = 0
    MAX_LOOP_COUNTER = 30
    print('Waiting for data...', end='', flush=True)
    while (loop_counter < MAX_LOOP_COUNTER):
        try:
            if (fly):
                tbl = db[int(id_str)].table('stream0', fill=True)
            else:
                tbl = db[int(id_str)].table(fill=True)
            haz_data = True
            print('done')
            break
        except:
            loop_counter += 1
            ttime.sleep(1)

    # Check if we haz data
    if (not haz_data):
        print('Data collection timed out!')
        return
    
    # Get the data
    if (use_trans == True):
        y = tbl['it'].values[0] / tbl['im'].values[0]
    else:
        if bin_low is None:
            bin_low = xs.channel1.rois.roi01.bin_low.get()
        if bin_high is None:
            bin_high = xs.channel1.rois.roi01.bin_high.get()
        d = np.array(tbl[fluor_key])[0]
        if (d.ndim == 1):
            d = np.array(tbl[fluor_key])
        d = np.stack(d)
        if (d.ndim == 2):
            d = np.sum(d[:, bin_low:bin_high], axis=1)
        elif (d.ndim == 3):
            d = np.sum(d[:, :, bin_low:bin_high], axis=(1, 2))
        try:
            I0 = np.array(tbl['i0'])[0]
        except KeyError:
            I0 = np.array(tbl['sclr_i0'])
        if (normalize):
            y = d / I0
        else:
            y = d
    x = np.array(tbl[pos])[0]
    if (x.size == 1):
        x = np.array(tbl[pos])
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    dydx = np.gradient(y, x)
    #try:
    #    hf = h5py.File('/home/xf05id1/current_user_data/knife_edge_scan.h5', 'a')
    #    tmp_str = 'dataset_%s' % id_str
    #    hf.create_dataset(tmp_str, data=[d,y,x,y]) #raw cts, norm_cts, x_pos, y_pos
    #    hf.close()
    #    ftxt = open('/home/xf05id1/current_user_data/knife_edge_scan.txt','a')
    #    ftxt.write(data=[d,y,x,y])
    #    ftxt.close()
    #except:
    #    pass

    # Fit the raw data
    # def f_int_gauss(x, A, sigma, x0, y0, m)
    # def f_offset_erf(x, A, sigma, x0, y0):
    # def f_two_erfs(x, A1, sigma1, x1, y1,
    #                   A2, sigma2, x2, y2):
    p_guess = [0.5*np.amax(y),
               1.000,
               0.5*(x[0] + x[-1]) - 1.0,
               np.amin(y) + 0.5*np.amax(y),
               -0.5*np.amax(y),
               1.000,
               0.5*(x[0] + x[-1]) + 1.0,
               np.amin(y) + 0.5*np.amax(y)]
    try:
        # popt, _ = curve_fit(f_offset_erf, x, y, p0=p_guess)
        popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    C = 2 * np.sqrt(2 * np.log(2))
    cent_position = (popt[2]+popt[6])/2
    print(f'The beam size is {C * popt[1]:.4f} um')
    print(f'The beam size is {C * popt[5]:.4f} um')

    #print(f'\nThe left edge is at\t{popt[2]:.4f}.')
    #print(f'The right edge is at\t{popt[6]:.4f}.')
    print(f'The center is at\t{(popt[2]+popt[6])/2:.4f}.')

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_two_erfs(x_plot, *popt)
    # y_plot = f_offset_erf(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Display fit of raw data
    fig, ax = plt.subplots()
    ax.plot(x, y, '*', label='Raw Data')
    if (plot_guess):
        ax.plot(x_plot, f_two_erfs(x_plot, *p_guess), '--', label='Guess fit')
    ax.plot(x_plot, y_plot, '-', label='Final fit')
    ax.set_title(f'Scan {id_str}')
    ax.legend()
    return cent_position 
