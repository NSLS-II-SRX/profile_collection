print(f'Loading {__file__}...')
import h5py

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
    print('\nThe beam size isf um' % (1000 * C * popt2[1]))
    print('The edge is at.4f mm\n' % (popt2[2]))

# Run a knife-edge scan
def nano_knife_edge(motor, start, stop, stepsize, acqtime,
                    fly=True, high2low=False, use_trans=True,
                    scan_only=False, extra_dets=None):
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
    dets = [sclr1]
    # Add fluorescence detector
    if (use_trans == False):
        dets.append(xs2)
    # Add extra detectors
    if extra_dets is None:
        extra_dets = []
    dets = dets + extra_dets

    # Need to convert stepsize to number of points
    num = np.round((stop - start) / stepsize) + 1

    # Set counting time
    sclr1.preset_time.put(acqtime)
    if (use_trans == False):
        xs2.settings.acquire_time.put(acqtime)
        yield from abs_set(xs2.total_points, num)
    
    yield from mv(shut_b,'Open')

    # Run the scan
    if (motor.name == 'hf_stage_y'):
        if fly:
            yield from y_scan_and_fly(start, stop, num,
                                      hf_stage.x.position, hf_stage.x.position+0.001, 1,
                                      acqtime)
        else:
            yield from scan(dets, motor, start, stop, num)
    else:
        if fly:
            yield from scan_and_fly(start, stop, num,
                                    hf_stage.y.position, hf_stage.y.position+0.001, 1,
                                    acqtime)
        else:
            # table = LiveTable([motor])
            # @subs_decorator(table)
            # LiveTable([motor])
            yield from scan(dets, motor, start, stop, num)

    # Do not do fitting, only do the scan
    if (scan_only):
        return

    # Get the information from the previous scan
    haz_data = False
    loop_counter = 0
    MAX_LOOP_COUNTER = 30
    print('Waiting for data...', end='', flush=True)
    while (loop_counter < MAX_LOOP_COUNTER):
        try:
            if (fly == True):
                tbl = db[-1].table('stream0', fill=True)
            else:
                tbl = db[-1].table(fill=True)
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
    
    id_str = db[-1].start['scan_id']

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
        d = np.array(list(db[-1].data('xs2_channel1', fill=True)))
        # y = np.sum(d[:, 934:954], axis=(1))#if Pt lines
        y = np.sum(d[:, 934:954], axis=(1))/tbl['sclr_im'].values[0]#if Pt lines
        # y = np.sum(d[:, 961:981], axis=(1))/tbl['sclr_it'].values[0] #if Au lines
        # y = np.sum(d[:, 531:551], axis=(1)) #if Au lines
        # y = np.sum(np.array(tbl['xs2_channel1'])[0][:, 934:954], axis=(1))
        # y = y / np.array(tbl['i0'])[0]
    x = np.array(tbl[pos])
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    dydx = np.gradient(y, x)
    try:
        hf = h5py.File('/home/xf05id1/current_user_data/knife_edge_scan.h5', 'a')
        tmp_str = 'dataset_%s' id_str
        hf.create_dataset(tmp_str, data=[x,y])
        hf.close()
        ftxt = open('/home/xf05id1/current_user_data/knife_edge_scan.txt','a')
        ftxt.write(data=[x,y])
        ftxt.close()
    except:
        pass

    # Fit the raw data
    # def f_int_gauss(x, A, sigma, x0, y0, m)
    # def f_offset_erf(x, A, sigma, x0, y0):
    # def f_two_erfs(x, A1, sigma1, x1, y1,
    #                   A2, sigma2, x2, y2):
    p_guess = [0.5*np.amax(y),
               1.000,
               0.5*(x[0] + x[-1]) - 2.5,
               np.amin(y) + 0.5*np.amax(y),
               -0.5*np.amax(y),
               1.000,
               0.5*(x[0] + x[-1]) + 2.5,
               np.amin(y) + 0.5*np.amax(y)]
    # if high2low:
    #     p_guess[0] = -0.5 * np.amin(y)
    try:
        # popt, _ = curve_fit(f_offset_erf, x, y, p0=p_guess)
        popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    C = 2 * np.sqrt(2 * np.log(2))
    print('\nThe beam size isf um' % (C * popt[1]))
    print('\nThe beam size isf um' % (C * popt[5]))

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_two_erfs(x_plot, *popt)
    # y_plot = f_offset_erf(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Display fit of raw data
    plt.figure('Raw')
    plt.clf()
    plt.plot(x, y, '*', label='Raw Data')
    #plt.plot(x_plot, f_int_gauss(x_plot, *p_guess), '-', label='Guess fit')
    plt.plot(x_plot, y_plot, '-', label='Final fit')
    plt.title('Scans' % (id_str))
    plt.legend()

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
    C = 2 * np.sqrt(2 * np.log(2))
    try:
        p_guess = [np.amin(dydx), 1, x[np.argmin(dydx)], 0, 0]
        popt2, _ = curve_fit(f_gauss, x, dydx, p0=p_guess)
        print('beamsize =f' % (C*popt2[1]))
    except:
        print('fail')
        popt2 = p_guess
        pass
    try:
        p_guess = [np.amax(dydx), 1, x[np.argmax(dydx)], 0, 0]
        popt3, _ = curve_fit(f_gauss, x, dydx, p0=p_guess)
        print('beamsize =f' % (C*popt3[1]))
    except:
        print('fail')
        popt3 = p_guess
        pass


    # Plot the fit
    plt.figure('Derivative')
    plt.clf()
    plt.plot(x, dydx, '*', label='dydx raw')
    plt.plot(x_plot, dydx_plot, '-', label='dydx fit')
    #plt.plot(x_plot, f_gauss(x_plot, *p_guess), '-', label='Guess')
    plt.plot(x_plot, f_gauss(x_plot, *popt2), '-', label='Fit')
    plt.plot(x_plot, f_gauss(x_plot, *popt3), '-', label='Fit')
    plt.title('Scans' % (id_str))
    plt.legend()

    # Report findings
    C = 2 * np.sqrt(2 * np.log(2))
    print('\nThe beam size isf um' % (C * popt2[1]))
    print('The edge is at.4f mm\n' % (popt2[2]))

# Run a knife-edge scan
# def nano_knife_edge_scanonly(motor, start, stop, stepsize, acqtime,
#                fly=True, high2low=False, use_trans=True):
#     """
#     motor       motor   motor used for scan
#     start       float   starting position
#     stop        float   stopping position
#     stepsize    float   distance between data points
#     acqtime     float   counting time per step
#     fly         bool    if the motor can fly, then fly that motor
#     high2low    bool    scan from high transmission to low transmission
#                         ex. start will full beam and then block with object (knife/wire)
#     """
# 
#     # Set detectors
#     det = [sclr1]
#     if (use_trans == False):
#         det.append(xs2)
# 
#     # Need to convert stepsize to number of points
#     num = np.round((stop - start) / stepsize) + 1
# 
#     # Set counting time
#     sclr1.preset_time.put(acqtime)
#     if (use_trans == False):
#         xs2.settings.acquire_time.put(acqtime)
#         yield from abs_set(xs2.total_points, num)
# 
#     # Run the scan
#     if (motor.name == 'hf_stage_y'):
#         if fly:
#             yield from y_scan_and_fly(start, stop, num,
#                                       hf_stage.x.position, hf_stage.x.position+0.001, 1,
#                                       acqtime)
#         else:
#             yield from scan(det, motor, start, stop, num)
#     else:
#         if fly:
#             yield from scan_and_fly(start, stop, num,
#                                     hf_stage.y.position, hf_stage.y.position+0.001, 1,
#                                     acqtime)
#         else:
#             # table = LiveTable([motor])
#             # @subs_decorator(table)
#             # LiveTable([motor])
#             yield from scan(det, motor, start, stop, num)
# 
#     # Get the information from the previous scan
#     haz_data = False
#     loop_counter = 0
#     MAX_LOOP_COUNTER = 30
#     print('Waiting for data...', end='', flush=True)
#     while (loop_counter < MAX_LOOP_COUNTER):
#         try:
#             if (fly == True):
#                 tbl = db[-1].table('stream0', fill=True)
#             else:
#                 tbl = db[-1].table(fill=True)
#             haz_data = True
#             print('done')
#             break
#         except:
#             loop_counter += 1
#             time.sleep(1)
# 
#     # Check if we haz data
#     if (not haz_data):
#         print('Data collection timed out!')
#         return
#     
# 
