print(f'Loading {__file__}...')

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


