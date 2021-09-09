print(f'Loading {__file__}...')

def ssa_hcen_scan(start, stop, num,
                  shutter=True, plot=True, plot_guess=False, scan_only=False):
    # Setup metadata
    scan_md = {}

    # Setup LiveCallbacks
    def my_factory(name):
        fig = plt.figure(num=name)
        ax = fig.gca()
        ax.cla()
        return fig, ax

    liveplotx = 'slt_ssa_h_cen_readback'
    liveploty = 'xbpm2_sumX'
    livetableitem = [liveplotx, 'xbpm2_sumX']
    plotme = HackLivePlot(liveploty, x=liveplotx,
                          fig_factory=partial(my_factory, name='SSA Slit Scan'))
    livecallbacks = [LiveTable(livetableitem),
                     plotme]

    # Setup the scan
    @subs_decorator(livecallbacks)
    def myscan():
        yield from scan([slt_ssa.h_cen, sclr1, xbpm2],
                        slt_ssa.h_cen,
                        start,
                        stop,
                        num,
                        md=scan_md)

    # Record old position
    old_pos = slt_ssa.h_cen.position

    # Run the scan
    if (shutter):
        yield from mv(shut_b, 'Open')

    ret = yield from myscan()

    if (shutter):
        yield from mv(shut_b, 'Close')

    # Return to old position
    yield from mv(slt_ssa.h_cen, old_pos)

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
    x_key = 'slt_ssa_h_cen_readback'
    y_key = 'xbpm2_sumX'
    x = tbl[x_key].values
    y = -tbl[y_key].values # the sum used to be negative
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    dydx = np.gradient(y, x)
    try:
        with h5py.File('/home/xf05id1/current_user_data/ssa_hcen_scan.h5', 'a') as hf:
            tmp_str = f'dataset_{id_str}'
            hf.create_dataset(tmp_str, data=[x, y]) # ssa_h_cen, bpm_cts
    except:
        pass

    # Fit the raw data
    # def f_gauss(x, A, sigma, x0, y0, m):
    # def f_int_gauss(x, A, sigma, x0, y0, m)
    # def f_offset_erf(x, A, sigma, x0, y0):
    # def f_two_erfs(x, A1, sigma1, x1, y1,
    #                   A2, sigma2, x2, y2):
    p_guess = [np.amin(y),
               0.017,
               0.5*(x[0] + x[-1]),
               np.mean(y[:3]),
               0]
    try:
        popt, _ = curve_fit(f_gauss, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    C = 2 * np.sqrt(2 * np.log(2))
    print(f'\nThe beam size is {C * popt[1]:.4f} mm')
    print(f'The center is at\t{popt[2]:.4f} mm.\n')

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_gauss(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Display fit of raw data
    if (plot and 'plotme' in locals()):
        plotme.ax.cla()
        plotme.ax.plot(x, y, '*', label='Raw Data')
        if (plot_guess):
            plotme.ax.plot(x_plot, f_gauss(x_plot, *p_guess), '--', label='Guess fit')
        plotme.ax.plot(x_plot, y_plot, '-', label='Final fit')
        plotme.ax.set_title(f'Scan {id_str}')
        plotme.ax.set_xlabel(x_key)
        plotme.ax.set_ylabel(y_key)
        plotme.ax.legend()

    # print(ret)
    return ret

def JJ_scan(motor, start, stop, num, shutter=True):
    # Setup metadata
    scan_md = {}
    get_stock_md(scan_md)

    # Setup LiveCallbacks
    liveplotfig1 = plt.figure()
    liveplotx = motor.name
    liveploty = im.name
    livetableitem = [motor.name, im.name, i0.name]
    livecallbacks = [LiveTable(livetableitem),
                     LivePlot(liveploty, x=liveplotx, fig=liveplotfig1)]

    # Setup the scan
    @subs_decorator(livecallbacks)
    def myscan():
        yield from scan([motor, sclr1],
                        motor,
                        start,
                        stop,
                        num,
                        md=scan_md)

    # Record old position
    old_pos = motor.position

    # Run the scan
    if (shutter):
        yield from mv(shut_b, 'Open')

    ret = yield from myscan()

    if (shutter):
        yield from mv(shut_b, 'Close')

    # Return to old position
    yield from mv(motor, old_pos)

    return ret

def slit_nanoflyscan(scan_motor, scan_start, scan_stop, scan_stepsize, acqtime,
                    slit_motor, slit_start, slit_stop, slit_stepsize, slitgap_motor, slit_gap,
                    normalize=False, scan_only=False, shutter=False,
                    plot=True, plot_guess=False):
    """
    scan_motor       motor   motor used for scan
    scan_start       float   starting position
    scan_stop        float   stopping position
    scan_stepsize    float   distance between data points
    slit_motor       motor   slit motor used for scan
    slit_start       float   starting position
    slit_stop        float   stopping position
    slit_stepsize    float   distance between data points
    slit_gap         float   the gap to close the slit to
    acqtime          float   counting time per step
    """

    if ('nano_stage_s' not in scan_motor.name):
        print('Please use a supported motor: nano_stage.sx or nano_stage.sy')
        print('Exiting...')
        return

    slit_orig_gap = slitgap_motor.user_readback.get()
    slit_orig_pos = slit_motor.user_readback.get()

    # close the gap for scan
    yield from mov(slitgap_motor, slit_gap)

    # Need to convert stepsize to number of points
    snum = np.round((scan_stop - scan_start) / scan_stepsize) + 1
    lnum = np.round((slit_stop - slit_start) / slit_stepsize) + 1
    lnum_list = np.arange(lnum)

    slit_pos = slit_start + lnum_list * slit_stepsize

    # Define a figure factory
    def my_factory(name):
        fig = plt.figure(num=name)
        ax = fig.gca()
        return fig, ax

    # Open the shutter
    yield from check_shutters(True, 'Open') 
    
    # Setup the scan
    plotme = HackLiveFlyerPlot(xs.channel1.rois.roi01.value.name,
                               xstart=scan_start,
                               xstep=(scan_stop-scan_start)/(snum-1),
                               xlabel=scan_motor.name,
                               fig_factory=partial(my_factory, name='Slit Scan'))

    if (scan_motor.name == 'nano_stage_sx'):
        y0 = nano_stage.sy.user_readback.get()
        @subs_decorator(plotme)
        def _knife_plan():
           yield from  nano_scan_and_fly(scan_start, scan_stop, snum,
                                         y0, y0, 1, acqtime,
                                         shutter=False, plot=False)
    else:
        x0 = nano_stage.sx.user_readback.get()
        @subs_decorator(plotme)
        def _knife_plan():
            yield from nano_y_scan_and_fly(scan_start, scan_stop, snum,
                                           x0, x0, 1, acqtime,
                                           shutter=False, plot=False)

    def _plan():
        uid = yield from _knife_plan()
        return uid
    
    uid_list = []
    for ii in slit_pos:
        yield from mov(slit_motor, ii)
        uid = yield from _plan()
        uid_list.append(uid)

    # Finish up
    yield from check_shutters(True, 'Close') 
    yield from mov(slit_motor, slit_orig_pos)
    yield from mov(slitgap_motor, slit_orig_gap)

    return uid_list


def slit_nanoflyscan_cal(scan_id_list=[], interp_range=None, orthogonality=False,
                         bin_low=None, bin_high=None, normalize=True):
   
    """
    This function takes a list of scan_id, process them one by one, then analyzes the optical abberations.

    scan_id          list    a list of slitscan ids
    interp_range     list    list of the fitting range, can drop some bad points
    orthogonality    bool    calculate the spherical abb or the astigmatism
    bin_low          integer the lower limit of ROI (usually Pt or Au), by default it's taking roi1
    bin_high         integer uppper limit of ROI
    normalize        bool    whether or not to normalize the data by the ion chamber, I0
    """

    numline = len(scan_id_list)
    slit_range = np.zeros((numline,), dtype=np.float)
    line_pos_seq = np.zeros((numline,), dtype=np.float)

    # Mirror parameters
    f_v = 295 * 1e+3  # um
    f_h = 125 * 1e+3
    theta_v = 3 # mrad
    theta_h = 3 # mrad
    L_h = 100 # mm  Length of horizontal mirror
    conversion_factor_orth = np.array([-1.6581375e-4, 5.89e-4]) #unit: p/urad (V x H)
    pitch_motion_conversion = np.array([225, 100]) # unit: mm (V x H)
    delta_fine_pitch = 0.0 # unit: um
 
    for idx, scan_id in enumerate(scan_id_list):
        h = db[int(scan_id)]
        scan_id = h.start['scan_id']

        # Get the information from the previous scan
        haz_data = False
        loop_counter = 0
        MAX_LOOP_COUNTER = 15
        print(f'Scan ID: {int(scan_id):d}\nWaiting for data...', end='', flush=True)
        while (loop_counter < MAX_LOOP_COUNTER):
            try:
                tbl = h.table('stream0', fill=True)
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
        
        fluor_key = 'fluor'
        d = np.array(tbl[fluor_key])[0]
        if 'nano_stage_sx' in h.start['scan']['fast_axis']['motor_name']:
            flag_dir = 'HOR'
            pos = 'enc1'
            slit_range[idx] = h.table('baseline')['jjslits_h_trans'][1]
        elif 'nano_stage_sy' in h.start['scan']['fast_axis']['motor_name']:
            flag_dir = 'VER'
            pos = 'enc2'
            slit_range[idx] = h.table('baseline')['jjslits_v_trans'][1]
        else:
            print('Cannot automatically determine entrance slit positions. Exiting...')
            return

        if bin_low is None:
            bin_low = xs.channel1.rois.roi01.bin_low.get()
        if bin_high is None:
           bin_high = xs.channel1.rois.roi01.bin_high.get()
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
        x = x.astype(np.float64) # position data
        y = y.astype(np.float64) # fluo data
       
        numpts = x.shape
        
    # ind_line = np.linspace(0, len(cts), numline, endpoint=False, dtype=np.int)
    
    # numline_array = np.arange(len(cts)/numpts)
    # line_seq = np.zeros((int(numline), int(numpts)))
    # pos_seq = np.zeros((int(numline), int(numpts)))
    # I0_seq = np.zeros((int(numline), int(numpts)))
    # slit_pos_seq = np.zeros(int(numline))
    
    # for i in range(int(numline)):
    #     line_seq[i,:] = cts[ind_line[i]:ind_line[i] + int(numpts)]
    #     pos_seq[i,:] = pos[ind_line[i]:ind_line[i] + int(numpts)]
    #     I0_seq[i,:] = I0[ind_line[i]:ind_line[i] + int(numpts)]
    #     slit_pos_seq[i] = slit_pos[ind_line[i] + 1]
    
    # pos_seq_plt = pos_seq[0, :]
    # norm_line_seq = line_seq / I0_seq
    # At this point, i haz data and full variables so now itz timez to plot

        # if from_RE == []:
        #     _, ax = plt.subplots()
        # else:
        #     ax = from_RE[0].ax
        # #for i in range(numline):
        # ax.plot(x, y, label=f'y = {i+1}')
        # ax.set_ylabel('Slit position')
        # ax.set_ylabel('Normalized Signal')
        # ax.set_title(f'Scan {scan_id}')
        # ax.legend(loc='upper left')

        # if from_RE == []:
        #     _, ax = plt.subplots()
        # else:
        #     ax = from_RE[1].ax
        # #for i in range(numline):
        # ax.plot(x, y, label=f'y = {i+1}')
        # ax.set_ylabel('Slit position')
        # ax.set_ylabel('Raw Signal')
        # ax.set_title(f'Scan {scan_id}')
        # ax.legend(loc='upper left')

        # line fit
        # Error function with offset
        def f_offset_erf(x, A, sigma, x0, y0):
            x_star = (x - x0) / sigma
            return A * erf(x_star / np.sqrt(2)) + y0
        
        def f_two_erfs(x, A1, sigma1, x1, y1, A2, sigma2, x2, y2):
            x1_star = (x - x1) / sigma1
            x2_star = (x - x2) / sigma2
        
            f_combo = f_offset_erf(x, A1, sigma1, x1, y1) + f_offset_erf(x, A2, sigma2, x2, y2)
            return f_combo
        
        def line_fit(x, y):
            p_guess = [0.5 * np.amax(y),
                       1.000,
                       0.5 * (x[0] + x[-1]) - 2.5,
                       np.amin(y) + 0.5 * np.amax(y),
                       -0.5 * np.amax(y),
                       1.000,
                       0.5 * (x[0] + x[-1]) + 2.5,
                       np.amin(y) + 0.5 * np.amax(y)]       
            try:
                popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
            except:
                print('Raw fit failed.')
                popt = p_guess
        
            C = 2 * np.sqrt(2 * np.log(2))
            line_pos = (popt[2] + popt[6]) / 2
            print(f'The center beam position is {line_pos:.4f} um\n')
            return line_pos
        
        line_pos_seq[idx] = line_fit(x, y)  

    if interp_range is None:
        interp_range = np.arange(numline)

    calpoly_fit = np.polyfit(slit_range[interp_range], line_pos_seq[interp_range]/1000, orthogonality+1, full=True)
    p = np.poly1d(calpoly_fit[0])
    line_plt = p(slit_range[interp_range])
    p2v_line_pos = np.max(line_pos_seq[interp_range])-np.min(line_pos_seq[interp_range])
   
    if flag_dir == 'VER':
        C_f = f_v
        C_theta = theta_v
        conversion_factor_orth = conversion_factor_orth[0]
        pitch_motion_conversion = pitch_motion_conversion[0] 
    else:
        C_f = f_h
        C_theta = theta_h
        conversion_factor_orth = conversion_factor_orth[1]
        pitch_motion_conversion = pitch_motion_conversion[1] 


    defocus = -calpoly_fit[0][0] * C_f
    delta_theta = calpoly_fit[0][0] * C_theta
    actuator_move_h = delta_theta * L_h
    line_move_h = -2 * delta_theta * C_f * 1e-3
    print('Fitting results:')
    print(f'\tp is {calpoly_fit[0]}')
    # print(f'residual is {calpoly_fit[1]*1e+6} nm')
    print(f'\tP2V of line position is {p2v_line_pos:.4f} um')
    if (flag_dir == 'VER'):
        print(f'\tDefocus is {defocus:7.3f} um. Vkb correct by this amount.')
    if (flag_dir == 'HOR'):
        print(f'\tEquivalent to {delta_theta:7.6f} mrad. Hkb correct by this amount.')
        print(f'\tActuator should move by {actuator_move_h:7.3f} um.')
        print(f'\tLine feature should move {line_move_h:7.3f} um for h mirror pitch correction')

    if (np.abs(defocus) < 100 or orthogonality == 1):
        delta_fine_pitch = calpoly_fit[0][0]/conversion_factor_orth*1e-3*pitch_motion_conversion
        delta_theta_quad = calpoly_fit[0][0]/conversion_factor_orth
        delta_focal_plane_z = delta_theta_quad*1e-3/C_theta*C_f
        print('\nOrthagonality correction:')
        print(f'\tQuadratic term corresponds to pitch angle {delta_theta_quad:7.3f} urad.')
        print(f'\tQuadratic term corresponds to fine pitch move {delta_fine_pitch:7.3f} um.')
        print(f'\tQuadratic term corresponds to coarse Z {delta_focal_plane_z:7.3f} um.')


    # Doesn't work in RunEngine. Need to make Liveplot.
    fig, ax = plt.subplots()
    ax.plot(slit_range, line_pos_seq/1000, 'ro', slit_range[interp_range], line_plt)
    ax.set_title(f'Scan {scan_id}')
    ax.set_xlabel(f'Slit Pos (mm)')
    ax.set_ylabel(f'Line Pos (mm)')

    fname = f'slitscan_{scan_id_list[0]}.png'
    root = '/home/xf05id1/current_user_data/knife_edge_scans/'
    try:
        os.makedirs(root, exist_ok=True)
        plt.savefig(root + fname, dpi=300)
    except:
        print('Could not save plot.')


def focusKB(direction, **kwargs):
    """
    NEEDS TESTING!

    direction   string  direction to scan in
                        ['hor' or 'ver']
    """

    if 'hor' in direction.lower():
        kwargs.setdefault('scan_motor', nano_stage.sx)
        kwargs.setdefault('slit_motor', jjslits.h_trans)
        slit_range = 0.500
        kwargs.setdefault('slit_stepsize', 0.05)
        kwargs.setdefault('slitgap_motor', jjslits.h_gap)
        kwargs.setdefault('slit_gap', 0.05)
        N = 11
    elif 'ver' in direction.lower():
        kwargs.setdefault('scan_motor', nano_stage.sy)
        kwargs.setdefault('slit_motor', jjslits.v_trans)
        slit_range = 1.00
        kwargs.setdefault('slit_stepsize', 0.1)
        kwargs.setdefault('slitgap_motor', jjslits.v_gap)
        kwargs.setdefault('slit_gap',  0.10)
        N = 11
    else:
        print("This is for vertical or horizontal scans. Please choose one of these directions")

    kwargs.setdefault('scan_start', -8)
    kwargs.setdefault('scan_stop', 8)
    kwargs.setdefault('scan_stepsize', 0.050)
    kwargs.setdefault('acqtime', 0.100)
    
    slit_center = kwargs['slit_motor'].user_readback.get()
    kwargs.setdefault('slit_start', slit_center - 0.5 * slit_range)
    kwargs.setdefault('slit_stop', slit_center + 0.5 * slit_range)

    # print(*kwargs)
    uids = yield from slit_nanoflyscan(**kwargs)

    # Fit the data
    # N = len(uids)
    scanids = np.linspace(-N, -1, num=N)
    slit_nanoflyscan_cal(scan_id_list=scanids, interp_range=scanids[1:-1].astype('int'), orthogonality=False)


