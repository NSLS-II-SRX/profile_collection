print(f'Loading {__file__}...')

def ssa_hcen_scan(start, stop, num, shutter=True):
    # Setup metadata
    scan_md = {}
    get_stock_md(scan_md)

    # Setup LiveCallbacks
    liveplotfig1 = plt.figure()
    liveplotx = 'h_cen_readback'
    # liveploty = im.name
    liveploty = 'xbpm2_sumX'
    livetableitem = ['h_cen_readback', im.name, i0.name, xbpm2_sumX]
    livecallbacks = [LiveTable(livetableitem)]
    #                  LivePlot(liveploty, x=liveplotx, fig=liveplotfig1)]

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

def slit_nanoKB_scan(slit_motor, sstart, sstop, sstep,
                     edge_motor, estart, estop, estep, acqtime,
                     shutter=True):
    """
    Scan the beam defining slits (JJs) across the mirror.
    Perform a knife-edge scan at each position to check focal position.

    Parameters
    ----------
    slit_motor : motor
    slit motor that you want to scan
    sstart :
    """

    scan_md = {}
    get_stock_md(scan_md)


    # calculate number of points
    snum = np.int(np.abs(np.round((sstop - sstart)/sstep)) + 1)
    enum = np.int(np.abs(np.round((estop - estart)/estep)) + 1)

    # Setup detectors
    dets = [sclr1, xs]

    # Set counting time
    sclr1.preset_time.put(acqtime)
    xs.external_trig.put(False)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put(enum * snum)

    # LiveGrid
    livecallbacks = []
    roi_name = 'roi{:02}'.format(1)
    roi_key = getattr(xs.channel1.rois, roi_name).value.name
    livecallbacks.append(LiveTable([slit_motor.name, edge_motor.name, roi_key]))
    livecallbacks.append(LivePlot(roi_key, x=edge_motor.name))
    # xlabel='Position [um]', ylabel='Intensity [cts]'))
    plot_lines1 = LivePlot('')
    plot_lines2 = LivePlot('')
    plot_fit = LivePlot('')

    myplan = grid_scan(dets,
                       slit_motor, sstart, sstop, snum,
                       edge_motor, estart, estop, enum, False,
                       md=scan_md)
    myplan = subs_wrapper(myplan,
                          {'all': livecallbacks})

    # Open shutter
    if (shutter):
        yield from mv(shut_b,'Open')

    # grid scan
    uid = yield from myplan

    # Open shutter
    if (shutter):
        yield from mv(shut_b,'Close')

    scan_id = db[uid].start['scan_id']
    slit_nanoKB_scan_corr(scan_id, from_RE=[plot_lines1, plot_lines2, plot_fit])

    return uid


def slit_nanoKB_scan_corr(scan_id, from_RE=[], orthogonality=0, interp_range=None, bin_low=None, bin_high=None):

    # scan_id = '70190'
    h = db[int(scan_id)]
    #tbl = h.table('stream0', fill=True)
    #tbl = h.table(fill=True)
    # df = h.table(fill=True)
    #h.table().keys()
    # Get the information from the previous scan
    haz_data = False
    loop_counter = 0
    MAX_LOOP_COUNTER = 15
    print('Waiting for data...', end='', flush=True)
    while (loop_counter < MAX_LOOP_COUNTER):
        try:
            df = h.table(fill=True)
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
    
    #df.shape
    #df.dtypes
    
    #vertical scan
    cts = np.array(list(h.data('xs_channel1', fill=True)))
    if bin_low is None:
        bin_low = xs.channel1.rois.roi01.bin_low.get()
    if bin_high is None:
       bin_high = xs.channel1.rois.roi01.bin_high.get()
    cts = np.sum(cts[:, bin_low:bin_high], axis=1) #Pt

    if 'nano_stage_sy' in h.start['motors']:
        flag_dir = 'VER'
        pos = df['nano_stage_sy']
        slit_pos = df['jjslits_v_trans']
    elif 'nano_stage_sx' in h.start['motors']:
        flag_dir = 'HOR'
        pos = df['nano_stage_sx']
        slit_pos = df['jjslits_h_trans']
    else:
        print('Unknown motor')
        return

    I0 = df['sclr_i0']
    
    numpts = h.start['shape'][1]
    numline = h.start['shape'][0]
    
    ind_line = np.linspace(0, len(cts), numline, endpoint=False, dtype=np.int)
    # ind_line = ind_line.astype(int)
    
    # numline_array = np.arange(len(cts)/numpts)
    line_seq = np.zeros((int(numline), int(numpts)))
    pos_seq = np.zeros((int(numline), int(numpts)))
    I0_seq = np.zeros((int(numline), int(numpts)))
    slit_pos_seq = np.zeros(int(numline))
    
    for i in range(int(numline)):
        line_seq[i,:] = cts[ind_line[i]:ind_line[i] + int(numpts)]
        pos_seq[i,:] = pos[ind_line[i]:ind_line[i] + int(numpts)]
        I0_seq[i,:] = I0[ind_line[i]:ind_line[i] + int(numpts)]
        slit_pos_seq[i] = slit_pos[ind_line[i] + 1]
    
    pos_seq_plt = pos_seq[0, :]
    norm_line_seq = line_seq / I0_seq
    # line_seq.shape
    # At this point, i haz data and full variables so now itz timez to plot

    if from_RE == []:
        _, ax = plt.subplots()
    else:
        ax = from_RE[0].ax
    for i in range(numline):
        ax.plot(pos_seq_plt, norm_line_seq[i, :], label=f'y = {i+1}')
    ax.set_ylabel('Slit position')
    ax.set_ylabel('Normalized Signal')
    ax.set_title(f'Scan {scan_id}')
    ax.legend(loc='upper left')

    if from_RE == []:
        _, ax = plt.subplots()
    else:
        ax = from_RE[1].ax
    for i in range(numline):
        ax.plot(pos_seq_plt, line_seq[i, :], label=f'y = {i+1}')
    ax.set_ylabel('Slit position')
    ax.set_ylabel('Raw Signal')
    ax.set_title(f'Scan {scan_id}')
    ax.legend(loc='upper left')

    #line fit
    # Error function with offset
    def f_offset_erf(x, A, sigma, x0, y0):
        x_star = (x - x0) / sigma
        return A * erf(x_star / np.sqrt(2)) + y0
    
    def f_two_erfs(x, A1, sigma1, x1, y1, A2, sigma2, x2, y2):
        x1_star = (x - x1) / sigma1
        x2_star = (x - x2) / sigma2
    
        f_combo = f_offset_erf(x, A1, sigma1, x1, y1) + f_offset_erf(x, A2, sigma2, x2, y2)
        return f_combo
    
    def line_fit(x,y):
        p_guess = [0.5*np.amax(y),1.000,0.5*(x[0] + x[-1]) - 2.5,np.amin(y) + 0.5*np.amax(y),-0.5*np.amax(y),1.000,0.5*(x[0] + x[-1]) + 2.5,np.amin(y) + 0.5*np.amax(y)]       
        try:
                # popt, _ = curve_fit(f_offset_erf, x, y, p0=p_guess)
            popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
        except:
            print('Raw fit failed.')
            popt = p_guess
    
        C = 2 * np.sqrt(2 * np.log(2))
        line_pos = (popt[2]+popt[6])/2
        print('\nThe center beam position is %f um' % ((popt[2]+popt[6])/2))
        return line_pos
    
    line_pos_seq = np.zeros (int(numline))
    for i in range(numline):
        line_pos_seq[i] = line_fit(pos_seq_plt, norm_line_seq[i,:])   

    if interp_range is None:
        if flag_dir == 'VER':
            interp_range = np.arange(numline)[:-1]
        else:
            interp_range = np.arange(numline)[1:-1]

    calpoly_fit = np.polyfit(slit_pos_seq[interp_range], line_pos_seq[interp_range]/1000, orthogonality+1, full=True)
    p = np.poly1d(calpoly_fit[0])
    line_plt = p(slit_pos_seq[interp_range])
    p2v_line_pos = np.max(line_pos_seq[interp_range])-np.min(line_pos_seq[interp_range])

    # Mirror parameters
    f_v = 295*1e+3  # um
    f_h = 125*1e+3
    theta_v = 3 #mrad
    theta_h = 3 #mrad
    conversion_factor_orth = np.array([-1.6581375e-4, 5.89e-4]) #unit: p/urad (V x H)
    pitch_motion_conversion = np.array([225, 100]) # unit: mm (V x H)
    delta_fine_pitch = 0.0 # unit: um
    
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


    print(f'p is {calpoly_fit[0]}')
    #print(f'residual is {calpoly_fit[1]*1e+6} nm')
    print(f'P2V of line position is {p2v_line_pos} um')
    defocus = -calpoly_fit[0][0] * C_f
    delta_theta = calpoly_fit[0][0] * C_theta
    line_move = 2 * delta_theta * C_f * 1e-3
    print('defocus is ' '{:7.3f}'.format(defocus), 'um. Vkb correct by this amount.')
    print('equivalent to ' '{:7.6f}'.format(delta_theta), 'mrad. Hkb correct by this amount.')
    #print('actuator should move by' '{:7.3f}'.format(actuator_move), 'um.')
    print('Line feature should move' '{:7.3f}'.format(line_move), 'um.')

    if orthogonality == 1:
        delta_fine_pitch = calpoly_fit[0][0]/conversion_factor_orth*1e-3*pitch_motion_conversion
        delta_theta_quad = calpoly_fit[0][0]/conversion_factor_orth
        delta_focal_plane_z = delta_theta_quad*1e-3/C_theta*C_f
        print('quadratic term corresponds to pitch angle' '{:7.3f}'.format(delta_theta_quad), 'urad.')
        print('quadratic term corresponds to fine pitch move' '{:7.3f}'.format(delta_fine_pitch), 'um.')
        print('quadratic term corresponds to coarse Z ' '{:7.3f}'.format(delta_focal_plane_z), 'um.')


    if from_RE == []:
        _, ax = plt.subplots()
    else:
        ax = from_RE[2].ax
    ax.plot(slit_pos_seq, line_pos_seq/1000, 'ro', slit_pos_seq[interp_range], line_plt)
    ax.set_title(f'scan {scan_id}')
    ax.set_xlabel(f'Slit Pos (mm)')
    ax.set_ylabel(f'Line Pos (mm)')

    fname = f'slitscan_{scan_id}.png'
    root = '/home/xf05id1/current_user_data/'
    try:
        os.makedirs(root, exist_ok=True)
        plt.savefig(root + fname, dpi=300)
    except:
        print('Could not save plot.')

