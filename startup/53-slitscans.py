print(f'Loading {__file__}...')

def ssa_hcen_scan(start, stop, num, shutter=True):
    # Setup metadata
    scan_md = {}
    get_stock_md(scan_md)

    # Setup LiveCallbacks
    liveplotfig1 = plt.figure()
    liveplotx = 'h_cen_readback'
    liveploty = im.name
    livetableitem = ['h_cen_readback', im.name, i0.name]
    livecallbacks = [LiveTable(livetableitem),
                     LivePlot(liveploty, x=liveplotx, fig=liveplotfig1)]

    # Setup the scan
    @subs_decorator(livecallbacks)
    def myscan():
        yield from scan([slt_ssa.h_cen, sclr1],
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
    dets = [sclr1, xs2]

    # Set counting time
    sclr1.preset_time.put(acqtime)
    xs2.external_trig.put(False)
    xs2.settings.acquire_time.put(acqtime)
    xs2.total_points.put(enum * snum)

    # LiveGrid
    livecallbacks = []
    roi_name = 'roi{:02}'.format(1)
    roi_key = getattr(xs2.channel1.rois, roi_name).value.name
    livecallbacks.append(LiveTable([slit_motor.name, edge_motor.name, roi_key]))
    livecallbacks.append(LivePlot(roi_key, x=edge_motor.name))
    # xlabel='Position [um]', ylabel='Intensity [cts]'))

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

    return uid
