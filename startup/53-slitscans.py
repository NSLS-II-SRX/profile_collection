
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

