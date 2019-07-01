def collect_xrd(pos=[], empty_pos=[], acqtime=1, N=1,
                dark_frame=False, shutter=True):
    # Scan parameters
    if (acqtime < 0.0066392):
        print('The detector should not operate faster than 7 ms.')
        print('Changing the scan dwell time to 7 ms.')
        acqtime = 0.007
    
    N_pts = len(pos)
    if (pos == []):
        # pos = [[hf_stage.x.position, hf_stage.y.position, hf_stage.z.position]]
        pos = [[hf_stage.topx.position, hf_stage.y.position]]
    if (empty_pos != []):
        N_pts = N_pts + 1
    if (dark_frame):
        N_pts = N_pts + 1

    N_tot = N_pts * N

    # Setup detector
    xrd_det = [dexela]
    for d in xrd_det:
        d.cam.stage_sigs['acquire_time'] = acqtime
        d.cam.stage_sigs['acquire_period'] = acqtime + 0.100 
        d.cam.stage_sigs['num_images'] = 1
        d.stage_sigs['total_points'] = N_tot
        d.hdf5.stage_sigs['num_capture'] = N_tot

    
    # Collect dark frame
    if (dark_frame):
        # Check shutter
        if (shutter):
           yield from bps.mov(shut_b, 'Close')

        # Trigger detector
        yield from count(xrd_det, num=N)

        # Write to logfile
        logscan('xrd_count')

    if (empty_pos != []):
        # Move into position
        # yield from bps.mov(hf_stage.x, empty_pos[0])
        yield from bps.mov(hf_stage.topx, empty_pos[0])
        yield from bps.mov(hf_stage.y, empty_pos[1])
        if (len(empty_pos) == 3):
            yield from bps.mov(hf_stage.z, empty_pos[2])

        # Check shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Open')

        # Trigger the detector
        yield from count(xrd_det, num=N)
        # yield from bps.trigger_and_read(xrd_det)

        # Close shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Close')

        # Write to logfile
        logscan('xrd_count')

    # Loop through positions
    if (pos == []):
        pos = [[hf_stage.x.position, hf_stage.y.position]]
    for i in range(len(pos)):
        i = int(i)
        # Move into position
        # yield from bps.mov(hf_stage.x, pos[i][0])
        yield from bps.mov(hf_stage.topx, pos[i][0])
        yield from bps.mov(hf_stage.y, pos[i][1])
        if (len(pos[i]) == 3):
            yield from bps.mov(hf_stage.z, pos[i][2])

        # Check shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Open')

        # Trigger the detector
        yield from count(xrd_det, num=N)
        # yield from bps.trigger_and_read(xrd_det)

        # Close shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Close')

        # Write to logfile
        logscan('xrd_count')

        
def xrd_fly(*args, extra_dets=[dexela], **kwargs):
    kwargs.setdefault('xmotor', hf_stage.x)
    kwargs.setdefault('ymotor', hf_stage.y)
    _xs = kwargs.pop('xs', xs)
    kwargs.setdefault('flying_zebra', flying_zebra)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    # To fly both xs and merlin
    # yield from scan_and_fly_base([_xs, merlin], *args, **kwargs)
    # To fly only xs
    yield from scan_and_fly_base(dets, *args, **kwargs)

