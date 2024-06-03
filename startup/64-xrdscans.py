print(f'Loading {__file__}...')

import skimage.io as io


def update_scan_id():
    scanrecord.current_scan.put(db[-1].start['uid'][:6])
    scanrecord.current_scan_id.put(str(RE.md['scan_id']))


def collect_xrd(pos=[], empty_pos=[], acqtime=1, N=1,
                dark_frame=False, shutter=True):
    # Scan parameters
    if (acqtime < 0.0066392):
        print('The detector should not operate faster than 7 ms.')
        print('Changing the scan dwell time to 7 ms.')
        acqtime = 0.007

    N_pts = len(pos)
    if (pos == []):
        pos = [[hf_stage.x.position, hf_stage.y.position, hf_stage.z.position]]
        # pos = [[hf_stage.topx.position, hf_stage.y.position]]
    if (empty_pos != []):
        N_pts = N_pts + 1
    if (dark_frame):
        N_pts = N_pts + 1

    N_tot = N_pts * N

    # Setup detector
    xrd_det = [dexela]
    for d in xrd_det:
        # d.cam.stage_sigs['acquire_time'] = acqtime
        # d.cam.stage_sigs['acquire_period'] = acqtime + 0.100
        # d.cam.stage_sigs['num_images'] = 1
        # d.stage_sigs['total_points'] = N_tot
        # d.hdf5.stage_sigs['num_capture'] = N_tot
        d.stage_sigs['cam.acquire_time'] = acqtime
        d.stage_sigs['cam.acquire_period'] = acqtime + 0.100
        d.stage_sigs['cam.image_mode'] = 1
        d.stage_sigs['cam.num_images'] = N
        d.stage_sigs['total_points'] = N_tot
        d.stage_sigs['hdf5.num_capture'] = N_tot


    # Collect dark frame
    if (dark_frame):
        # Check shutter
        if (shutter):
           yield from bps.mov(shut_b, 'Close')

        # Trigger detector
        yield from count(xrd_det, num=N)
        # yield from count(xrd_det)
        update_scan_id()

        # Write to logfile
        # logscan('xrd_count')

    if (empty_pos != []):
        # Move into position
        yield from bps.mov(hf_stage.x, empty_pos[0])
        # yield from bps.mov(hf_stage.topx, empty_pos[0])
        yield from bps.mov(hf_stage.y, empty_pos[1])
        if (len(empty_pos) == 3):
            yield from bps.mov(hf_stage.z, empty_pos[2])

        # Check shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Open')

        # Trigger the detector
        yield from count(xrd_det, num=N)
        # yield from count(xrd_det)
        # yield from bps.trigger_and_read(xrd_det)
        update_scan_id()

        # Close shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Close')

        # Write to logfile
        # logscan('xrd_count')

    # Loop through positions
    if (pos == []):
        pos = [[hf_stage.x.position, hf_stage.y.position]]
    for i in range(len(pos)):
        i = int(i)
        # Move into position
        yield from bps.mov(hf_stage.x, pos[i][0])
        # yield from bps.mov(hf_stage.topx, pos[i][0])
        yield from bps.mov(hf_stage.y, pos[i][1])
        if (len(pos[i]) == 3):
            yield from bps.mov(hf_stage.z, pos[i][2])

        # Check shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Open')

        # Trigger the detector
        # Keep getting TimeoutErrors setting capture complete to False
        # This is a lot of duct tape, but hopefully this will let the scans
        # work instead of timing out
        need_data = True
        num_tries = 0
        while (need_data):
            try:
                num_tries = num_tries + 1
                yield from count(xrd_det, num=N)
                # yield from count(xrd_det)
                # yield from bps.trigger_and_read(xrd_det)
                need_data = False
                update_scan_id()
            except TimeoutError:
                dexela.unstage()
                if (num_tries >= 5):
                    print('Timeout has occured 5 times.')
                    raise
                print('TimeoutError: Trying again...')
                yield from bps.sleep(5)
            except:
                raise


        # Close shutter
        if (shutter):
            yield from bps.mov(shut_b, 'Close')

        # Write to logfile
        # logscan('xrd_count')


def collect_xrd_map(xstart, xstop, xnum,
                    ystart, ystop, ynum, acqtime=1,
                    dark_frame=False, shutter=True):

    # Scan parameters
    if (acqtime < 0.0066392):
        print('The detector should not operate faster than 7 ms.')
        print('Changing the scan dwell time to 7 ms.')
        acqtime = 0.007

    N = xnum * ynum

    # Setup detector
    xrd_det = [dexela]
    for d in xrd_det:
        d.stage_sigs['cam.acquire_time'] = acqtime
        d.stage_sigs['cam.acquire_period'] = acqtime + 0.100
        d.stage_sigs['cam.image_mode'] = 1
        d.stage_sigs['cam.num_images'] = 1
        d.stage_sigs['total_points'] = N
        d.stage_sigs['hdf5.num_capture'] = N

    # Collect dark frame
    if (dark_frame):
        # Check shutter
        if (shutter):
           yield from bps.mov(shut_b, 'Close')

        # Trigger detector
        yield from count(xrd_det, num=N)
        update_scan_id()

        # Write to logfile
        # logscan('xrd_count')

    if shutter:
        yield from mv(shut_b, 'Open')

    #scan_dets = xrd_det.append(sclr1)
    #print(xrd_det)
    # yield from outer_product_scan(scan_dets,
    #                               hf_stage.x, xstart, xstop, xnum,
    #                               hf_stage.y, ystart, ystop, ynum, False)
    yield from grid_scan(xrd_det,
                         hf_stage.x, xstart, xstop, xnum,
                         hf_stage.y, ystart, ystop, ynum, False)

    if shutter:
        yield from mv(shut_b, 'Close')

    # Write to logfile
    # logscan('xrd_count')
    update_scan_id()


@parameter_annotation_decorator({
    "parameters": {
        "extra_dets": {"default": "['dexela']"},
    }
})
def xrd_fly(*args, extra_dets=[dexela], **kwargs):
    kwargs.setdefault('xmotor', nano_stage.sx)
    kwargs.setdefault('ymotor', nano_stage.sy)
    kwargs.setdefault('flying_zebra', nano_flying_zebra)

    yield from abs_set(kwargs['flying_zebra'].fast_axis, 'NANOHOR', wait=True)
    yield from abs_set(kwargs['flying_zebra'].slow_axis, 'NANOVER')

    _xs = kwargs.pop('xs', xs)
    if extra_dets is None:
        extra_dets = []
    dets = [_xs] + extra_dets
    # To fly both xs and merlin
    # yield from scan_and_fly_base([_xs, merlin], *args, **kwargs)
    # To fly only xs
    yield from scan_and_fly_base(dets, *args, **kwargs)


def make_tiff(scanid, *, scantype='', fn=''):
    h = db[int(scanid)]
    if scanid == -1:
        scanid = int(h.start['scan_id'])

    if scantype == '':
        scantype = h.start['scan']['type']

    if ('FLY' in scantype):
        d = list(h.data('dexela_image', stream_name='stream0', fill=True))
        d = np.array(d)

        (row, col, imgY, imgX) = d.shape
        if (d.size == 0):
            print('Error collecting dexela data...')
            return
        d = np.reshape(d, (row*col, imgY, imgX))
    elif ('STEP' in scantype):
        d = list(h.data('dexela_image', fill=True))
        d = np.array(d)
        d = np.squeeze(d)
    # elif (scantype == 'count'):
    #     d = list(h.data('dexela_image', fill=True))
    #     d = np.array(d)
    else:
        print('I don\'t know what to do.')
        return

    if fn == '':
        fn = f"scan{scanid}_xrd.tiff"
    try:
        io.imsave(fn, d.astype('uint16'))
    except:
        print(f'Error writing file!')
