
def xs2_1d_scan(stage, x0, x1, nx, dwell, dets=[xs2, sclr1], shutter=True):
    yield from check_shutters(shutter, 'Open')
    yield from mov(xs2.external_trig, False)
    yield from mov(xs2.cam.acquire_time, dwell)
    yield from mov(xs2.total_points, nx)
    yield from mov(xs2.cam.num_images, nx)
    yield from mov(sclr1.preset_time, dwell)
    yield from subs_wrapper(
        scan(dets, stage, x0, x1, nx),
        {
            "all": [
                LivePlot(
                    "xs2_channels_channel01_mcarois_mcaroi01_total_rbv",
                    x=stage.name,
                )
            ]
        },
    )
    yield from check_shutters(shutter, 'Close')

def xs2_1d_relscan(stage, neg_dx, pos_dx, nx, dwell, dets=[xs2, sclr1], shutter=True):
    yield from check_shutters(shutter, 'Open')
    yield from mov(xs2.external_trig, False)
    yield from mov(xs2.cam.acquire_time, dwell)
    yield from mov(xs2.total_points, nx)
    yield from mov(xs2.cam.num_images, nx)
    yield from mov(sclr1.preset_time, dwell)
    yield from subs_wrapper(
        rel_scan(dets, stage, neg_dx, pos_dx, nx),
        {
            "all": [
                LivePlot(
                    "xs2_channels_channel01_mcarois_mcaroi01_total_rbv",
                    x=stage.name,
                )
            ]
        },
    )
    yield from check_shutters(shutter, 'Close')

def overnight():
    x0 = 4.2
    x1 = 4.2
    nx = 1

    x = np.linspace(x0, x1, num=nx)
    for xi in x:
        yield from mov(confocal_stage.x, xi)
        yield from xs2_1d_relscan(confocal_stage.y, -0.5, 0.5, 101, 1)
        yield from xs2_1d_relscan(confocal_stage.z, -0.5, 0.5, 101, 1)

