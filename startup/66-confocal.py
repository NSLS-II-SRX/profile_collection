
def xs2_1d_scan(stage, x0, x1, nx, dwell, dets=[xs2, sclr1]):
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

def xs2_1d_relscan(stage, neg_dx, pos_dx, nx, dwell, dets=[xs2, sclr1]):
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

