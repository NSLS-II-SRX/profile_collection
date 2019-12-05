from bluesky.suspenders import SuspendFloor, SuspendCeil, SuspendBoolHigh, SuspendBoolLow
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp


def shuttergenerator(shutter, value):
    return (yield from bpp.rewindable_wrapper(bps.mv(shutter, value), False))


### Ring current suspender
susp_rc = SuspendFloor(ring_current, 140, resume_thresh=160, sleep=10*60,
                       pre_plan=list(shuttergenerator(shut_b, 'Close')),
                       post_plan=list(shuttergenerator(shut_b, 'Open')))


### Cryo cooler suspender
susp_cryo = SuspendCeil(cryo_v19, 0.8, resume_thresh=0.2, sleep=15*60,
                        pre_plan=list(shuttergenerator(shut_b, 'Close')),
                        post_plan=list(shuttergenerator(shut_b, 'Open')))


### Shutter status suspender
susp_shut_fe = SuspendBoolHigh(EpicsSignalRO(shut_fe.status.pvname), sleep=10)
susp_shut_a = SuspendBoolHigh(EpicsSignalRO(shut_a.status.pvname), sleep=10)
susp_shut_b = SuspendBoolHigh(EpicsSignalRO(shut_b.status.pvname), sleep=10)


### HDCM bragg temperature suspender
susp_dcm_bragg_temp = SuspendCeil(dcm.temp_pitch, 120, resume_thresh=118, sleep = 1)


### Install suspenders 
RE.install_suspender(susp_rc)
RE.install_suspender(susp_shut_fe)
RE.install_suspender(susp_dcm_bragg_temp)

