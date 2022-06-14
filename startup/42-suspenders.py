print(f'Loading {__file__}...')

from bluesky.suspenders import (SuspendFloor, SuspendCeil,
                                SuspendBoolHigh, SuspendBoolLow)
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from ophyd.sim import FakeEpicsSignal


def shuttergenerator(shutter, value):
    return (yield from bpp.rewindable_wrapper(bps.mv(shutter, value), False))
    # return (yield from bpp.rewindable_wrapper(bps.mv(shutter, value), True))

def another_generator(pv, value):
    return (yield from bps.mov(pv, value))


# Ring current suspender
dummy_rc = EpicsSignalRO("XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_WID:RBV", name="dummy_rc")
susp_rc = SuspendFloor(dummy_rc, 140, resume_thresh=160, sleep=10,
                       pre_plan=list(shuttergenerator(shut_b, 'Close')),
                       post_plan=list(shuttergenerator(shut_b, 'Open')))
# susp_rc = SuspendFloor(ring_current, 140, resume_thresh=160, sleep=10*60,
#                        pre_plan=list(shuttergenerator(shut_b, 'Close')),
#                        post_plan=list(shuttergenerator(shut_b, 'Open')))


# Cryo cooler suspender
susp_cryo = SuspendCeil(cryo_v19, 0.8, resume_thresh=0.2, sleep=15*60,
                        pre_plan=list(shuttergenerator(shut_b, 'Close')),
                        post_plan=list(shuttergenerator(shut_b, 'Open')))

# Testing suspenders using Filter box
# susp_shut_testing = SuspendBoolHigh(EpicsSignalRO(attenuators.Fe_shutter.pvname), sleep=5,
#                               pre_plan=list(shuttergenerator(shut_d, 1)),
#                               post_plan=list(shuttergenerator(shut_d, 0)))


# Shutter status suspender
susp_shut_fe = SuspendBoolHigh(EpicsSignalRO(shut_fe.status.pvname), sleep=10)
susp_shut_a = SuspendBoolHigh(EpicsSignalRO(shut_a.status.pvname), sleep=10)
susp_shut_b = SuspendBoolHigh(EpicsSignalRO(shut_b.status.pvname), sleep=10)


# HDCM bragg temperature suspender
susp_dcm_bragg_temp = SuspendCeil(dcm.temp_pitch, 120, resume_thresh=118, sleep=1)

# Fly-scan retry mechanism
failed_row = EpicsSignalRO('XF:05IDD-ES:1{Dev:Zebra1}:PC_GATE_OUT', name='failed_row')
susp_failed_row = SuspendCeil(failed_row, 0.9, resume_thresh=0.5, sleep=1)

# Install suspenders
RE.install_suspender(susp_rc)
#RE.install_suspender(susp_shut_fe)
RE.install_suspender(susp_dcm_bragg_temp)
# RE.install_suspender(susp_shut_testing)
# RE.install_suspender(susp_failed_row)
