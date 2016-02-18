from ophyd import (EpicsSignal, EpicsSignalRO, EpicsMotor,
                   Device, Signal, PseudoPositioner, PseudoSingle)
from ophyd.utils.epics_pvs import set_and_wait
from ophyd.ophydobj import StatusBase, MoveStatus
from ophyd import Component as Cpt

from bluesky import Msg
from bluesky.plan_tools import trigger_and_read, wrap_with_decorator, run_wrapper

class UVDoneMOVN(PermissiveGetSignal):
    def __init__(self, parent, moving, readback, actuate='actuate', stop='stop_signal',
                 **kwargs):
        super().__init__(parent=parent, value=1, **kwargs)
        self._rbv = readback
        self._brake = moving
        self._act = actuate
        self._stp = stop
        self.target = None

    def put(self, *arg, **kwargs):
        raise TypeError("You con not tell an undulator motor it is done")

    def _put(self, *args, **kwargs):
        return super().put(*args, **kwargs)

    def _watcher(self, obj=None, **kwargs):
        target = self.target
        if target is None:
            return

        rbv = getattr(self.parent, self._rbv)
        stop = getattr(self.parent, self._stp)
        moving = getattr(self.parent, self._brake)

        cur_value = rbv.get()
        not_moving = not moving.get()

        # come back and check this threshold value
        if not_moving and abs(target - cur_value) < 0.002:
            self._put(1)
            self._remove_cbs()
            return

        # if it is not moving, but we are not where we want to be, 
        # poke it again
        if not_moving:
            actuate = getattr(self.parent, self._act)

            print('re actuated')
            actuate.put(1)

    def _stop_watcher(self, *arg, **kwargs):
        print('STOPPED')
        self.reset(None)
        stop = getattr(self.parent, self._stp)
        self._put(1)
        stop.put(1)

    def reset(self, target):
        self.target = target
        self._put(0)
        self._remove_cbs()

    def _remove_cbs(self):
        rbv = getattr(self.parent, self._rbv)
        stop = getattr(self.parent, self._stp)
        moving = getattr(self.parent, self._brake)

        rbv.clear_sub(self._watcher)
        moving.clear_sub(self._watcher)
        stop.clear_sub(self._stop_watcher)

class UndulatorPositioner(PVPositioner):
    def _move_async(self, position, **kwargs):
        '''Move and do not wait until motion is complete (asynchronous)'''
        if self.actuate is not None:
            set_and_wait(self.setpoint, position)
            self.actuate.put(self.actuate_value, wait=False)
        else:
            self.setpoint.put(position, wait=False)
    
    def move(self, v, *args, **kwargs):
        self.done.reset(v)
        ret = super().move(v, *args, **kwargs)
        self.moving.subscribe(self.done._watcher,
                                event_type=self.moving.SUB_VALUE)
        self.readback.subscribe(self.done._watcher,
                                event_type=self.readback.SUB_VALUE)

        self.stop_signal.subscribe(self.done._stop_watcher,
                                event_type=self.stop_signal.SUB_VALUE, run=False)

        return ret

    def stop(self):
        self.done.reset(None)
        super().stop()

def st_watcher(st):
    while not st.done:
        print(st)
        time.sleep(.5)
    print(st)


class UndlatorMotorUSU(UndulatorPositioner):
    readback = Cpt(EpicsSignal, '}REAL_POSITION_US_UPPER') 
    setpoint = Cpt(EpicsSignal, '-Mtr:6}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:6}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:6}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:3}Pos.MOVN')


class UndlatorMotorUSL(UndulatorPositioner):
    readback = Cpt(EpicsSignal, '}REAL_POSITION_US_LOWER')
    setpoint = Cpt(EpicsSignal, '-Mtr:8}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:8}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:8}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:8}Pos.MOVN')


class UndlatorMotorDSU(UndulatorPositioner):
    readback = Cpt(EpicsSignal, '}REAL_POSITION_DS_UPPER')
    setpoint = Cpt(EpicsSignal, '-Mtr:5}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:5}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:5}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:5}Pos.MOVN')


class UndlatorMotorDSL(UndulatorPositioner):
    readback = Cpt(EpicsSignal, '}REAL_POSITION_DS_LOWER')
    setpoint = Cpt(EpicsSignal, '-Mtr:7}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:7}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:7}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:4}Pos.MOVN')


class PowerUndulator(Device):
    us_lower = Cpt(UndlatorMotorUSL, '')
    us_upper = Cpt(UndlatorMotorUSU, '')
    ds_lower = Cpt(UndlatorMotorDSL, '')
    ds_upper = Cpt(UndlatorMotorDSU, '')

pu = PowerUndulator('SR:C5-ID:G1{IVU21:1', name='pu')

class UTemperatures(Device):
    T1 = Cpt(EpicsSignal, '-Pt:1}T')
    T2 = Cpt(EpicsSignal, '-Pt:2}T')
    T3 = Cpt(EpicsSignal, '-Pt:3}T')
    T4 = Cpt(EpicsSignal, '-Pt:4}T')
    T5 = Cpt(EpicsSignal, '-Pt:5}T')
    T6 = Cpt(EpicsSignal, '-Pt:6}T')
    T7 = Cpt(EpicsSignal, '-Pt:7}T')
    T8 = Cpt(EpicsSignal, '-Pt:8}T')
    T9 = Cpt(EpicsSignal, '-Pt:9}T')
    T10 = Cpt(EpicsSignal, '-Pt:10}T')
    T11 = Cpt(EpicsSignal, '-Pt:11}T')
    T12 = Cpt(EpicsSignal, '-Pt:12}T')
    T13 = Cpt(EpicsSignal, '-Pt:13}T')
    T14 = Cpt(EpicsSignal, '-Pt:14}T')
    T15 = Cpt(EpicsSignal, '-Pt:15}T')
    T16 = Cpt(EpicsSignal, '-Pt:16}T')

ut = UTemperatures('SR:C5-ID:G1{IVU21:1')

CRAB_LIMIT = 0.050  # 50 microns
TARGET_THRESH = 0.002 # 2 microns, 

@wrap_with_decorator(run_wrapper)
def ud_crab_plan(pu, us_u, us_l, ds_u, ds_l, other_dets):
    # magic goes here
    if abs(us_u - ds_u) > CRAB_LIMIT:
        raise ValueError("exceded crab limit on upper")

    if abs(us_l - ds_l) > CRAB_LIMIT:
        raise ValueError("exceded crab limit on lower")

    def traj(pu):
        while True:
            done_count = 0
            # MOVE THE UPSTREAM UPPER
            cur_usu = pu.us_upper.position
            cur_dsu = pu.ds_upper.position
            if abs(cur_usu - us_u) > TARGET_THRESH:
                # moving out
                if us_u > cur_usu:
                    target = min(us_u, cur_dsu + CRAB_LIMIT)
                else:
                    target = max(us_u, cur_dsu - CRAB_LIMIT)
                yield pu.us_upper, target
            else:
                done_count += 1
    
            # MOVE THE DOWNSTREAM UPPER
            cur_usu = pu.us_upper.position
            cur_dsu = pu.ds_upper.position
            if abs(cur_dsu - ds_u) > TARGET_THRESH:
                # moving out
                if ds_u > cur_dsu:
                    target = min(ds_u, cur_usu + CRAB_LIMIT)
                else:
                    target = max(ds_u, cur_usu - CRAB_LIMIT)
                yield pu.ds_upper, target
            else:
                done_count += 1
            
            # MOVE THE UPSTREAM lower
            cur_usl = pu.us_lower.position
            cur_dsl = pu.ds_lower.position
            if abs(cur_usl - us_l) > TARGET_THRESH:
                # moving out
                if us_l > cur_usl:
                    target = min(us_l, cur_dsl + CRAB_LIMIT)
                else:
                    target = max(us_l, cur_dsl - CRAB_LIMIT)
                yield pu.us_lower, target
            else:
                done_count += 1

            # MOVE THE DOWNSTREAM lower
            cur_usl = pu.us_lower.position
            cur_dsl = pu.ds_lower.position
            if abs(cur_dsl - ds_l) > TARGET_THRESH:
                # moving out
                if ds_l > cur_dsl:
                    target = min(ds_l, cur_usl + CRAB_LIMIT)
                else:
                    target = max(ds_l, cur_usl - CRAB_LIMIT)
                yield pu.ds_lower, target
            else:
                done_count += 1
                
            if done_count == 4:
                return

    for mot, target in traj(pu):
        print("About to move {} to {}".format(mot.name, target))
        # yield Msg('checkpoint', None)
        # yield Msg('pause', None)
        # yield Msg('clear_checkpoint', None)
        st = yield Msg('set', mot, target)
        # move the motor
        fail_time = ttime.time() + 40 * 5
        while not st.done:
            yield from trigger_and_read([pu] + other_dets)
            if ttime.time() > fail_time:
                mot.stop()
                raise RuntimeError("Undulator move timed out")
            yield Msg('sleep', None, 1)
            

        if st.error > .002:
            raise RuntimeError("only got with in {} of target {}".
                               format(st.error, st.target))

        for j in range(2):
            yield Msg('sleep', None, 1)
            yield from trigger_and_read([pu] + other_dets)

def play():
    for a in [6.46, 6.47, 6.48]:
        yield from ud_crab_plan(pu, a, 6.46, 6.46, 6.46, [ut])
        yield from EnergyPlan()


# gs.RE(play())
# gs.RE(play())
# gs.RE(play())
# gs.RE(play())
