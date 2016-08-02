from ophyd import (EpicsSignal, EpicsSignalRO, EpicsMotor,
                   Device, Signal, PseudoPositioner, PseudoSingle)
from ophyd.utils.epics_pvs import set_and_wait
from ophyd.ophydobj import StatusBase, MoveStatus
from ophyd import Component as Cpt, Signal
import time as ttime
from cycler import cycler
from bluesky import Msg
from bluesky.plans import open_run, close_run, trigger_and_read

#from bluesky.plan_tools import trigger_and_read, wrap_with_decorator, run_wrapper

triggger_and_read = wrap_with_decorator = run_wrapper = None


class UVDoneMOVN(Signal):
    """Signal for use as done signal for use in individual mode undulator motors

    This is a soft-signal that monitors several real PVs to sort out when the
    positioner is done moving.

    If the positioner looks like it has stopped (ex, moving is 0) but the readback
    is not close enough to the target, then re-actuate the motor.

    Parameters
    ----------
    parent : Device
         This comes from Cpt magic

    moving : str
        Name of the 'moving' signal on parent
    readback : str
        Name of the 'readback' signal on the parent
    actuate : str
        Name of the 'actuate' signal on the parent
    stop : str
        Name of the stop signal on the parent

    kwargs : ??
        All passed through to base Signal

    Attributes
    ----------
    target : float or None
        Where the positioner is going.  If ``None``, the callbacks short-circuit

    """
    def __init__(self, parent, moving, readback, actuate='actuate',
                 stop='stop_signal',
                 **kwargs):
        super().__init__(parent=parent, value=1, **kwargs)
        self._rbv = readback
        self._brake = moving
        self._act = actuate
        self._stp = stop
        self.target = None
        self._next_reactuate_time = 0

    def put(self, *arg, **kwargs):
        raise TypeError("You con not tell an undulator motor it is done")

    def _put(self, *args, **kwargs):
        return super().put(*args, **kwargs)

    def _watcher(self, obj=None, **kwargs):
        '''The callback to install on readback and moving signals

        This callback watches if the position has gotten close enough to
        it's target _and_ has stopped moving, and then flips this signal to 1 (which
        in turn flips the Status object)

        '''
        target = self.target
        if target is None:
            return

        rbv = getattr(self.parent, self._rbv)
        moving = getattr(self.parent, self._brake)

        cur_value = rbv.get()
        not_moving = not moving.get()

        # come back and check this threshold value
        # this is 2 microns
        if not_moving and abs(target - cur_value) < 0.002:       
            self._put(1)
            self._remove_cbs()
            return

        # if it is not moving, but we are not where we want to be,
        # poke it again
        if not_moving:
            cur_time = ttime.time()
            if cur_time > self._next_reactuate_time:
                actuate = getattr(self.parent, self._act)
                print('re actuated', self.parent.name)
                actuate.put(1)
                self._next_reactuate_time = cur_time + 1

    def _stop_watcher(self, *arg, **kwargs):
        '''Call back to be installed on the stop signal

        if this gets flipped, clear all of the other callbacks and tell
        the status object that it is done.

        TODO: mark status object as failed
        TODO: only trigger this on 0 -> 1 transposition
        '''
        print('STOPPED')
        # set the target to None and remove all callbacks
        self.reset(None)
        # flip this signal to 1 to signal it is done
        self._put(1)
        # push stop again 'just to be safe'
        # this is paranoia related to the re-kicking the motor is the
        # other callback
        stop = getattr(self.parent, self._stp)
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
    '''Base class for undulator motors

    - patches up behavior of actuate
    - installs done callbacks in proper places.  Assumes the above Done signal
      is used
    '''
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
    '''helper function to watch status objects

    Prints to screen every 0.5s
    '''
    while not st.done:
        print(st, st.done)
        time.sleep(.1)
    print(st)


class UndlatorMotorUSU(UndulatorPositioner):
    'Upstream upper motor on SRX undulator'
    readback = Cpt(EpicsSignal, '}REAL_POSITION_US_UPPER')
    setpoint = Cpt(EpicsSignal, '-Mtr:6}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:6}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:6}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:3}Pos.MOVN')


class UndlatorMotorUSL(UndulatorPositioner):
    'Upstream lower motor on SRX undulator'
    readback = Cpt(EpicsSignal, '}REAL_POSITION_US_LOWER')
    setpoint = Cpt(EpicsSignal, '-Mtr:8}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:8}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:8}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:8}Pos.MOVN')


class UndlatorMotorDSU(UndulatorPositioner):
    'Downstream upper motor on SRX undulator'
    readback = Cpt(EpicsSignal, '}REAL_POSITION_DS_UPPER')
    setpoint = Cpt(EpicsSignal, '-Mtr:5}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:5}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:5}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:5}Pos.MOVN')


class UndlatorMotorDSL(UndulatorPositioner):
    'Downstream lower motor on SRX undulator'
    readback = Cpt(EpicsSignal, '}REAL_POSITION_DS_LOWER')
    setpoint = Cpt(EpicsSignal, '-Mtr:7}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:7}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    stop_signal = Cpt(EpicsSignal, '-Mtr:7}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:4}Pos.MOVN')


class UndlatorMotorElevation(UndulatorPositioner):
    readback = Cpt(EpicsSignal, '}REAL_ELEVATION_US')
    setpoint = Cpt(EpicsSignal, '-Mtr:1}Inp:Pos')
    actuate = Cpt(EpicsSignal, '-Mtr:1}Sw:Go')
    done = Cpt(UVDoneMOVN, None, moving='moving',
               readback='readback', add_prefix=())
    #### CHECK THE STOP SIGNAL PV
    stop_signal = Cpt(EpicsSignal, '-Mtr:1}Pos.STOP')

    moving = Cpt(EpicsSignal, '-Mtr:1}Pos.MOVN')

    ds_elevation = Cpt(EpicsSignal, '}REAL_ELEVATION_DS')
    avg_elevation = Cpt(EpicsSignal, '}REAL_ELEVATION_AVG')


class PowerUndulator(Device):
    'Simple aggregate device to hold undulator motors'
    us_lower = Cpt(UndlatorMotorUSL, '', read_attrs=['readback', 'setpoint', 'moving'])
    us_upper = Cpt(UndlatorMotorUSU, '', read_attrs=['readback', 'setpoint', 'moving'])
    ds_lower = Cpt(UndlatorMotorDSL, '', read_attrs=['readback', 'setpoint', 'moving'])
    ds_upper = Cpt(UndlatorMotorDSU, '', read_attrs=['readback', 'setpoint', 'moving'])
    elevation = Cpt(UndlatorMotorElevation, '', read_attrs=['readback', 'setpoint', 'moving'])

pu = PowerUndulator('SR:C5-ID:G1{IVU21:1', name='pu')


class UTemperatures(Device):
    'Undulator temperatures'
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

TILT_LIMIT = 0.090  # 99 microns
CRAB_LIMIT = 0.050  # 50 microns
TARGET_THRESH = 0.002 # 2 microns,


def ud_crab_plan(pu, us_u, us_l, ds_u, ds_l, other_dets=None):
    '''A generator plan for crabbing the undulator to new position

    This is a single-use plan for moving the undulator to a new position ::

       plan = ud_crab_plan(pu, a, 6.46, 6.46, 6.46, [ut])
       gs.RE(plan)

    This plan round-robin moves the motors to the desired positions taking
    readings of all the motor positions and any additional detectors and ~1Hz

    Parameters
    ----------
    pu : PowerUndulator
        Bucket of undulator motors

    us_u : float
        The target upstream upper motor position

    us_l : float
        The target upstream lower motor position

    ds_u : float
        The target downstream upper motor position

    ds_l : float
        The target downstream lower motor position

    other_dets : list, optional
        List of other detectors to read
    '''
    pu.stop()
    if other_dets is None:
        other_dets = []
    # magic goes here
    #if abs(us_u - ds_u) > CRAB_LIMIT:
    if abs(us_u - ds_u) > TILT_LIMIT:
        raise ValueError("exceded tilt limit on upper |{} - {}| > {}".format(
                us_u,  ds_u, TILT_LIMIT))

    #if abs(us_l - ds_l) > CRAB_LIMIT:
    if abs(us_l - ds_l) > TILT_LIMIT:
        raise ValueError("exceded tilt limit on lower |{} - {}| > {}".format(
                us_l,  ds_l, TILT_LIMIT))

    def limit_position(pos, pair_pos, target, pair_target):
        if abs(pair_pos - pair_target) < TARGET_THRESH:
            # on final step
            limit = TILT_LIMIT
        else:
            limit = CRAB_LIMIT
        # moving out
        if target > pos:
            return min(target, pair_pos + limit)
        else:
            return max(target, pair_pos - limit)

    def traj(pu):
        while True:
            done_count = 0
            # MOVE THE UPSTREAM UPPER
            cur_usu = pu.us_upper.position
            cur_dsu = pu.ds_upper.position
            if abs(cur_usu - us_u) > TARGET_THRESH:
                target = limit_position(cur_usu, cur_dsu, us_u, ds_u)
                yield pu.us_upper, target
            else:
                done_count += 1

            # MOVE THE DOWNSTREAM UPPER
            cur_usu = pu.us_upper.position
            cur_dsu = pu.ds_upper.position
            if abs(cur_dsu - ds_u) > TARGET_THRESH:
                target = limit_position(cur_dsu, cur_usu, ds_u, us_u)
                yield pu.ds_upper, target
            else:
                done_count += 1

            # MOVE THE UPSTREAM lower
            cur_usl = pu.us_lower.position
            cur_dsl = pu.ds_lower.position
            if abs(cur_usl - us_l) > TARGET_THRESH:
                target = limit_position(cur_usl, cur_dsl, us_l, ds_l)
                yield pu.us_lower, target
            else:
                done_count += 1

            # MOVE THE DOWNSTREAM lower
            cur_usl = pu.us_lower.position
            cur_dsl = pu.ds_lower.position
            if abs(cur_dsl - ds_l) > TARGET_THRESH:
                target = limit_position(cur_dsl, cur_usl, ds_l, us_l)
                yield pu.ds_lower, target
            else:
                done_count += 1

            if done_count == 4:
                return

    yield from open_run()
    yield from trigger_and_read([pu] + other_dets)
    for mot, target in traj(pu):
        print("About to move {} to {}".format(mot.name, target))
        # yield Msg('checkpoint', None)
        # yield Msg('pause', None)
        # yield Msg('clear_checkpoint', None)
        st = yield Msg('set', mot, target, timeout=None)
        # move the motor
        # speed is mm / s measured on us lower 2016-06-02
        # timeout is 3 * max_crab / speed
        fail_time = ttime.time() + (TILT_LIMIT / .0003) * 4
        while not st.done:
            yield from trigger_and_read([pu] + other_dets)
            if ttime.time() > fail_time:
                mot.stop()
                raise RuntimeError("Undulator move timed out")
            yield Msg('checkpoint')
            yield Msg('sleep', None, 1)

        if st.error > .002:
            raise RuntimeError("only got with in {} of target {}".
                               format(st.error, st.target))
        yield Msg('checkpoint')
        for j in range(2):
            yield Msg('sleep', None, 1)
            yield from trigger_and_read([pu] + other_dets)
        yield Msg('checkpoint')
    yield from close_run()


def play():
    '''Example of how to make a composite 'master' plan
    '''
    for a in [6.46, 6.47, 6.48]:
        yield from ud_crab_plan(pu, a, 6.46, 6.46, 6.46, [ut])
        yield from EnergyPlan()

name_cycle = (cycler('d', ['pu']) * cycler('end', ['us', 'ds']) * 
              cycler('side', ['upper', 'lower']) * 
              cycler('read', ['readback', 'moving']))
lt = LiveTable(['{d}_{end}_{side}_{read}'.format(**_p) for _p in name_cycle], 
                print_header_interval=15)

# gs.RE(play())
#
