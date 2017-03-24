from ophyd import Device, Component as Cpt
from ophyd.signal import Signal
import bluesky.plans as bp
import numpy as np
from bluesky.examples import Mover, NullStatus
from collections import OrderedDict
import time as ttime


x = Mover('x', OrderedDict([('x', lambda x: x),
                            ('x_setpoint', lambda x: x)]),
          {'x': 0}, conf={'velocity': 3})

y = Mover('y', OrderedDict([('y', lambda y: y),
                            ('y_setpoint', lambda y: y)]),
          {'y': 0}, conf={'velocity': 3})

shutter = Mover('shutter', {'shutter': lambda shutter: shutter},
                {'shutter': 'closed'}, conf={'velocity': 3})


class EncoderAxis(Device):
    pos = Cpt(Signal)


class Encoder(Device):
    x = Cpt(EncoderAxis, 'x')
    y = Cpt(EncoderAxis, 'x')


class MAIA(Device):
    x_motor = None  # EpicsMotor
    y_motor = None  # EpicsMotor
    blog = None  # metadata log
    encoder = Cpt(Encoder, '')  # complex object, values copied from motors
    pixel = None  # pixel shape, pitch info, origin
    scan = None  # metadata about scan (dwell time, scan order)
    pixel_enable = None   # bool
    pixel_event_enable = None  # bool

    def __init__(self, *args, conf, **kwargs):
        super().__init__(*args, **kwargs)
        self._conf = conf

    def configure(self, d):
        old = dict(self._conf)
        self._conf.update(d)
        return old, dict(self._conf)

    def read_configuration(self):
        return {k: {'value': v, 'timestamp': ttime.time()}
                for k, v in self._conf.items()}

    @property
    def conf_attrs(self):
        return list(self._conf_state)

    def stage(self):
        ...
        # copy info from motors -> encoder
        # set pixel information
        # set scan information

    def unstage(self):
        ...

    def kickoff(self):
        ...
        # enable blogging
        # enable pixels
        # enable pixel event
        # blog.group.next
        # set blog_enabled
        return NullStatus()

    def complete(self):
        return NullStatus()

    def collect(self):
        return []

    def describe_collect(self):
        return {}


maia = MAIA('srx:maia', name='maia', conf={})


def first_key_plan(pos):
    ret = yield from bp.read(pos)
    if ret is not None:
        return next(iter(ret.values()))['value']
    return None


def fly_maia(xstart, xstop, xnum,
             ystart, ystop, ynum, *,
             md=None):
    if md is None:
        md = {}
    x_pitch = abs(xstop - xstart) / xnum
    y_pitch = abs(ystop - ystart) / ynum

    # TODO compute this based on someting
    ssr_x = 5
    ssr_y = 5

    yield from bp.abs_set(x, xstart, group='maia_sync')
    yield from bp.abs_set(y, ystart, group='maia_sync')

    yield from bp.wait(group='maia_sync')

    maia_old_c, mia_new_c = yield from bp.configure(
        maia, {'x.pitch': x_pitch, 'y.pitch': y_pitch})

    # make sure the maia agrees with where the motors are
    # this is important because it works off of dead-reckoning to know
    # where it is (?)  this is not configuration

    x_val = yield from first_key_plan(x)
    y_val = yield from first_key_plan(x)
    # TODO, depends on actual device
    yield from bp.mv(maia.encoder.x.pos, x_val)
    yield from bp.mv(maia.encoder.y.pos, y_val)

    # set the motors to the right speed
    # TODO reset these
    old_x_c, _ = yield from bp.configure(x, {'velocity': ssr_x})
    old_y_c, _ = yield from bp.configure(y, {'velocity': ssr_y})

    yield from bp.mv(shutter, 'open')
    yield from bp.open_run(md)
    yield from bp.stage(maia)
    yield from bp.kickoff(maia)
    # by row
    for i, x_pos in enumerate(np.linspace(xstart, xstop, xnum)):
        # move to the row we want
        yield from bp.mv(x, x_pos)
        if i % 2:
            # for odd-rows move from start to stop
            yield from bp.mv(y, ystop)
        else:
            # for even-rows move from stop to start
            yield from bp.mv(y, ystart)
    # stop the maia ("I'll wait until your done")
    yield from bp.complete(maia)
    # shut the stutter
    yield from bp.mv(shutter, 'close')
    # collect data from maia
    yield from bp.collect(maia)
    yield from bp.close_run()

    yield from bp.configure(x, old_x_c)
    yield from bp.configure(y, old_y_c)
    yield from bp.configure(maia, maia_old_c)
