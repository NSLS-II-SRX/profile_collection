from ophyd import Device, Component as Cpt
from ophyd.signal import Signal
import bluesky.plans as bp
import numpy as np
from bluesky.examples import Mover, NullStatus
from collections import OrderedDict
import time as ttime



class EncoderAxis(Device):
    pos = Cpt(Signal)


class Encoder(Device):
    x = Cpt(EncoderAxis, 'x')
    y = Cpt(EncoderAxis, 'x')


class MAIA(Kandinskivars):
    fly_keys = ['blog.info.blogd_data_path',
                'blog.info.blogd_working_directory',
                'blog.info.run_number']
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def kickoff(self):
        # paranoia enableing
        self.maia.pixel_enable_cmd.set(True)
        self.maia.pixel_event_enable_cmd.set(True)
        self.photon_enable_sp.set(1)
        # this is the one that matters
        return self.blog.newrun.set(1)

    def complete(self):
        return self.blog.endrun.set(1)

    def collect(self):
        vals = OrderedDict()
        for k in self.fly_keys:
            vals.update(getattr(self, k).read())
        
        data = {k: v['value'] for k, v in vals.items()}
        ts = {k: v['timestamp'] for k, v in vals.items()}
        yield {'data': data, 'timestamps': ts,
               'time': ttime.time(), 'seq_num': 0}

    def describe_collect(self):
        descs = OrderedDict()
        for k in self.fly_keys:
            descs.update(getattr(self, k).describe())
            
        return {'primary': descs}


# maia = MAIA('SRX:MAIA', name='maia')


def first_key_plan(pos):
    ret = yield from bp.read(pos)
    if ret is not None:
        return next(iter(ret.values()))['value']
    return None


def fly_maia(xstart, xstop, xnum,
             ystart, ystop, ynum, 
             dwell, *,
             md=None):
    shutter = shut_fast
    if md is None:
        md = {}
        
    md = dict(md)
    
    md.update(xstart=xstart, xstop=xstop,
              xnum=xnum, ynum=ynum, dwell=dwell)
    
    if xstart > xstop:
        xstop, xtart = xstart, xstop

    if ystart > ystop:
        ystop, ytart = ystart, ystop
        
    x_pitch = abs(xstop - xstart) / xnum
    y_pitch = abs(ystop - ystart) / ynum

    # TODO compute this based on someting
    spd_x = (x_pitch / dwell) 

    yield from bp.abs_set(hf_stage.x, xstart, group='maia_sync')
    yield from bp.abs_set(hf_stage.y, ystart, group='maia_sync')

    yield from bp.wait(group='maia_sync')

    cf_ret = yield from bp.configure(
        maia, {'x.pitch': x_pitch, 'y.pitch': y_pitch})
    if cf_ret is not None:
        maia_old_c, mia_new_c = cf_ret
    else:
        maia_old_c, mia_new_c = {}, {}

    # make sure the maia agrees with where the motors are
    # this is important because it works off of dead-reckoning to know
    # where it is (?)  this is not configuration

    x_val = yield from first_key_plan(hf_stage.x)
    y_val = yield from first_key_plan(hf_stage.y)
    # TODO, depends on actual device
    yield from bp.mv(maia.enc_axis_0_pos_sp, x_val)
    yield from bp.mv(maia.enc_axis_1_pos_sp, y_val)

    yield from bp.mv(maia.x_pixel_dim_origin_sp, xstart)
    yield from bp.mv(maia.y_pixel_dim_origin_sp, ystart)
    
    yield from bp.mv(maia.x_pixel_dim_pitch_sp, x_pitch)
    yield from bp.mv(maia.y_pixel_dim_pitch_sp, y_pitch)
    
    yield from bp.mv(maia.x_pixel_dim_coord_extent_sp, xnum)
    yield from bp.mv(maia.y_pixel_dim_coord_extent_sp, ynum)
    yield from bp.mv(maia.scan_order_sp, '01')
    yield from bp.mv(maia.pixel_dwell, dwell)
    
    #    yield from bp.mv(maia.maia_scan_info
    #need something to generate a filename here.
    #    yield from bp.mv(maia.blog_group_next_sp,datafile))
    #start blog in kickoff?

    # set the motors to the right speed
    cf_ret = yield from bp.configure(hf_stage.x, {'velocity': spd_x})
    if cf_ret is not None:
        old_x_c = cf_ret[0]
    else:
        old_x_c = {}
    #old_y_c, _ = yield from bp.configure(y, {'velocity': ssr_y})

    yield from bp.mv(shutter, 'Open')
    # TODO shove this into the maia md
    maia_cur_run = yield from first_key_plan(maia.blog_runno_mon)
    if maia_cur_run is not None:
        maia_next_run = maia_next_run + 1
    else:
        maia_next_run = -1
    md['maia_runno'] = maia_next_run
    start_uid = yield from bp.open_run(md)
    #long int here.  consequneces of changing?
#    yield from bp.mv(maia.scan_number_sp,start_uid)
    yield from bp.stage(maia)  # currently a no-op
    
    def _raster_plan():
        yield from bp.kickoff(maia, wait=True)
        # by row
        for i, y_pos in enumerate(np.linspace(ystart, ystop, ynum)):
            # move to the row we want
            yield from bp.mv(hf_stage.y, y_pos)
            if i % 2:
                # for odd-rows move from start to stop
                yield from bp.mv(hf_stage.x, xstop)
            else:
                # for even-rows move from stop to start
                yield from bp.mv(hf_stage.x, xstart)
                
    def _cleanup_plan():
        # stop the maia ("I'll wait until you're done")
        yield from bp.complete(maia, wait=True)
        # shut the shutter
        yield from bp.mv(shutter, 'Close')
        # collect data from maia
        yield from bp.collect(maia)
    
        yield from bp.unstage(maia)
        yield from bp.close_run()

        yield from bp.configure(hf_stage.x, old_x_c)
        yield from bp.configure(maia, maia_old_c)

    
