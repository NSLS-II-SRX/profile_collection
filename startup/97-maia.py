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
        self.pixel_enable_cmd.value.put(1)
        self.pixel_event_enable_cmd.value.put(1)
        self.photon_enable_sp.value.put(1)
        
        st = DeviceStatus(self)
        def _cb_discard(value, **kwargs):
            if value == 0:
                st._finished()
                self.blog_discard_mon.value.clear_sub(_cb_discard)
                
        self.blog_discard_mon.value.subscribe(_cb_discard, run=False)    
        self.newrun_cmd.value.put('1')            
        return st

    def complete(self):
        st = DeviceStatus(self)
        def _cb_discard(value, **kwargs):
            if value == 1:
                st._finished()
                self.blog_discard_mon.value.clear_sub(_cb_discard)
                
        self.blog_discard_mon.value.subscribe(_cb_discard, run=False)    

        self.endrun_cmd.value.put('1')
        return st

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


maia = MAIA('SRX:MAIA', name='maia')


def first_key_plan(pos):
    ret = yield from bp.read(pos)
    if ret is not None:
        return next(iter(ret.values()))['value']
    return None

sample_md = {'sample': {'name': 'Ni mesh', 'owner': 'stolen'}}

def fly_maia(ystart, ystop, ynum,
             xstart, xstop, xnum,
             dwell, *, group=None,
             md=None):
    '''Run a flyscan with the maia


    Parameters
    ----------
    ystart, ystop : float
        The limits of the scan along the slow direction in absolute mm.

    ynum : int
        The number of pixels (rows) along the slow direction.

    xstart, xstop : float
        The limits of the scan along the fast direction in absolute mm.

    xnum : int
        The number of pixels (columns) along the fast direction.

    dwell : float
        The dwelll time in s.  This is used to set the motor velocity.
       
    group : str, optional
        The file group.  This goes into the file path that maia writes to.

    md : dict, optional
        Metadata to put into the start document.  

        If there is a 'sample' key, then it must be a dictionary and the
        keys

           ['info', 'name', 'owner', 'serial', 'type']

        are passed through to the maia metadata.

        If there is a 'scan' key, then it must be a dictionary and the
        keys

             ['region', 'info', 'seq_num', 'seq_total']

        are passed through to maia metadata.
    '''
    shutter = shut_b
    md = md or {}
    _md = {'detectors': ['maia'], 'shape': [ynum, xnum],
           'motors': [m.name for m in [hf_stage.y, hf_stage.x]],
           'num_steps': xnum*ynum,
           'plan_args': dict(
               ystart=ystart, ystop=ystop, ynum=ynum, 
               xstart=xstart, xstop=xstop, xnum=xnum,
               dwell=dwell, group=repr(group), md=md),
           'extents': [[ystart, ystop], [xstart, xstop]],
           'snaking': [False, True],
           'plan_name': 'fly_maia'}
    _md.update(md)
    
    md = _md

    sample_md = md.get('sample', {})
    for k in ['info', 'name', 'owner', 'serial', 'type']:
        v = sample_md.get(k, '')
        sig = getattr(maia, 'meta_val_sample_{}_sp.value'.format(k))
        yield from bp.mv(sig, str(v))

    scan_md = md.get('scan', {})
    for k in ['region', 'info', 'seq_num', 'seq_total']:
        v = sample_md.get(k, '')
        sig = getattr(maia, 'meta_val_scan_{}_sp.value'.format(k))
        yield from bp.mv(sig, str(v))

    if group is not None:
        yield from bp.mv(maia.blog_group_next_sp.value, group)
        
    
    if xstart > xstop:
        xstop, xtart = xstart, xstop

    if ystart > ystop:
        ystop, ytart = ystart, ystop

    # Pitch must match what raster driver uses for pitch ...
    x_pitch = abs(xstop - xstart) / (xnum - 1)
    y_pitch = abs(ystop - ystart) / (ynum - 1)

    # TODO compute this based on someting
    spd_x = (x_pitch / dwell) 

    yield from bp.abs_set(hf_stage.x, xstart, group='maia_sync')
    yield from bp.abs_set(hf_stage.y, ystart, group='maia_sync')

    yield from bp.wait(group='maia_sync')

    # cf_ret = yield from bp.configure(
    #     maia, {'x.pitch': x_pitch, 'y.pitch': y_pitch})
    # if cf_ret is not None:
    #     maia_old_c, mia_new_c = cf_ret
    # else:
    #     maia_old_c, mia_new_c = {}, {}
    # 
    # make sure the maia agrees with where the motors are
    # this is important because it works off of dead-reckoning to know
    # where it is (?)  this is not configuration

    x_val = yield from first_key_plan(hf_stage.x)
    y_val = yield from first_key_plan(hf_stage.y)
    # TODO, depends on actual device
    yield from bp.mv(maia.enc_axis_0_pos_sp.value, x_val)
    yield from bp.mv(maia.enc_axis_1_pos_sp.value, y_val)

    yield from bp.mv(maia.x_pixel_dim_origin_sp.value, xstart)
    yield from bp.mv(maia.y_pixel_dim_origin_sp.value, ystart)
    
    yield from bp.mv(maia.x_pixel_dim_pitch_sp.value, x_pitch)
    yield from bp.mv(maia.y_pixel_dim_pitch_sp.value, y_pitch)
    
    yield from bp.mv(maia.x_pixel_dim_coord_extent_sp.value, xnum)
    yield from bp.mv(maia.y_pixel_dim_coord_extent_sp.value, ynum)
    yield from bp.mv(maia.scan_order_sp.value, '01')
    yield from bp.mv(maia.meta_val_scan_order_sp.value, '01')
    yield from bp.mv(maia.pixel_dwell.value, dwell)
    yield from bp.mv(maia.meta_val_scan_dwell.value, str(dwell))

    yield from bp.mv(maia.meta_val_beam_particle_sp.value, 'photon')
    
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

    eng_kev = yield from first_key_plan(energy.energy.readback)
    if eng_kev is not None:
        yield from bp.mv(maia.meta_val_beam_energy_sp.value,
                         '{:.2f}'.format(eng_kev*1000))
    
    yield from bp.mv(shutter, 'Open')
    start_uid = yield from bp.open_run(md)

    yield from bp.mv(maia.meta_val_scan_crossref_sp.value, start_uid)
    #long int here.  consequneces of changing?
#    yield from bp.mv(maia.scan_number_sp,start_uid)
    yield from bp.stage(maia)  # currently a no-op
    
    def _raster_plan():
        yield from bp.kickoff(maia, wait=True)
        yield from bp.checkpoint()
        # by row
        for i, y_pos in enumerate(np.linspace(ystart, ystop, ynum)):
            yield from bp.checkpoint()
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
        yield from bp.mv(maia.meta_val_scan_crossref_sp.value, '')
        for k in ['info', 'name', 'owner', 'serial', 'type']:
            sig = getattr(maia, 'meta_val_sample_{}_sp.value'.format(k))
            yield from bp.mv(sig, '')

        for k in ['region', 'info', 'seq_num', 'seq_total']:
            sig = getattr(maia, 'meta_val_scan_{}_sp.value'.format(k))
            yield from bp.mv(sig, '')
        yield from bp.mv(maia.meta_val_beam_energy_sp.value, '')
        yield from bp.mv(maia.meta_val_scan_dwell.value, '')
        yield from bp.mv(maia.meta_val_scan_order_sp.value, '')        
        # yield from bp.configure(hf_stage.x, old_x_c)
        # yield from bp.configure(maia, maia_old_c)

    return (yield from bp.finalize_wrapper(_raster_plan(),
                                           _cleanup_plan()))
    
