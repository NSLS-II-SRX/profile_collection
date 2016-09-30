# -*- coding: utf-8 -*-
"""
set up for wire scan for HF mode

"""

from bluesky.plans import OuterProductAbsScanPlan
import bluesky.plans as bp
from bluesky.callbacks import LiveRaster
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import epics
import os
import numpy


#matplotlib.pyplot.ticklabel_format(style='plain')
def get_stock_md():
    md = {}
    md['beamline_status']  = {'energy':  energy.energy.position 
                                #'slt_wb': str(slt_wb.position),
                                #'slt_ssa': str(slt_ssa.position)
                                }
                                
    md['initial_sample_position'] = {'hf_stage_x': hf_stage.x.position,
                                       'hf_stage_y': hf_stage.z.position,
                                       'hf_stage_z': hf_stage.z.position}
    md['wb_slits'] = {'v_gap' : slt_wb.v_gap.position,
                            'h_gap' : slt_wb.h_gap.position,
                            'v_cen' : slt_wb.v_cen.position,
                            'h_cen' : slt_wb.h_cen.position
                            }
    md['hfm'] = {'y' : hfm.y.position,
                               'bend' : hfm.bend.position} 
    md['ssa_slits'] = {'v_gap' : slt_ssa.v_gap.position,
                            'h_gap' : slt_ssa.h_gap.position,
                            'v_cen' : slt_ssa.v_cen.position,
                            'h_cen' : slt_ssa.h_cen.position                                      
                             }                                      
    return md

def get_stock_md_xfm():
    md = {}
    md['beamline_status']  = {'energy':  energy.energy.position 
                                #'slt_wb': str(slt_wb.position),
                                #'slt_ssa': str(slt_ssa.position)
                                }
                                
    md['initial_sample_position'] = {'stage27a_x': stage.x.position,
                                       'stage27a_y': stage.y.position,
                                       'stage27a_z': stage.z.position}
    md['wb_slits'] = {'v_gap' : slt_wb.v_gap.position,
                            'h_gap' : slt_wb.h_gap.position,
                            'v_cen' : slt_wb.v_cen.position,
                            'h_cen' : slt_wb.h_cen.position
                            }
    md['hfm'] = {'y' : hfm.y.position,
                               'bend' : hfm.bend.position} 
    md['ssa_slits'] = {'v_gap' : slt_ssa.v_gap.position,
                            'h_gap' : slt_ssa.h_gap.position,
                            'v_cen' : slt_ssa.v_cen.position,
                            'h_cen' : slt_ssa.h_cen.position                                      
                             }                                      
    return md                                       

def hf2dwire(*, xstart, xnumstep, xstepsize, 
            zstart, znumstep, zstepsize, 
            acqtime, numrois=1, i0map_show=True, itmap_show=False,
            energy=None, u_detune=None):

    '''
    input:
        xstart, xnumstep, xstepsize (float)
        zstart, znumstep, zstepsize (float)
        acqtime (float): acqusition time to be set for both xspress3 and F460
        numrois (integer): number of ROIs set to display in the live raster scans. This is for display ONLY. 
                           The actualy number of ROIs saved depend on how many are enabled and set in the read_attr
                           However noramlly one cares only the raw XRF spectra which are all saved and will be used for fitting.
        i0map_show (boolean): When set to True, map of the i0 will be displayed in live raster, default is True
        itmap_show (boolean): When set to True, map of the trasnmission diode will be displayed in the live raster, default is True   
        energy (float): set energy, use with caution, hdcm might become misaligned
        u_detune (float): amount of undulator to detune in the unit of keV
    '''

    #record relevant meta data in the Start document, defined in 90-usersetup.py
    md = get_stock_md()

    #setup the detector
    # TODO do this with configure
    current_preamp.exp_time.put(acqtime-0.2)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put((xnumstep+1)*(znumstep+1))
    
    det = [current_preamp, xs]        

    #setup the live callbacks
    livecallbacks = []
    
    livetableitem = [hf_stage.x, hf_stage.z, 'current_preamp_ch0', 'current_preamp_ch2', 'xs_channel1_rois_roi01_value']

    xstop = xstart + xnumstep*xstepsize
    zstop = zstart + znumstep*zstepsize  
  
    print('xstop = '+str(xstop))  
    print('zstop = '+str(zstop)) 
    
    
    for roi_idx in range(numrois):
        roi_name = 'roi{:02}'.format(roi_idx+1)
        
        roi_key = getattr(xs.channel1.rois, roi_name).value.name
        livetableitem.append(roi_key)
        
    #    livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_net_count')
    #    livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_count')
    #    #roimap = LiveRaster((xnumstep, znumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_net_count', clim=None, cmap='viridis', xlabel='x', ylabel='y', extent=None)
        colormap = 'inferno' #previous set = 'viridis'
    #    roimap = LiveRaster((znumstep, xnumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_count', clim=None, cmap='inferno', 
    #                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, zstop, zstart])

        roimap = myLiveRaster((znumstep+1, xnumstep+1), roi_key, clim=None, cmap='inferno', 
                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, zstop, zstart])
        livecallbacks.append(roimap)


    if i0map_show is True:
        i0map = myLiveRaster((znumstep+1, xnumstep+1), 'current_preamp_ch2', clim=None, cmap='inferno', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, zstop, zstart])
        livecallbacks.append(i0map)

    if itmap_show is True:
        itmap = myLiveRaster((znumstep+1, xnumstep+1), 'current_preamp_ch0', clim=None, cmap='inferno', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, zstop, zstart])
        livecallbacks.append(itmap)
#    commented out liveTable in 2D scan for now until the prolonged time issue is resolved
    livecallbacks.append(LiveTable(livetableitem)) 

    
    #setup the plan  

    if energy is not None:
        if u_detune is not None:
            # TODO maybe do this with set
            energy.detune.put(u_detune)
        # TODO fix name shadowing
        yield from bp.abs_set(energy, energy, wait=True)
    

    
#    shut_b.open_cmd.put(1)
#    while (shut_b.close_status.get() == 1):
#        epics.poll(.5)
#        shut_b.open_cmd.put(1)    
    
    hf2dwire_scanplan = OuterProductAbsScanPlan(det, hf_stage.z, zstart, zstop, znumstep+1, hf_stage.x, xstart, xstop, xnumstep+1, True, md=md)
    hf2dwire_scanplan = bp.subs_wrapper( hf2dwire_scanplan, livecallbacks)
    scaninfo = yield from hf2dwire_scanplan

#    shut_b.close_cmd.put(1)
#    while (shut_b.close_status.get() == 0):
#        epics.poll(.5)
#        shut_b.close_cmd.put(1)

    #write to scan log    
    logscan('2dwire')    
    
    return scaninfo
    
class myLiveRaster(CallbackBase):
    """Simple callback that fills in values based on a raster
    This simply wraps around a `AxesImage`.  seq_num is used to
    determine which pixel to fill in
    Parameters
    ----------
    raster_shap : tuple
        The (row, col) shape of the raster
    I : str
        The field to use for the color of the markers
    clim : tuple, optional
       The color limits
    cmap : str or colormap, optional
       The color map to use
    """
    def __init__(self, raster_shape, I, *,
                 clim=None, cmap='viridis',
                 xlabel='x', ylabel='y', extent=None):
        fig, ax = plt.subplots()
        self.I = I
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect(.001)
        self.ax = ax
        self.fig = fig
        self._Idata = np.ones(raster_shape) * np.nan
        self._norm = mcolors.Normalize()
        if clim is not None:
            self._norm.vmin, self._norm.vmax = clim
        self.clim = clim
        self.cmap = cmap
        self.raster_shape = raster_shape
        self.im = None
        self.extent = extent

    def start(self, doc):
        if self.im is not None:
            raise RuntimeError("Can not re-use LiveRaster")
        self._Idata = np.ones(self.raster_shape) * np.nan
        im = self.ax.imshow(self._Idata, norm=self._norm,
                            cmap=self.cmap, interpolation='none',
                            extent=self.extent)
        self.im = im
        self.ax.set_title('scan {uid} [{sid}]'.format(sid=doc['scan_id'],
                                                      uid=doc['uid'][:6]))
        self.snaking = doc.get('snaking', (False, False))

        cb = self.fig.colorbar(im)
        cb.set_label(self.I)

    def event(self, doc):
        if self.I not in doc['data']:
            return

        seq_num = doc['seq_num'] - 1
        pos = list(np.unravel_index(seq_num, self.raster_shape))
        if self.snaking[1] and (pos[0] % 2):
            pos[1] = self.raster_shape[1] - pos[1] - 1
        pos = tuple(pos)
        self._Idata[pos] = doc['data'][self.I]
        if self.clim is None:
            self.im.set_clim(np.nanmin(self._Idata), np.nanmax(self._Idata))

        self.im.set_array(self._Idata)
