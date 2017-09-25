from ophyd.areadetector import (AreaDetector, PixiradDetectorCam, ImagePlugin,
                                TIFFPlugin, StatsPlugin, HDF5Plugin,
                                ProcessPlugin, ROIPlugin, TransformPlugin,
                                OverlayPlugin)
from ophyd.areadetector.plugins import PluginBase
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.device import BlueskyInterface
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreHDF5IterativeWrite,
                                                 FileStoreTIFFSquashing,
                                                 FileStoreTIFF)
from ophyd import Signal
from ophyd import Component as C

from hxntools.handlers import register
register(db)

class SRXTIFFPlugin(TIFFPlugin, FileStoreTIFF,
                    FileStoreIterativeWrite):
    file_number_sync = None

class BPMCam(SingleTrigger, AreaDetector):
    cam = C(AreaDetectorCam, '')
    image_plugin = C(ImagePlugin, 'image1:')

    tiff = C(SRXTIFFPlugin, 'TIFF1:',
             #write_path_template='/epicsdata/bpm1-cam1/2016/2/24/')
             write_path_template='/epicsdata/bpm1-cam1/%Y/%m/%d/',
             root='/epicsdata', reg=db.reg)
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    # this is flakey?
    # stats5 = C(StatsPlugin, 'Stats5:')
    pass

bpmAD = BPMCam('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpmAD', read_attrs=['tiff'])
bpmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
bpmAD.tiff.read_attrs = []
bpmAD.stats1.read_attrs = ['total']
bpmAD.stats2.read_attrs = ['total']
bpmAD.stats3.read_attrs = ['total']
bpmAD.stats4.read_attrs = ['total']

class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    file_number_sync = None

class SRXPixirad(SingleTrigger,AreaDetector):

    det = C(PixiradDetectorCam, 'cam1:')
    image = C(ImagePlugin, 'image1:')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    tiff = C(SRXTIFFPlugin, 'TIFF1:',
             write_path_template='/epicsdata/pixirad/%Y/%m/%d/',
             root='/epicsdata')

#pixi = SRXPixirad('XF:05IDD-ES:1{Det:Pixi}', name='pixi', read_attrs=['stats1','stats2','stats3','stats4','tiff'])
#pixi.stats1.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.stats2.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.stats3.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.stats4.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.tiff.read_attrs = []

class SRXHFVLMCam(SingleTrigger,AreaDetector):
    cam = C(AreaDetectorCam, '')
    image_plugin = C(ImagePlugin, 'image1:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    over1 = C(OverlayPlugin, 'Over1:')
    trans1 = C(TransformPlugin, 'Trans1:')
    tiff = C(SRXTIFFPlugin, 'TIFF1:',
             write_path_template='/epicsdata/hfvlm/%Y/%m/%d/',
             root='/epicsdata',
             reg=db.reg)

hfvlmAD = SRXHFVLMCam('XF:05IDD-BI:1{Mscp:1-Cam:1}', name='hfvlm', read_attrs=['tiff'])
hfvlmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
hfvlmAD.tiff.read_attrs = []
hfvlmAD.stats1.read_attrs = ['total']
hfvlmAD.stats2.read_attrs = ['total']
hfvlmAD.stats3.read_attrs = ['total']
hfvlmAD.stats4.read_attrs = ['total']

class SRXPCOEDGECam(SingleTrigger,AreaDetector):
    cam = C(AreaDetectorCam, 'cam1:')
    image_plugin = C(ImagePlugin, 'image1:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    stats5 = C(StatsPlugin, 'Stats5:')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    tiff = C(SRXTIFFPlugin, 'TIFF1:',
            read_path_template='/data/PCOEDGE/2017-3/',
             write_path_template='C:/epicsdata/pcoedge/2017-3\\',
             root='/data',
             reg=db.reg)

pcoedge = SRXPCOEDGECam('XF:05IDD-ES:1{Det:PCO}',name='pcoedge')
###    read_attrs=['tiff'])
##pcoedge.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4', 'stats5', 'cam']
pcoedge.read_attrs = ['tiff', 'stats1', 'cam']
pcoedge.tiff.read_attrs = ['file_name']
pcoedge.stats1.read_attrs = ['total','centroid','sigma_x','sigma_y']
pcoedge.stats1.centroid.read_attrs = ['x','y']
##pcoedge.stats1.read_attrs = ['total']
##pcoedge.stats2.read_attrs = ['total']
##pcoedge.stats3.read_attrs = ['total']
##pcoedge.stats4.read_attrs = ['total']
##pcoedge.stats4.read_attrs = ['total','sigma_x','sigma_y']
from pathlib import PurePath
from hxntools.detectors.xspress3 import (XspressTrigger, Xspress3Detector,
                                         Xspress3Channel, Xspress3FileStore, logger)
from databroker.assets.handlers import Xspress3HDF5Handler, HandlerBase

class BulkXSPRESS(HandlerBase):
    HANDLER_NAME = 'XPS3_FLY'
    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]
    
db.reg.register_handler(BulkXSPRESS.HANDLER_NAME, BulkXSPRESS,
                       overwrite=True)

class Xspress3FileStoreFlyable(Xspress3FileStore):
    fly_next = Cpt(Signal, value=False)

    #fixing upstream bug
    def read(self):
        timestamp = time.time()
        uids = [self._reg.register_datum(self._filestore_res, kw)
                for kw in self._get_datum_args(self.parent._abs_trigger_count)]

        return {self.mds_keys[ch]: {'timestamp': timestamp,
                                    'value': uid,
                                    }
                for uid, ch in zip(uids, self.channels)
                }
    
    @property
    def filestore_res(self):
        return self._filestore_res

    @property
    def fs_type(self):
        if self.fly_next.get():
            self.fly_next.put(False)
            return BulkXSPRESS.HANDLER_NAME
        return Xspress3HDF5Handler.HANDLER_NAME

    def stage(self):
        # if should external trigger
        ext_trig = self.parent.external_trig.get()

        logger.debug('Stopping xspress3 acquisition')
        # really force it to stop acquiring
        self.settings.acquire.put(0, wait=True)

        total_points = self.parent.total_points.get()
        spec_per_point = self.parent.spectra_per_point.get()
        total_capture = total_points * spec_per_point

        # stop previous acquisition
        self.stage_sigs[self.settings.acquire] = 0

        # re-order the stage signals and disable the calc record which is
        # interfering with the capture count
        self.stage_sigs.pop(self.num_capture, None)
        self.stage_sigs.pop(self.settings.num_images, None)
        self.stage_sigs[self.num_capture_calc_disable] = 1

        if ext_trig:
            logger.debug('Setting up external triggering')
            self.stage_sigs[self.settings.trigger_mode] = 'TTL Veto Only'
            self.stage_sigs[self.settings.num_images] = total_capture
        else:
            logger.debug('Setting up internal triggering')
            # self.settings.trigger_mode.put('Internal')
            # self.settings.num_images.put(1)
            self.stage_sigs[self.settings.trigger_mode] = 'Internal'
            self.stage_sigs[self.settings.num_images] = spec_per_point

        self.stage_sigs[self.auto_save] = 'No'
        logger.debug('Configuring other filestore stuff')

        logger.debug('Making the filename')
        filename, read_path, write_path = self.make_filename()

        logger.debug('Setting up hdf5 plugin: ioc path: %s filename: %s',
                     write_path, filename)

        logger.debug('Erasing old spectra')
        self.settings.erase.put(1, wait=True)

        # this must be set after self.settings.num_images because at the Epics
        # layer  there is a helpful link that sets this equal to that (but
        # not the other way)
        self.stage_sigs[self.num_capture] = total_capture

        # actually apply the stage_sigs
        ret = super().stage()

        self._fn = self.file_template.get() % (self._fp,
                                               self.file_name.get(),
                                               self.file_number.get())

        if not self.file_path_exists.value:
            raise IOError("Path {} does not exits on IOC!! Please Check"
                          .format(self.file_path.value))

        logger.debug('Inserting the filestore resource: %s', self._fn)
        fn = PurePath(self._fn).relative_to(self.reg_root)
        # This change needs to be upstreamed        
        self._filestore_res = self._resource = self._reg.insert_resource(
            self.fs_type, str(fn), {},
            root=str(self.reg_root))

        # this gets auto turned off at the end
        self.capture.put(1)

        # Xspress3 needs a bit of time to configure itself...
        # this does not play nice with the event loop :/
        time.sleep(self._config_time)

        return ret



class SrxXspress3Detector(XspressTrigger, Xspress3Detector):
    # TODO: garth, the ioc is missing some PVs?
    #   det_settings.erase_array_counters
    #       (XF:05IDD-ES{Xsp:1}:ERASE_ArrayCounters)
    #   det_settings.erase_attr_reset (XF:05IDD-ES{Xsp:1}:ERASE_AttrReset)
    #   det_settings.erase_proc_reset_filter
    #       (XF:05IDD-ES{Xsp:1}:ERASE_PROC_ResetFilter)
    #   det_settings.update_attr (XF:05IDD-ES{Xsp:1}:UPDATE_AttrUpdate)
    #   det_settings.update (XF:05IDD-ES{Xsp:1}:UPDATE)
    roi_data = Cpt(PluginBase, 'ROIDATA:')
    channel1 = C(Xspress3Channel, 'C1_', channel_num=1, read_attrs=['rois'])
    channel2 = C(Xspress3Channel, 'C2_', channel_num=2, read_attrs=['rois'])
    channel3 = C(Xspress3Channel, 'C3_', channel_num=3, read_attrs=['rois'])

    hdf5 = Cpt(Xspress3FileStoreFlyable, 'HDF5:',
               read_path_template='/XF05IDD/XSPRESS3/2017-3/',
               write_path_template='/epics/data/2017-3/',
#               root='/data',
               root='/XF05IDD',
               reg=db.reg)

    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None,
                 **kwargs):
        if configuration_attrs is None:
            configuration_attrs = ['external_trig', 'total_points',
                                   'spectra_per_point', 'settings',
                                   'rewindable']
        if read_attrs is None:
            read_attrs = ['channel1', 'channel2', 'channel3', 'hdf5']
        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)

    def stop(self):
        ret = super.stop()
        self.hdf5.stop()
        return ret

    # Currently only using three channels. Uncomment these to enable more
    # channels:
    # channel4 = C(Xspress3Channel, 'C4_', channel_num=4)
    # channel5 = C(Xspress3Channel, 'C5_', channel_num=5)
    # channel6 = C(Xspress3Channel, 'C6_', channel_num=6)
    # channel7 = C(Xspress3Channel, 'C7_', channel_num=7)
    # channel8 = C(Xspress3Channel, 'C8_', channel_num=8)


xs = SrxXspress3Detector('XF:05IDD-ES{Xsp:1}:', name='xs')
xs.channel1.rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
xs.channel2.rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
xs.channel3.rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
xs.hdf5.num_extra_dims.put(0)
xs.channel2.vis_enabled.put(1)
xs.channel3.vis_enabled.put(1)
xs.settings.num_channels.put(3)

for i in range(1,4):
    ch=getattr(xs.channel1.rois,'roi{:02}.value'.format(i))
    ch.name = 'ROI_{:02}'.format(i)
