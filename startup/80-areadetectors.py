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
                                                 FileStoreTIFF,
                                                 FileStoreHDF5, new_short_uid,
                                                 FileStoreBase
                                                 )
from ophyd import Signal
from ophyd import Component as C
from hxntools.detectors.merlin import MerlinDetector
from hxntools.handlers import register
import itertools

register(db)


class SRXTIFFPlugin(TIFFPlugin, FileStoreTIFF,
                    FileStoreIterativeWrite):
    file_number_sync = None

class BPMCam(SingleTrigger, AreaDetector):
    cam = C(AreaDetectorCam, '')
    image_plugin = C(ImagePlugin, 'image1:')

    # tiff = C(SRXTIFFPlugin, 'TIFF1:',
    #          #write_path_template='/epicsdata/bpm1-cam1/2016/2/24/')
    #          #write_path_template='/epicsdata/bpm1-cam1/%Y/%m/%d/',
    #          #root='/epicsdata', reg=db.reg)
    #          write_path_template='/nsls2/xf05id1/data/bpm1-cam1/%Y/%m/%d/',
    #          root='/nsls2/xf05id1')
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

# bpmAD = BPMCam('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpmAD', read_attrs=['tiff'])
# bpmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
# bpmAD.tiff.read_attrs = []
bpmAD = BPMCam('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpmAD', read_attrs=[])
bpmAD.read_attrs = ['stats1', 'stats2', 'stats3', 'stats4']
bpmAD.stats1.read_attrs = ['total']
bpmAD.stats2.read_attrs = ['total']
bpmAD.stats3.read_attrs = ['total']
bpmAD.stats4.read_attrs = ['total']

class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    file_number_sync = None

class FileStoreBulkReadable(FileStoreIterativeWrite):

    def _reset_data(self):
        self._datum_uids.clear()
        self._point_counter = itertools.count()

    def bulk_read(self, timestamps):
        # if you see this, delete this line but report to DAMA
        raise Exception("should not be here")
        image_name = self.image_name

        uids = [self.generate_datum(self.image_name, ts, {}) for ts in timestamps]

        #clear so unstage will not save the images twice:
        self._reset_data()
        return {image_name: uids}

    @property
    def image_name(self):
        return self.parent._image_name


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
             #write_path_template='/epicsdata/pixirad/%Y/%m/%d/',
             #root='/epicsdata')
             write_path_template='/nsls2/xf05id1/data/pixirad/%Y/%m/%d/',
             root='/nsls2/xf05id1')

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
             #write_path_template='/epicsdata/hfvlm/%Y/%m/%d/',
             #root='/epicsdata',
             write_path_template='/nsls2/xf05id1/data/hfvlm/%Y/%m/%d/',
             root='/nsls2/xf05id1')

# hfvlmAD = SRXHFVLMCam('XF:05IDD-BI:1{Mscp:1-Cam:1}', name='hfvlm', read_attrs=['tiff'])
# hfvlmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
# hfvlmAD.tiff.read_attrs = []
# hfvlmAD.stats1.read_attrs = ['total']
# hfvlmAD.stats2.read_attrs = ['total']
# hfvlmAD.stats3.read_attrs = ['total']
# hfvlmAD.stats4.read_attrs = ['total']

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
            read_path_template='/data/PCOEDGE/%Y/%m/%d/',
             write_path_template='C:/epicsdata/pcoedge/%Y/%m/%d/',
             root='/data')

#pcoedge = SRXPCOEDGECam('XF:05IDD-ES:1{Det:PCO}',name='pcoedge')
###    read_attrs=['tiff'])
##pcoedge.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4', 'stats5', 'cam']
#pcoedge.read_attrs = ['tiff', 'stats1', 'cam']
#pcoedge.tiff.read_attrs = ['file_name']
#pcoedge.stats1.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pcoedge.stats1.centroid.read_attrs = ['x','y']
##pcoedge.stats1.read_attrs = ['total']
##pcoedge.stats2.read_attrs = ['total']
##pcoedge.stats3.read_attrs = ['total']
##pcoedge.stats4.read_attrs = ['total']
##pcoedge.stats4.read_attrs = ['total','sigma_x','sigma_y']
from pathlib import PurePath
from hxntools.detectors.xspress3 import (XspressTrigger, Xspress3Detector,
                                         Xspress3Channel, Xspress3FileStore,
                                         logger)
from databroker.assets.handlers import Xspress3HDF5Handler, HandlerBase


class BulkXSPRESS(HandlerBase):
    HANDLER_NAME = 'XPS3_FLY'
    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]


class BulkMerlin(BulkXSPRESS):
    HANDLER_NAME = 'MERLIN_FLY_STREAM_V1'
    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]


class BulkMerlinDEBUG(BulkXSPRESS):
    # This is for data take in 'capture' mode, only used for debugging
    # once.
    HANDLER_NAME = 'MERLIN_FLY'
    def __call__(self):
        return self._handle['entry/instrument/detector/data'][1:]


db.reg.register_handler(BulkXSPRESS.HANDLER_NAME, BulkXSPRESS,
                        overwrite=True)
# needed to get at some debugging data
db.reg.register_handler('MERLIN_FLY', BulkMerlinDEBUG,
                        overwrite=True)
db.reg.register_handler(BulkMerlin.HANDLER_NAME, BulkMerlin,
                        overwrite=True)

from ophyd.areadetector.filestore_mixins import FileStorePluginBase
class Xspress3FileStoreFlyable(Xspress3FileStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def filestore_res(self):
        raise Exception("don't want to be here")
        return self._filestore_res

    @property
    def filestore_spec(self):
        if self.parent._mode is SRXMode.fly:
           return BulkXSPRESS.HANDLER_NAME
        return Xspress3HDF5Handler.HANDLER_NAME

    def generate_datum(self, key, timestamp, datum_kwargs):
        if self.parent._mode is SRXMode.step:
            return super().generate_datum(key, timestamp, datum_kwargs)
        elif self.parent._mode is SRXMode.fly:
            # we are doing something _very_ dirty here to skip a level of the inheritance
            # this is brittle is if the MRO changes we may not hit all the level we expect to
            return FileStorePluginBase.generate_datum(self, key, timestamp, datum_kwargs)

    def warmup(self):
        """
        A convenience method for 'priming' the plugin.
        The plugin has to 'see' one acquisition before it is ready to capture.
        This sets the array size, etc.

        NOTE : this comes from:
            https://github.com/NSLS-II/ophyd/blob/master/ophyd/areadetector/plugins.py
        We had to replace "cam" with "settings" here.
        Also modified the stage sigs.

        """
        print("warming up the hdf5 plugin...")
        set_and_wait(self.enable, 1)
        sigs = OrderedDict([(self.parent.settings.array_callbacks, 1),
                            (self.parent.settings.image_mode, 'Single'),
                            (self.parent.settings.trigger_mode, 'Internal'),
                            # just in case tha acquisition time is set very long...
                            (self.parent.settings.acquire_time , 1),
                            #(self.parent.settings.acquire_period, 1),
                            (self.parent.settings.acquire, 1)])

        original_vals = {sig: sig.get() for sig in sigs}

        for sig, val in sigs.items():
            ttime.sleep(0.1)  # abundance of caution
            set_and_wait(sig, val)

        ttime.sleep(2)  # wait for acquisition

        for sig, val in reversed(list(original_vals.items())):
            ttime.sleep(0.1)
            set_and_wait(sig, val)
        print("done")

    def describe(self):
        desc = super().describe()

        if self.parent._mode is SRXMode.fly:
            spec = {'external': 'FileStore:',
                    'dtype' : 'array',
                    # TODO do not hard code
                    'shape' : (self.parent.settings.num_images.get(), 3, 4096),
                    'source': self.prefix
            }
            return {self.parent._f_key: spec}
        else:
            return super().describe()



class SRXXspressTrigger(XspressTrigger):
    def trigger(self):
        if self._staged != Staged.yes:
            raise RuntimeError("not staged")

        self._status = DeviceStatus(self)
        self.settings.erase.put(1)
        self._acquisition_signal.put(1, wait=False)
        trigger_time = ttime.time()
        if self._mode is SRXMode.step:
            for sn in self.read_attrs:
                if sn.startswith('channel') and '.' not in sn:
                    ch = getattr(self, sn)
                    self.dispatch(ch.name, trigger_time)
        elif self._mode is SRXMode.fly:
            self.dispatch(self._f_key, trigger_time)
        else:
            raise Exception(f"unexpected mode {self._mode}")
        self._abs_trigger_count += 1
        return self._status

class SrxXspress3Detector(SRXXspressTrigger, Xspress3Detector):
    # TODO: garth, the ioc is missing some PVs?
    #   det_settings.erase_array_counters
    #       (XF:05IDD-ES{Xsp:1}:ERASE_ArrayCounters)
    #   det_settings.erase_attr_reset (XF:05IDD-ES{Xsp:1}:ERASE_AttrReset)
    #   det_settings.erase_proc_reset_filter
    #       (XF:05IDD-ES{Xsp:1}:ERASE_PROC_ResetFilter)
    #   det_settings.update_attr (XF:05IDD-ES{Xsp:1}:UPDATE_AttrUpdate)
    #   det_settings.update (XF:05IDD-ES{Xsp:1}:UPDATE)
    roi_data = Cpt(PluginBase, 'ROIDATA:')

    # Currently only using three channels. Uncomment these to enable more
    channel1 = C(Xspress3Channel, 'C1_', channel_num=1, read_attrs=['rois'])
    channel2 = C(Xspress3Channel, 'C2_', channel_num=2, read_attrs=['rois'])
    channel3 = C(Xspress3Channel, 'C3_', channel_num=3, read_attrs=['rois'])
    # channels:
    # channel4 = C(Xspress3Channel, 'C4_', channel_num=4)
    # channel5 = C(Xspress3Channel, 'C5_', channel_num=5)
    # channel6 = C(Xspress3Channel, 'C6_', channel_num=6)
    # channel7 = C(Xspress3Channel, 'C7_', channel_num=7)
    # channel8 = C(Xspress3Channel, 'C8_', channel_num=8)

    create_dir = Cpt(EpicsSignal, 'HDF5:FileCreateDir')

    hdf5 = Cpt(Xspress3FileStoreFlyable, 'HDF5:',
               read_path_template='/nsls2/xf05id1/XF05ID1/XSPRESS3/%Y/%m/%d/',
               write_path_template='/epics/data/%Y/%m/%d/',
               root='/nsls2/xf05id1/XF05ID1')

    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method so we can modify this as part of a plan
    fly_next = Cpt(Signal, value=False)


    def __init__(self, prefix, *, f_key='fluor', configuration_attrs=None,
                 read_attrs=None,
                 **kwargs):
        self._f_key = f_key
        if configuration_attrs is None:
            configuration_attrs = ['external_trig', 'total_points',
                                   'spectra_per_point', 'settings',
                                   'rewindable']
        if read_attrs is None:
            read_attrs = ['channel1', 'channel2', 'channel3', 'hdf5']
        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)
        # this is possiblely one too many places to store this
        # in the parent class it looks at if the extrenal_trig signal is high
        self._mode = SRXMode.step

        self.create_dir.put(-3)

    def stop(self, *, success=False):
        ret = super().stop()
        # todo move this into the stop method of the settings object?
        self.settings.acquire.put(0)
        self.hdf5.stop(success=success)
        return ret

    def stage(self):
        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            self._mode = SRXMode.fly
        return super().stage()

    def unstage(self):
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret

xs = SrxXspress3Detector('XF:05IDD-ES{Xsp:1}:', name='xs')
xs.channel1.rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
xs.channel2.rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
xs.channel3.rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
xs.hdf5.num_extra_dims.put(0)
xs.channel2.vis_enabled.put(1)
xs.channel3.vis_enabled.put(1)
xs.settings.num_channels.put(3)

xs.settings.configuration_attrs = ['acquire_period',
			'acquire_time',
			'gain',
			'image_mode',
			'manufacturer',
			'model',
			'num_exposures',
			'num_images',
			'temperature',
			'temperature_actual',
			'trigger_mode',
			'config_path',
			'config_save_path',
			'invert_f0',
			'invert_veto',
			'xsp_name',
			'num_channels',
			'num_frames_config',
			'run_flags',
			'trigger_signal']


# This is necessary for when the ioc restarts
# we have to trigger one image for the hdf5 plugin to work correclty
# else, we get file writing errors
xs.hdf5.warmup()


# Working xs2 detector
class SrxXspress3Detector2(SRXXspressTrigger, Xspress3Detector):
    # TODO: garth, the ioc is missing some PVs?
    #   det_settings.erase_array_counters
    #       (XF:05IDD-ES{Xsp:1}:ERASE_ArrayCounters)
    #   det_settings.erase_attr_reset (XF:05IDD-ES{Xsp:1}:ERASE_AttrReset)
    #   det_settings.erase_proc_reset_filter
    #       (XF:05IDD-ES{Xsp:1}:ERASE_PROC_ResetFilter)
    #   det_settings.update_attr (XF:05IDD-ES{Xsp:1}:UPDATE_AttrUpdate)
    #   det_settings.update (XF:05IDD-ES{Xsp:1}:UPDATE)
    roi_data = Cpt(PluginBase, 'ROIDATA:')

    # XS2 only uses 1 channel. Currently only using three channels. Uncomment these to enable more
    channel1 = C(Xspress3Channel, 'C1_', channel_num=1, read_attrs=['rois'])
    # channel2 = C(Xspress3Channel, 'C2_', channel_num=2, read_attrs=['rois'])
    # channel3 = C(Xspress3Channel, 'C3_', channel_num=3, read_attrs=['rois'])

    create_dir = Cpt(EpicsSignal, 'HDF5:FileCreateDir')

    hdf5 = Cpt(Xspress3FileStoreFlyable, 'HDF5:',
               read_path_template='/nsls2/xf05id1/data/2019-2/XS3MINI',
               write_path_template='/home/xspress3/data/SRX/2019-2',
               root='/nsls2/xf05id1')

    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method so we can modify this as part of a plan
    fly_next = Cpt(Signal, value=False)


    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None,
                 **kwargs):
        if configuration_attrs is None:
            configuration_attrs = ['external_trig', 'total_points',
                                   'spectra_per_point', 'settings',
                                   'rewindable']
        if read_attrs is None:
            read_attrs = ['channel1', 'hdf5']
        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)
        # this is possiblely one too many places to store this
        # in the parent class it looks at if the extrenal_trig signal is high
        self._mode = SRXMode.step

        # self.create_dir.put(-3)

    def stop(self, *, success=False):
        ret = super().stop()
        # todo move this into the stop method of the settings object?
        self.settings.acquire.put(0)
        self.hdf5.stop(success=success)
        return ret

    def stage(self):
        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            self._mode = SRXMode.fly
        return super().stage()

    def unstage(self):
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret


try:
    xs2 = SrxXspress3Detector2('XF:05IDD-ES{Xsp:2}:', name='xs2')
    xs2.channel1.rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
    xs2.hdf5.num_extra_dims.put(0)
    xs2.hdf5.warmup()
except TimeoutError:
    xs2 = None
    print('\nCannot connect to xs2. Continuing without device.\n')
except:
    xs2 = None
    print('\nUnexpected error connecting to xs2.\n', sys.exc_info()[0], end='\n\n')



for i in range(1,4):
    ch=getattr(xs.channel1.rois,'roi{:02}.value'.format(i))
    ch.name = 'ROI_{:02}'.format(i)

class MerlinFileStoreHDF5(FileStoreBase):

    _spec = 'TPX_HDF5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs.update([('auto_increment', 'Yes'),
                                ('array_counter', 0),
                                ('auto_save', 'Yes'),
                                ('num_capture', 0),  # this will be updated later
                                (self.file_template, '%s%s_%6.6d.h5'),
                                (self.file_write_mode, 'Stream'),
                                # (self.compression, 'zlib'),
                                (self.capture, 1)
                                ])
        self._point_counter = None

    def unstage(self):
        self._point_counter = None
        return super().unstage()

    def make_filename(self):
        filename = new_short_uid()
        formatter = datetime.datetime.now().strftime
        write_path = formatter(self.write_path_template)
        read_path = formatter(self.read_path_template)


        fn, read_path, write_path = filename, read_path, write_path
        return fn, read_path, write_path

    @property
    def filestore_spec(self):
        if self.parent._mode is SRXMode.fly:
            return BulkMerlin.HANDLER_NAME
        return 'TPX_HDF5'

    def generate_datum(self, key, timestamp, datum_kwargs):
        if self.parent._mode is SRXMode.fly:
            return super().generate_datum(key, timestamp, datum_kwargs)
        elif self.parent._mode is SRXMode.step:
            i = next(self._point_counter)
            datum_kwargs = datum_kwargs or {}
            datum_kwargs.update({'point_number': i})
            return super().generate_datum(key, timestamp, datum_kwargs)

    def stage(self):
        # Make a filename.
        filename, read_path, write_path = self.make_filename()

        # Ensure we do not have an old file open.
        set_and_wait(self.capture, 0)
        # These must be set before parent is staged (specifically
        # before capture mode is turned on. They will not be reset
        # on 'unstage' anyway.
        set_and_wait(self.file_path, write_path)
        set_and_wait(self.file_name, filename)
        set_and_wait(self.file_number, 0)
        staged = super().stage()

        # AD does this same templating in C, but we can't access it
        # so we do it redundantly here in Python.
        self._fn = self.file_template.get() % (read_path,
                                               filename,
                                               self.file_number.get() - 1)
                                               # file_number is *next* iteration
        self._fp = read_path
        if not self.file_path_exists.get():
            raise IOError("Path %s does not exist on IOC."
                          "" % self.file_path.get())

        if self.parent._mode is SRXMode.fly:
            res_kwargs = {}
        else:
            res_kwargs = {'frame_per_point': 1}
            self._point_counter = itertools.count()

        logger.debug("Inserting resource with filename %s", self._fn)
        self._generate_resource(res_kwargs)

        return staged

class HDF5PluginWithFileStoreMerlin(HDF5Plugin, MerlinFileStoreHDF5):

    def stage(self):
        if np.array(self.array_size.get()).sum() == 0:
            raise Exception("you must warmup the hdf plugin via the `warmup()` "
                            "method on the hdf5 plugin.")

        return super().stage()



class SRXMerlin(SingleTrigger, MerlinDetector):
    total_points = Cpt(Signal, value=1, doc="The total number of points to be taken")
    fly_next = Cpt(Signal, value=False, doc="latch to put the detector in 'fly' mode")

    hdf5 = Cpt(HDF5PluginWithFileStoreMerlin, 'HDF1:',
               read_attrs=[],
               read_path_template='/nsls2/xf05id1/XF05ID1/MERLIN/%Y/%m/%d',
               configuration_attrs=[],
               write_path_template='/epicsdata/merlin/%Y/%m/%d',
               root='/nsls2/xf05id1')

    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    stats5 = Cpt(StatsPlugin, 'Stats5:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    transform1 = Cpt(TransformPlugin, 'Trans1:')

    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')

    # def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None,
    #              **kwargs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = SRXMode.step

    def stop(self, success=False):
        ret = super().stop(success=success)
        self.hdf5.stop()
        return ret

    def stage(self):
        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            # According to Ken's comments in hxntools, this is a de-bounce time
            # when in external trigger mode

            # moved this to the plan
            # self.stage_sigs[self.cam.acquire_time] = 0.005
            # self.stage_sigs[self.cam.acquire_period] = 0.0066392

            self.stage_sigs[self.cam.trigger_mode] = 2
            self._mode = SRXMode.fly
        else:
            # Set trigger mode
            self.stage_sigs[self.cam.trigger_mode] = 0

            # Make sure we respect whatever the exposure time is set to
            count_time = self.cam.acquire_time.get()
            if count_time is not None:
                self.stage_sigs[self.cam.acquire_time] = count_time
                self.stage_sigs[self.cam.acquire_period] = count_time + 0.005

            # self.stage_sigs.pop(self.cam.acquire_time)
            # self.stage_sigs.pop(self.cam.acquire_period)
            # self.stage_sigs[self.cam.trigger_mode] = 0

            self._mode = SRXMode.step

        return super().stage()

    def unstage(self):
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret

try:
    merlin = SRXMerlin('XF:05IDD-ES{Merlin:1}', name='merlin', read_attrs=['hdf5', 'cam', 'stats1'])
    merlin.hdf5.read_attrs = []
except TimeoutError:
    print('\nCannot connect to Merlin. Continuing without device.\n')
except:
    print('\nUnexpected error connecting to Merlin.\n', sys.exc_info()[0], end='\n\n')
