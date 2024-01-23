print(f'Loading {__file__}...')


import os
import h5py
import datetime

# import ophyd
from hxntools.detectors.dexela import (DexelaDetector,)
from nslsii.detectors.xspress3 import (logger, )
from databroker.assets.handlers import HandlerBase
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreHDF5IterativeWrite,
                                                 FileStoreTIFFSquashing,
                                                 FileStoreTIFF,
                                                 FileStoreHDF5,
                                                 new_short_uid,
                                                 FileStoreBase,
                                                 )
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector import (AreaDetector, PixiradDetectorCam, ImagePlugin,
                                TIFFPlugin, StatsPlugin, HDF5Plugin,
                                ProcessPlugin, ROIPlugin, TransformPlugin,
                                OverlayPlugin)
from ophyd import Component as Cpt


class BulkDexela(HandlerBase):
    HANDLER_NAME = 'DEXELA_FLY_V1'

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]

    def close(self):
        self._handle.close()
        self._handle = None
        super().close()


db.reg.register_handler(BulkDexela.HANDLER_NAME, BulkDexela,
                        overwrite=True)


class DexelaFileStoreHDF5(FileStoreBase):
    @property
    def filestore_spec(self):
        if self.parent._mode is SRXMode.fly:
            return BulkDexela.HANDLER_NAME
        return 'TPX_HDF5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs.update([('auto_increment', 'Yes'),
                                ('array_counter', 0),
                                ('auto_save', 'Yes'),
                                ('num_capture', 0),  # will be updated later
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
        # set_and_wait(self.capture, 0)  // deprecated
        self.capture.set(0).wait()
        # These must be set before parent is staged (specifically
        # before capture mode is turned on. They will not be reset
        # on 'unstage' anyway.
        # set_and_wait(self.file_path, write_path)  // deprecated
        self.file_path.set(write_path).wait()
        # set_and_wait(self.file_name, filename)  // deprecated
        self.file_name.set(filename).wait()
        # set_and_wait(self.file_number, 0)  // deprecated
        self.file_number.set(0).wait()
        if self.parent._mode is SRXMode.step:
            # set_and_wait(self.num_capture, self.parent.total_points.get())  // deprecated
            self.num_capture.set(self.parent.total_points.get()).wait()

        staged = super().stage()

        # AD does this same templating in C, but we can't access it
        # so we do it redundantly here in Python.
        # file_number is *next* iteration
        self._fn = self.file_template.get() % (read_path,
                                               filename,
                                               self.file_number.get() - 1)
        self._fp = read_path
        if not self.file_path_exists.get():
            raise IOError("Path %s does not exist on IOC."
                          "" % self.file_path.get())

        if self.parent._mode is SRXMode.fly:
            res_kwargs = {}
        else:
            # set_and_wait(self.parent.cam.num_images, 1)  // deprecated
            self.parent.cam.num_images.set(1).wait()
            res_kwargs = {'frame_per_point': 1}

            self._point_counter = itertools.count()

        logger.debug("Inserting resource with filename %s", self._fn)
        self._generate_resource(res_kwargs)

        return staged


class DexelaHDFWithFileStore(HDF5Plugin, DexelaFileStoreHDF5):
    def stage(self):
        if np.array(self.array_size.get()).sum() == 0:
            raise Exception("You must warmup the hdf plugin via the `warmup()`"
                            " method on the hdf5 plugin.")

        return super().stage()


class SRXDexelaDetector(SingleTrigger, DexelaDetector):
    total_points = Cpt(Signal,
                       value=1,
                       doc="The total number of points to be taken")
    hdf5 = Cpt(DexelaHDFWithFileStore, 'HDF1:',
               read_attrs=[],
               configuration_attrs=[],
               # write_path_template='Z:\\%Y\\%m\\%d\\',
               # write_path_template='C:\\temp\\dexela\\%Y\\%m\\%d\\',
               write_path_template='W:\\legacy\\dexela\\%Y\\%m\\%d\\',
               # write_path_template='C:\\temp\\write_here\\',
               # read_path_template='/nsls2/xf05id1/XF05ID1/dexela/%Y/%m/%d/',
               read_path_template='/nsls2/data/srx/legacy/dexela/%Y/%m/%d/',
               root='/nsls2/data/srx/legacy/dexela/')
    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method so we can modify this as part of a plan
    fly_next = Cpt(Signal, value=False)

    roi1 = Cpt(ROIPlugin, 'ROI1:')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = SRXMode.step
        # self.cam.trigger_mode = EpicsSignalWithRBV("XF:05IDD-ES{Dexela:1}cam1:TriggerMode")

    def stage(self):
        # do the latching
        if self.fly_next.get():
            self.fly_next.put(False)
            self._mode = SRXMode.fly

        self.cam.stage_sigs['image_mode'] = 'Multiple'
        if self._mode is SRXMode.fly:
            # self.cam.stage_sigs['trigger_mode'] = 'Ext. Edge Single'
            self.cam.stage_sigs['trigger_mode'] = 'Ext. Bulb'
        else:
            self.cam.stage_sigs['trigger_mode'] = 'Int. Fixed Rate'

        return super().stage()

    def unstage(self):
        try:
            ret = super().unstage()
        finally:
            self._mode = SRXMode.step
        return ret


try:
    dexela = SRXDexelaDetector('XF:05IDD-ES{Dexela:1}', name='dexela')
    dexela.read_attrs = ['hdf5']
    # TODO: Automatically warmup
    # if np.array(dexela.cam.array_size.get()).sum() == 0:
    #     print("  Warmup...", end="", flush=True)
    #     dexela.hdf5.warmup()
    #     print("done")
except TimeoutError:
    dexela = None
    print('\nCannot connect to Dexela. Continuing without device.\n')
except Exception:
    dexela = None
    print('\nUnexpected error connecting to Dexela.\n',
          sys.exc_info()[0],
          end='\n\n')
