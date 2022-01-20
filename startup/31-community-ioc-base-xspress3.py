print(f"Loading {__file__}...")
  
from databroker.assets.handlers import Xspress3HDF5Handler

from ophyd import EpicsSignal
from ophyd.areadetector.filestore_mixins import FileStorePluginBase
from ophyd.areadetector.plugins import HDF5Plugin


# copied from nslsii.detectors.xspress3
class CommunityXspress3FileStore(FileStorePluginBase, HDF5Plugin):
    '''Xspress3 acquisition -> filestore'''
    num_capture_calc = Cpt(EpicsSignal, 'NumCapture_CALC')
    num_capture_calc_disable = Cpt(EpicsSignal, 'NumCapture_CALC.DISA')
    filestore_spec = Xspress3HDF5Handler.HANDLER_NAME

    def __init__(self, basename, *, config_time=0.5,
                 # JL this format string will not work with the xspress3 community ioc ophyd device
                 mds_key_format='{self.parent.cam.name}_ch{chan}',  # JL changed settings to parent.cam
                 parent=None,
                 **kwargs):
        super().__init__(basename, parent=parent, **kwargs)
        
        # JL assume parent is a community IOC Xspress3 ophyd class
        det = parent
        # JL replace self.settings with self.parent.cam
        #self.settings = det.settings

        # Use the EpicsSignal file_template from the detector
        self.stage_sigs[self.blocking_callbacks] = 1
        self.stage_sigs[self.enable] = 1
        self.stage_sigs[self.compression] = 'zlib'
        self.stage_sigs[self.file_template] = '%s%s_%6.6d.h5'

        self._filestore_res = None
        
        # JL use the new channel methods
        self.channels = list(range(1, len([_ for _ in det.component_names
                                           if _.startswith('chan')]) + 1))

        # this was in original code, but I kinda-sorta nuked because
        # it was not needed for SRX and I could not guess what it did
        self._master = None

        self._config_time = config_time
        # JL probably need to handle these differently
        self.mds_keys = {chan: mds_key_format.format(self=self, chan=chan)
                         for chan in self.channels}

    def stop(self, success=False):
        ret = super().stop(success=success)
        self.capture.put(0)
        return ret

    def kickoff(self):
        # TODO
        raise NotImplementedError()

    def collect(self):
        # TODO (hxn-specific implementation elsewhere)
        raise NotImplementedError()

    def make_filename(self):
        fn, rp, write_path = super().make_filename()
        if self.parent.make_directories.get():
            makedirs(write_path)
        return fn, rp, write_path

    def unstage(self):
        try:
            i = 0
            # this needs a fail-safe, RE will now hang forever here
            # as we eat all SIGINT to ensure that cleanup happens in
            # orderly manner.
            # If we are here this is a sign that we have not configured the xs3
            # correctly and it is expecting to capture more points than it
            # was triggered to take.
            while self.capture.get() == 1:
                i += 1
                if (i % 50) == 0:
                    logger.warning('Still capturing data .... waiting.')
                time.sleep(0.1)
                if i > 150:
                    logger.warning('Still capturing data .... giving up.')
                    logger.warning('Check that the xspress3 is configured to take the right '
                                   'number of frames '
                                   f'(it is trying to take {self.parent.cam.num_images.get()})')  # JL changed settings to cam
                    self.capture.put(0)
                    break

        except KeyboardInterrupt:
            self.capture.put(0)
            logger.warning('Still capturing data .... interrupted.')

        return super().unstage()

    # JL this method is overridden in CommunityXspress3FileStoreFlyable
    #   so I'm leaving this alone for now
    def generate_datum(self, key, timestamp, datum_kwargs):
        sn, n = next((f'channel{j}', j)
                     for j in self.channels
                     if getattr(self.parent, f'channel{j}').name == key)
        datum_kwargs.update({'frame': self.parent._abs_trigger_count,
                             'channel': int(sn[7:])})
        self.mds_keys[n] = key
        super().generate_datum(key, timestamp, datum_kwargs)

    def stage(self):
        # if should external trigger
        ext_trig = self.parent.external_trig.get()

        logger.debug('Stopping xspress3 acquisition')
        # really force it to stop acquiring
        self.parent.cam.acquire.put(0, wait=True)

        total_points = self.parent.total_points.get()
        if total_points < 1:
            raise RuntimeError("You must set the total points")
        spec_per_point = self.parent.spectra_per_point.get()
        total_capture = total_points * spec_per_point

        # stop previous acquisition
        self.stage_sigs[self.parent.cam.acquire] = 0

        # re-order the stage signals and disable the calc record which is
        # interfering with the capture count
        self.stage_sigs.pop(self.num_capture, None)
        self.stage_sigs.pop(self.parent.cam.num_images, None)
        self.stage_sigs[self.num_capture_calc_disable] = 1

        if ext_trig:
            logger.debug('Setting up external triggering')
            self.stage_sigs[self.parent.cam.trigger_mode] = 'TTL Veto Only'
            self.stage_sigs[self.parent.cam.num_images] = total_capture
        else:
            logger.debug('Setting up internal triggering')
            # self.settings.trigger_mode.put('Internal')
            # self.settings.num_images.put(1)
            self.stage_sigs[self.parent.cam.trigger_mode] = 'Internal'
            self.stage_sigs[self.parent.cam.num_images] = spec_per_point

        # JL experimenting with "Yes" for flyscanning
        # originally 
        #self.stage_sigs[self.auto_save] = 'No'
        self.stage_sigs[self.auto_save] = 'Yes'
        logger.debug('Configuring other filestore stuff')

        logger.debug('Making the filename')
        filename, read_path, write_path = self.make_filename()

        logger.debug('Setting up hdf5 plugin: ioc path: %s filename: %s',
                     write_path, filename)

        # JL commented the next two lines while troubleshooting a delay in nano_scan_and_fly
        logger.debug('Erasing old spectra')
        #self.parent.cam.erase.put(1, wait=True)

        # this must be set after self.settings.num_images because at the Epics
        # layer  there is a helpful link that sets this equal to that (but
        # not the other way)
        self.stage_sigs[self.num_capture] = total_capture

        # actually apply the stage_sigs
        ret = super().stage()

        self._fn = self.file_template.get() % (self._fp,
                                               self.file_name.get(),
                                               self.file_number.get())

        if not self.file_path_exists.get():
            raise IOError("Path {} does not exits on IOC!! Please Check"
                          .format(self.file_path.get()))

        logger.debug('Inserting the filestore resource: %s', self._fn)
        self._generate_resource({})
        self._filestore_res = self._asset_docs_cache[-1][-1]

        # this gets auto turned off at the end
        self.capture.put(1)

        # Xspress3 needs a bit of time to configure itself...
        # this does not play nice with the event loop :/
        time.sleep(self._config_time)

        return ret

    def configure(self, total_points=0, master=None, external_trig=False,
                  **kwargs):
        raise NotImplementedError()

    # JL CommunityXspress3FileStoreFlyable overrides this method
    #   so I'm leaving it for now
    def describe(self):
        # should this use a better value?
        size = (self.width.get(), )

        spec_desc = {'external': 'FILESTORE:',
                     'dtype': 'array',
                     'shape': size,
                     'source': 'FileStore:'
                     }

        desc = OrderedDict()
        for chan in self.channels:
            key = self.mds_keys[chan]
            desc[key] = spec_desc

        return desc


# JL copied nslsii.detectors.xspress3.XspressTrigger and renamed CommunityXspressTrigger 
from ophyd import BlueskyInterface
class CommunityXspressTrigger(BlueskyInterface):
    """Base class for trigger mixin classes
    Subclasses must define a method with this signature:
    `acquire_changed(self, value=None, old_value=None, **kwargs)`
    """
    # TODO **
    # count_time = self.settings.acquire_period

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # settings
        self._status = None
        self._acquisition_signal = self.cam.acquire  # JL replaced settings with cam
        self._abs_trigger_count = 0

    def stage(self):
        self._abs_trigger_count = 0
        self._acquisition_signal.subscribe(self._acquire_changed)
        return super().stage()

    def unstage(self):
        ret = super().unstage()
        self._acquisition_signal.clear_sub(self._acquire_changed)
        self._status = None
        return ret

    def _acquire_changed(self, value=None, old_value=None, **kwargs):
        "This is called when the 'acquire' signal changes."
        if self._status is None:
            return
        if (old_value == 1) and (value == 0):
            # Negative-going edge means an acquisition just finished.
            self._status._finished()

    def trigger(self):
        if self._staged != Staged.yes:
            raise RuntimeError("not staged")

        self._status = DeviceStatus(self)
        self._acquisition_signal.put(1, wait=False)
        trigger_time = ttime.time()

        for sn in self.read_attrs:
            if sn.startswith('channel') and '.' not in sn:
                ch = getattr(self, sn)
                self.dispatch(ch.name, trigger_time)

        self._abs_trigger_count += 1
        return self._status

