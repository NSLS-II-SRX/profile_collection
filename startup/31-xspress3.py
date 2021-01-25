print(f"Loading {__file__}...")

import os
import h5py
import sys
import numpy as np
import time as ttime
from ophyd.areadetector.plugins import PluginBase
from ophyd import Signal, DeviceStatus
from ophyd import Component as Cpt
from ophyd.areadetector.filestore_mixins import FileStorePluginBase
from ophyd.device import Staged
from enum import Enum

from nslsii.detectors.xspress3 import (
    XspressTrigger,
    Xspress3Detector,
    Xspress3Channel,
    Xspress3FileStore,
)

try:
    from area_detector_handlers import HandlerBase
    from area_detector_handlers.handlers import Xspress3HDF5Handler
except ImportError:
    from databroker.assets.handlers import Xspress3HDF5Handler, HandlerBase


class SRXMode(Enum):
    step = 1
    fly = 2


class BulkXspress(HandlerBase):
    HANDLER_NAME = "XPS3_FLY"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self):
        return self._handle["entry/instrument/detector/data"][:]


db.reg.register_handler(BulkXspress.HANDLER_NAME, BulkXspress, overwrite=True)


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
            return BulkXspress.HANDLER_NAME
        return Xspress3HDF5Handler.HANDLER_NAME

    def generate_datum(self, key, timestamp, datum_kwargs):
        if self.parent._mode is SRXMode.step:
            return super().generate_datum(key, timestamp, datum_kwargs)
        elif self.parent._mode is SRXMode.fly:
            # we are doing something _very_ dirty here to skip a level
            # of the inheritance
            # this is brittle is if the MRO changes we may not hit all
            # the level we expect to
            return FileStorePluginBase.generate_datum(
                self, key, timestamp, datum_kwargs
            )

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
        print("Warming up the hdf5 plugin...", end="")
        set_and_wait(self.enable, 1)
        sigs = OrderedDict(
            [
                (self.parent.settings.array_callbacks, 1),
                (self.parent.settings.image_mode, "Single"),
                (self.parent.settings.trigger_mode, "Internal"),
                # In case the acquisition time is set very long
                (self.parent.settings.acquire_time, 1),
                # (self.parent.settings.acquire_period, 1),
                (self.parent.settings.acquire, 1),
            ]
        )

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
            spec = {
                "external": "FileStore:",
                "dtype": "array",
                # TODO do not hard code
                "shape": (self.parent.settings.num_images.get(), 3, 4096),
                "source": self.prefix,
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
                if sn.startswith("channel") and "." not in sn:
                    ch = getattr(self, sn)
                    self.dispatch(ch.name, trigger_time)
        elif self._mode is SRXMode.fly:
            self.dispatch(self._f_key, trigger_time)
        else:
            raise Exception(f"unexpected mode {self._mode}")
        self._abs_trigger_count += 1
        return self._status


class SrxXSP3Handler:
    XRF_DATA_KEY = "entry/instrument/detector/data"

    def __init__(self, filepath, **kwargs):
        self._filepath = filepath

    def __call__(self, **kwargs):
        with h5py.File(self._filepath, "r") as f:
            return np.asarray(f[self.XRF_DATA_KEY])


class SrxXspress3Detector(SRXXspressTrigger, Xspress3Detector):
    # TODO: garth, the ioc is missing some PVs?
    #   det_settings.erase_array_counters
    #       (XF:05IDD-ES{Xsp:1}:ERASE_ArrayCounters)
    #   det_settings.erase_attr_reset (XF:05IDD-ES{Xsp:1}:ERASE_AttrReset)
    #   det_settings.erase_proc_reset_filter
    #       (XF:05IDD-ES{Xsp:1}:ERASE_PROC_ResetFilter)
    #   det_settings.update_attr (XF:05IDD-ES{Xsp:1}:UPDATE_AttrUpdate)
    #   det_settings.update (XF:05IDD-ES{Xsp:1}:UPDATE)
    roi_data = Cpt(PluginBase, "ROIDATA:")

    erase = Cpt(EpicsSignal, "ERASE")

    array_counter = Cpt(EpicsSignal, "ArrayCounter_RBV")

    # Currently only using three channels. Uncomment these to enable more
    channel1 = Cpt(Xspress3Channel, "C1_", channel_num=1, read_attrs=["rois"])
    channel2 = Cpt(Xspress3Channel, "C2_", channel_num=2, read_attrs=["rois"])
    channel3 = Cpt(Xspress3Channel, "C3_", channel_num=3, read_attrs=["rois"])
    channel4 = Cpt(Xspress3Channel, "C4_", channel_num=4, read_attrs=["rois"])
    # channels:
    # channel5 = Cpt(Xspress3Channel, 'C5_', channel_num=5)
    # channel6 = Cpt(Xspress3Channel, 'C6_', channel_num=6)
    # channel7 = Cpt(Xspress3Channel, 'C7_', channel_num=7)
    # channel8 = Cpt(Xspress3Channel, 'C8_', channel_num=8)

    create_dir = Cpt(EpicsSignal, "HDF5:FileCreateDir")

    hdf5 = Cpt(
        Xspress3FileStoreFlyable,
        "HDF5:",
        read_path_template="/nsls2/xf05id1/XF05ID1/XSPRESS3/%Y/%m/%d/",
        # write_path_template='/epics/data/%Y/%m/%d/', #SRX old xspress3
        write_path_template="/home/xspress3/data/%Y/%m/%d/",#TES xspress3
        root="/nsls2/xf05id1/XF05ID1",
    )

    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method so we can modify this as part of a plan
    fly_next = Cpt(Signal, value=False)

    def __init__(
        self,
        prefix,
        *,
        f_key="fluor",
        configuration_attrs=None,
        read_attrs=None,
        **kwargs,
    ):
        self._f_key = f_key
        if configuration_attrs is None:
            configuration_attrs = [
                "external_trig",
                "total_points",
                "spectra_per_point",
                "settings",
                "rewindable",
            ]
        if read_attrs is None:
            read_attrs = ["channel1", "channel2", "channel3", "channel4", "hdf5"]
        super().__init__(
            prefix,
            configuration_attrs=configuration_attrs,
            read_attrs=read_attrs,
            **kwargs,
        )
        # this is possiblely one too many places to store this
        # in the parent class it looks at if the extrenal_trig signal is high
        self._mode = SRXMode.step

        # 2020-01-24
        # Commented out by AMK for using the xs3-server-IOC from TES
        # self.create_dir.put(-3)

    def stop(self, *, success=False):
        ret = super().stop()
        # todo move this into the stop method of the settings object?
        self.settings.acquire.put(0)
        self.hdf5.stop(success=success)
        return ret

    def stage(self):
        # Erase what is currently in the system
        # This prevents a single hot pixel in the upper-left corner of a map
        xs.erase.put(0)
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
    xs = SrxXspress3Detector("XF:05IDD-ES{Xsp:1}:", name="xs")
    xs.channel1.rois.read_attrs = ["roi{:02}".format(j)
                                   for j in [1, 2, 3, 4]]
    xs.channel2.rois.read_attrs = ["roi{:02}".format(j)
                                   for j in [1, 2, 3, 4]]
    xs.channel3.rois.read_attrs = ["roi{:02}".format(j)
                                   for j in [1, 2, 3, 4]]
    xs.channel4.rois.read_attrs = ["roi{:02}".format(j)
                                   for j in [1, 2, 3, 4]]
    if "TOUCHBEAMLINE" in os.environ and os.environ["TOUCHBEAMLINE"] == '1':
        xs.settings.num_channels.put(4) #4 for ME4 detector
        xs.channel1.vis_enabled.put(1)
        xs.channel2.vis_enabled.put(1)
        xs.channel3.vis_enabled.put(1)
        xs.channel4.vis_enabled.put(1)
        xs.hdf5.num_extra_dims.put(0)

        xs.settings.configuration_attrs = [
            "acquire_period",
            "acquire_time",
            "gain",
            "image_mode",
            "manufacturer",
            "model",
            "num_exposures",
            "num_images",
            "temperature",
            "temperature_actual",
            "trigger_mode",
            "config_path",
            "config_save_path",
            "invert_f0",
            "invert_veto",
            "xsp_name",
            "num_channels",
            "num_frames_config",
            "run_flags",
            "trigger_signal",
        ]

        # This is necessary for when the IOC restarts
        # We have to trigger one image for the hdf5 plugin to work correctly
        # else, we get file writing errors
        xs.hdf5.warmup()

        # Rename the ROIs
        for i in range(1, 4):
            ch = getattr(xs.channel1.rois, "roi{:02}.value".format(i))
            ch.name = "ROI_{:02}".format(i)
except TimeoutError:
    xs = None
    print("\nCannot connect to xs. Continuing without device.\n")
except Exception as ex:
    xs = None
    print("\nUnexpected error connecting to xs.\n")
    print(ex, end="\n\n")


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
    roi_data = Cpt(PluginBase, "ROIDATA:")

    # XS2 only uses 1 channel. Currently only using three channels.
    # Uncomment these to enable more
    channel1 = Cpt(Xspress3Channel, "C1_", channel_num=1, read_attrs=["rois"])
    # channel2 = Cpt(Xspress3Channel, 'C2_', channel_num=2, read_attrs=['rois'])
    # channel3 = Cpt(Xspress3Channel, 'C3_', channel_num=3, read_attrs=['rois'])

    erase = Cpt(EpicsSignal, "ERASE")

    array_counter = Cpt(EpicsSignal, "ArrayCounter_RBV")

    create_dir = Cpt(EpicsSignal, "HDF5:FileCreateDir")

    hdf5 = Cpt(
        Xspress3FileStoreFlyable,
        "HDF5:",
        read_path_template="/nsls2/xf05id1/data/2020-2/XS3MINI",
        write_path_template="/home/xspress3/data/SRX/2020-2",
        root="/nsls2/xf05id1",
    )

    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method so we can modify this as part of a plan
    fly_next = Cpt(Signal, value=False)

    def __init__(
        self,
        prefix,
        *,
        f_key="fluor",
        configuration_attrs=None,
        read_attrs=None,
        **kwargs,
    ):
        self._f_key = f_key
        if configuration_attrs is None:
            configuration_attrs = [
                "external_trig",
                "total_points",
                "spectra_per_point",
                "settings",
                "rewindable",
            ]
        if read_attrs is None:
            read_attrs = ["channel1", "hdf5"]
        super().__init__(
            prefix,
            configuration_attrs=configuration_attrs,
            read_attrs=read_attrs,
            **kwargs,
        )
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
    xs2 = SrxXspress3Detector2("XF:05IDD-ES{Xsp:2}:",
                               name="xs2",
                               f_key="fluor_xs2")
    xs2.channel1.rois.read_attrs = ["roi{:02}".format(j)
                                    for j in [1, 2, 3, 4]]
    if "TOUCHBEAMLINE" in os.environ and os.environ["TOUCHBEAMLINE"] == '1':
        xs2.hdf5.num_extra_dims.put(0)
        xs2.hdf5.warmup()
except TimeoutError:
    xs2 = None
    print("\nCannot connect to xs2. Continuing without device.\n")
except Exception as ex:
    xs2 = None
    print("\nUnexpected error connecting to xs2.\n", ex, end="\n\n")
