print(f"Loading {__file__}...")


import os
import threading
import h5py
import numpy as np
import time as ttime
from ophyd import Device, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
from hxntools.detectors.zebra import Zebra, EpicsSignalWithRBV


class CurrentPreampZebra(Device):
    ch0 = Cpt(EpicsSignalRO, "Cur:I0-I")
    ch1 = Cpt(EpicsSignalRO, "Cur:I1-I")
    ch2 = Cpt(EpicsSignalRO, "Cur:I2-I")
    ch3 = Cpt(EpicsSignalRO, "Cur:I3-I")

    # exp_time = Cpt(EpicsSignal, 'Per-SP')
    exp_time = Cpt(
        EpicsSignal, "XF:05IDD-ES:1{Dev:Zebra1}:PULSE3_WID", add_prefix=()
    )
    trigger_mode = Cpt(EpicsSignal, "Cmd:TrigMode")
    initi_trigger = Cpt(EpicsSignal, "Cmd:Init")
    zebra_trigger = Cpt(
        EpicsSignal, "XF:05IDD-ES:1{Dev:Zebra1}:SOFT_IN:B0", add_prefix=()
    )
    zebra_pulse_3_source = Cpt(
        EpicsSignal, "XF:05IDD-ES:1{Dev:Zebra1}:PULSE3_INP", add_prefix=()
    )

    current_scan_rate = Cpt(EpicsSignal, "Cmd:RdCur.SCAN")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.zebra_trigger] = 0
        #        self.stage_sigs[self.zebra_pulse_3_source] = 44
        self.stage_sigs[self.zebra_pulse_3_source] = 60

        self.current_scan_rate.put(9)
        # update
        # self.trigger_mode.put(5)
        self.stage_sigs[self.trigger_mode] = 5  # fix this
        self.initi_trigger.put(1, wait=True)

    def stage(self):

        # Customize what is done before every scan (and undone at the end)
        # self.stage_sigs[self.trans_diode] = 5
        # or just use pyepics directly if you need to
        ret = super().stage()
        self.initi_trigger.put(1, wait=True)
        return ret

    def trigger(self):
        init_ts = self.ch0.timestamp
        timeout = float(self.exp_time.get() + 0.8)

        def retrigger():
            print(f"[WW] Re-triggered ion chamber;"
                  f"I0 for this point is suspect.")
            self.zebra_trigger.put(0, wait=True)
            self.zebra_trigger.put(1, wait=True)

        def done_cb(
            *args, obj=None, old_value=None,
            value=None, timestamp=None, **kwargs
        ):
            # if the value has changed, assume it is done
            if value != old_value:
                tmr.cancel()
                ret._finished()
                obj.clear_sub(done_cb)

        tmr = threading.Timer(timeout, retrigger)
        tmr.start()
        ret = DeviceStatus(self)

        self.ch0.subscribe(done_cb, event_type=self.ch0.SUB_VALUE, run=False)
        self.zebra_trigger.put(0, wait=True)
        self.zebra_trigger.put(1, wait=True)

        return ret


current_preamp = CurrentPreampZebra("XF:05IDA{IM:1}", name="current_preamp")
# current_preamp = CurrentPreamp('XF:05IDA{IM:1}', name='current_preamp')


class ZebraPositionCaptureData(Device):
    """
    Data arrays for the Zebra position capture function and their metadata.
    """

    # Data arrays
    div1 = Cpt(EpicsSignal, "PC_DIV1")
    div2 = Cpt(EpicsSignal, "PC_DIV2")
    div3 = Cpt(EpicsSignal, "PC_DIV3")
    div4 = Cpt(EpicsSignal, "PC_DIV4")
    enc1 = Cpt(EpicsSignal, "PC_ENC1")
    enc2 = Cpt(EpicsSignal, "PC_ENC2")
    enc3 = Cpt(EpicsSignal, "PC_ENC3")
    enc4 = Cpt(EpicsSignal, "PC_ENC4")
    filt1 = Cpt(EpicsSignal, "PC_FILT1")
    filt2 = Cpt(EpicsSignal, "PC_FILT2")
    filt3 = Cpt(EpicsSignal, "PC_FILT3")
    filt4 = Cpt(EpicsSignal, "PC_FILT4")
    time = Cpt(EpicsSignal, "PC_TIME")
    # Array sizes
    num_cap = Cpt(EpicsSignal, "PC_NUM_CAP")
    num_down = Cpt(EpicsSignal, "PC_NUM_DOWN")
    # BOOLs to denote arrays with data
    cap_enc1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B0")
    cap_enc2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B1")
    cap_enc3_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B2")
    cap_enc4_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B3")
    cap_filt1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B4")
    cap_filt2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B5")
    cap_div1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B6")
    cap_div2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B7")
    cap_div3_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B8")
    cap_div4_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B9")


class ZebraPositionCapture(Device):
    """
    Signals for the position capture function of the Zebra
    """

    # Configuration settings and status PVs
    enc = Cpt(EpicsSignalWithRBV, "PC_ENC")
    dir = Cpt(EpicsSignalWithRBV, "PC_DIR")
    tspre = Cpt(EpicsSignalWithRBV, "PC_TSPRE")
    trig_source = Cpt(EpicsSignalWithRBV, "PC_ARM_SEL")
    arm = Cpt(EpicsSignal, "PC_ARM")
    disarm = Cpt(EpicsSignal, "PC_DISARM")
    armed = Cpt(EpicsSignalRO, "PC_ARM_OUT")
    gate_source = Cpt(EpicsSignalWithRBV, "PC_GATE_SEL")
    gate_start = Cpt(EpicsSignalWithRBV, "PC_GATE_START")
    gate_width = Cpt(EpicsSignalWithRBV, "PC_GATE_WID")
    gate_step = Cpt(EpicsSignalWithRBV, "PC_GATE_STEP")
    gate_num = Cpt(EpicsSignalWithRBV, "PC_GATE_NGATE")
    gated = Cpt(EpicsSignalRO, "PC_GATE_OUT")
    pulse_source = Cpt(EpicsSignalWithRBV, "PC_PULSE_SEL")
    pulse_start = Cpt(EpicsSignalWithRBV, "PC_PULSE_START")
    pulse_width = Cpt(EpicsSignalWithRBV, "PC_PULSE_WID")
    pulse_step = Cpt(EpicsSignalWithRBV, "PC_PULSE_STEP")
    pulse_max = Cpt(EpicsSignalWithRBV, "PC_PULSE_MAX")
    pulse = Cpt(EpicsSignalRO, "PC_PULSE_OUT")
    enc_pos1_sync = Cpt(EpicsSignal, "M1:SETPOS.PROC")
    enc_pos2_sync = Cpt(EpicsSignal, "M2:SETPOS.PROC")
    enc_pos3_sync = Cpt(EpicsSignal, "M3:SETPOS.PROC")
    enc_pos4_sync = Cpt(EpicsSignal, "M4:SETPOS.PROC")
    enc_res1 = Cpt(EpicsSignal, "M1:MRES")
    enc_res2 = Cpt(EpicsSignal, "M2:MRES")
    enc_res3 = Cpt(EpicsSignal, "M3:MRES")
    enc_res4 = Cpt(EpicsSignal, "M4:MRES")
    data_in_progress = Cpt(EpicsSignalRO, "ARRAY_ACQ")
    block_state_reset = Cpt(EpicsSignal, "SYS_RESET.PROC")
    data = Cpt(ZebraPositionCaptureData, "")

    def stage(self):
        self.arm.put(1)

        super().stage()

    def unstage(self):
        self.disarm.put(1)
        self.block_state_reset.put(1)

        super().unstage()


class SRXZebra(Zebra):
    """
    SRX Zebra device.
    """

    pc = Cpt(ZebraPositionCapture, "")

    def __init__(
        self, prefix, *,
        read_attrs=None, configuration_attrs=None, **kwargs
    ):
        if read_attrs is None:
            read_attrs = []
        if configuration_attrs is None:
            configuration_attrs = []

        super().__init__(
            prefix,
            read_attrs=read_attrs,
            configuration_attrs=configuration_attrs,
            **kwargs,
        )


# zebra = SRXZebra("XF:05IDD-ES:1{Dev:Zebra1}:", name="zebra")
# zebra.read_attrs = ["pc.data.enc1", "pc.data.enc2", "pc.data.time"]


class SRXFlyer1Axis(Device):
    """
    This is the Zebra.
    """

    LARGE_FILE_DIRECTORY_WRITE_PATH = (
        "/nsls2/xf05id1/XF05ID1/data/2019-3/fly_scan_ancillary/"
    )
    LARGE_FILE_DIRECTORY_READ_PATH = (
        "/nsls2/xf05id1/XF05ID1/data/2019-3/fly_scan_ancillary/"
    )
    KNOWN_DETS = {"xs", "xs2", "merlin", "dexela"}
    fast_axis = Cpt(Signal, value="HOR", kind="config")

    _encoder = Cpt(
        SRXZebra,
        "XF:05IDD-ES:1{Dev:Zebra1}:",
        name="zebra",
        add_prefix=(),
        read_attrs=["pc.data.enc1", "pc.data.enc2", "pc.data.time"],
    )
    @property
    def encoder(self):
        return self._encoder

    @property
    def detectors(self):
        return tuple(self._dets)

    @detectors.setter
    def detectors(self, value):
        dets = tuple(value)
        if not all(d.name in self.KNOWN_DETS for d in dets):
            raise ValueError(
                f"One or more of {[d.name for d in dets]}"
                f"is not known to the zebra. "
                f"The known detectors are {self.KNOWN_DETS})"
            )
        self._dets = dets

    @property
    def sclr(self):
        return self._sis

    # def __init__(self, encoder, dets, sclr1, fast_axis, *,
    #              reg=db.reg, **kwargs):
    def __init__(self, dets, sclr1, *, reg=db.reg, **kwargs):
        super().__init__("", parent=None, **kwargs)
        self._mode = "idle"
        self._dets = dets
        self._sis = sclr1
        self._filestore_resource = None
        # self._fast_axis = self.fast_axis

        # Gating info for encoder capture
        self.stage_sigs[self._encoder.pc.gate_num] = 1
        self.stage_sigs[self._encoder.pc.pulse_start] = 0

        # PC gate output is 31 for zebra. Use it to trigger xspress3 and I0
        self.stage_sigs[self._encoder.output1.ttl.addr] = 31
        self.stage_sigs[self._encoder.output3.ttl.addr] = 31
        # This is for the merlin
        self.stage_sigs[self._encoder.output2.ttl.addr] = 53
        # This is for the dexela
        # self.stage_sigs[self._encoder.output4.ttl.addr] = 55
        # This is for the xs2
        self.stage_sigs[self._encoder.output4.ttl.addr] = 31

        self.stage_sigs[self._encoder.pc.enc_pos1_sync] = 1
        self.stage_sigs[self._encoder.pc.enc_pos2_sync] = 1
        self.stage_sigs[self._encoder.pc.enc_pos3_sync] = 1
        self.stage_sigs[self._encoder.pc.enc_pos4_sync] = 1

        # Put SIS3820 into single count (not autocount) mode
        self.stage_sigs[self._sis.count_mode] = 0

        # Stop the SIS3820
        self._sis.stop_all.put(1)

        self._encoder.pc.block_state_reset.put(1)
        self.reg = reg
        self._document_cache = []
        self._last_bulk = None

    # def ver_fly_plan():
    #    yield from mv(zebra.fast_axis, 'VER')
    #    yield from _real_fly_scan()
    # def hor_fly_plan():
    #     yield from mv(zebar.fast_axis, 'HOR')
    #     yield from _read_fly_scan()
    def stage(self):
        dir = self.fast_axis.get()
        if dir == "HOR":
            self.stage_sigs[self._encoder.pc.enc] = "Enc2"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
            self.stage_sigs[self._encoder.pc.enc_res2] = 5e-6
        elif dir == "VER":
            self.stage_sigs[self._encoder.pc.enc] = "Enc1"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
            self.stage_sigs[self._encoder.pc.enc_res1] = 5e-6
        elif dir == "DET2HOR":
            self.stage_sigs[self._encoder.pc.enc] = "Enc3"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
            self.stage_sigs[self._encoder.pc.enc_res1] = 5e-5
        elif dir == "DET2VER":
            self.stage_sigs[self._encoder.pc.enc] = "Enc4"
            self.stage_sigs[self._encoder.pc.dir] = "Positive"
            self.stage_sigs[self._encoder.pc.enc_res1] = 5e-5

        super().stage()

    def describe_collect(self):

        ext_spec = "FileStore:"

        spec = {
            "external": ext_spec,
            "dtype": "array",
            "shape": [self._npts],
            "source": "",  # make this the PV of the array the det is writing
        }

        desc = OrderedDict()
        for chan in ("time", "enc1", "enc2"):
            desc[chan] = spec
            desc[chan]["source"] = getattr(self._encoder.pc.data, chan).pvname

        # Handle the detectors we are going to get
        for d in self._dets:
            desc.update(d.describe())

        # Handle the ion chamber that the zebra is collecting
        desc["i0"] = spec
        desc["i0"]["source"] = self._sis.mca2.pvname
        desc["i0_time"] = spec
        desc["i0_time"]["source"] = self._sis.mca1.pvname
        desc["im"] = spec
        desc["im"]["source"] = self._sis.mca3.pvname
        desc["it"] = spec
        desc["it"]["source"] = self._sis.mca4.pvname

        return {"stream0": desc}

    def kickoff(self, *, xstart, xstop, xnum, dwell):
        dets_by_name = {d.name: d for d in self.detectors}

        self._encoder.pc.arm.put(0)
        self._mode = "kicked off"
        self._npts = int(xnum)
        extent = xstop - xstart
        pxsize = extent / (xnum - 1)
        # 2 ms delay between pulses
        decrement = (pxsize / dwell) * 0.002
        if decrement < 1e-5:
            # print('Changing the pulse width')
            decrement = 1e-5
        self._encoder.pc.gate_start.put(xstart)
        self._encoder.pc.gate_step.put(extent + 0.0005)
        self._encoder.pc.gate_width.put(extent + 0.0001)

        self._encoder.pc.pulse_start.put(0.0)
        self._encoder.pc.pulse_max.put(xnum)
        self._encoder.pc.pulse_step.put(pxsize)
        self._encoder.pc.pulse_width.put(pxsize - decrement)
        # If decrement is too small, then zebra will not send individual pulses
        # but integrate over the entire line
        # Hopefully taken care of with decrement check above

        # For dexela, we will use time triggering in a pixel, not position
        if "dexela" in dets_by_name:
            self._encoder.output1.ttl.addr.put(52)
            self._encoder.output3.ttl.addr.put(52)
            self._encoder.pulse1.width.put(0.5 * dwell - 0.050)
        else:
            self._encoder.output1.ttl.addr.put(31)
            self._encoder.output3.ttl.addr.put(31)

        # If both values are not synced, then the X-position was not updating
        # during the scan and will remain at the initial value
        # - AMK
        self._encoder.pc.enc_pos1_sync.put(1)  # Sample Y
        self._encoder.pc.enc_pos2_sync.put(1)  # Sample X
        self._encoder.pc.enc_pos3_sync.put(1)  # Det2 Stage X
        self._encoder.pc.enc_pos4_sync.put(1)  # Det2 Stage Y

        # Arm the zebra
        self._encoder.pc.arm.put(1)

        st = (
            NullStatus()
        )  # TODO Return a status object *first*
           # and do the above asynchronously.
        return st

    def complete(self):
        """
        Call this when all needed data has been collected. This has no idea
        whether that is true, so it will obligingly stop immediately. It is
        up to the caller to ensure that the motion is actually complete.
        """
        # Our acquisition complete PV is: XF:05IDD-ES:1{Dev:Zebra1}:ARRAY_ACQ
        while self._encoder.pc.data_in_progress.get() == 1:
            ttime.sleep(0.01)
        # ttime.sleep(.1)
        self._mode = "complete"
        self._encoder.pc.block_state_reset.put(1)
        # see triggering errors of the xspress3 on suspension.  This is
        # to test the reset of the xspress3 after a line.

        for d in self._dets:
            d.stop(success=True)

        self.__filename = "{}.h5".format(uuid.uuid4())
        self.__filename_sis = "{}.h5".format(uuid.uuid4())
        self.__read_filepath = os.path.join(
            self.LARGE_FILE_DIRECTORY_READ_PATH, self.__filename
        )
        self.__read_filepath_sis = os.path.join(
            self.LARGE_FILE_DIRECTORY_READ_PATH, self.__filename_sis
        )
        self.__write_filepath = os.path.join(
            self.LARGE_FILE_DIRECTORY_WRITE_PATH, self.__filename
        )
        self.__write_filepath_sis = os.path.join(
            self.LARGE_FILE_DIRECTORY_WRITE_PATH, self.__filename_sis
        )

        self.__filestore_resource, datum_factory_z = resource_factory(
            "ZEBRA_HDF51",
            root="/",
            resource_path=self.__read_filepath,
            resource_kwargs={},
            path_semantics="posix",
        )
        self.__filestore_resource_sis, datum_factory_sis = resource_factory(
            "SIS_HDF51",
            root="/",
            resource_path=self.__read_filepath_sis,
            resource_kwargs={},
            path_semantics="posix",
        )

        time_datum = datum_factory_z({"column": "time"})
        enc1_datum = datum_factory_z({"column": "enc1"})
        enc2_datum = datum_factory_z({"column": "enc2"})
        sis_datum = datum_factory_sis({"column": "i0"})
        sis_datum_im = datum_factory_sis({"column": "im"})
        sis_datum_it = datum_factory_sis({"column": "it"})
        sis_time = datum_factory_sis({"column": "time"})

        self._document_cache.extend(
            ("resource", d)
            for d in (self.__filestore_resource, self.__filestore_resource_sis)
        )
        self._document_cache.extend(
            ("datum", d)
            for d in (
                time_datum,
                enc1_datum,
                enc2_datum,
                sis_datum,
                sis_time,
                sis_datum_im,
                sis_datum_it,
            )
        )

        # grab the asset documents from all of the child detectors
        for d in self._dets:
            self._document_cache.extend(d.collect_asset_docs())

        # Write the file.
        # @timer_wrapper
        def get_zebra_data():
            export_zebra_data(
                self._encoder, self.__write_filepath, self.fast_axis
            )

        # t_getzebradata = tic()
        get_zebra_data()
        # toc(t_getzebradata, str='Get Zebra data')

        # @timer_wrapper
        def get_sis_data():
            export_sis_data(
                self._sis, self.__write_filepath_sis, self._encoder
            )

        # t_sisdata = tic()
        get_sis_data()
        # toc(t_sisdata, str='Get SIS data')

        # Yield a (partial) Event document. The RunEngine will put this
        # into metadatastore, as it does all readings.
        self._last_bulk = {
            "time": ttime.time(),
            "seq_num": 1,
            "data": {
                "time": time_datum["datum_id"],
                "enc1": enc1_datum["datum_id"],
                "enc2": enc2_datum["datum_id"],
                "i0": sis_datum["datum_id"],
                "i0_time": sis_time["datum_id"],
                "im": sis_datum_im["datum_id"],
                "it": sis_datum_it["datum_id"],
            },
            "timestamps": {
                "time": time_datum["datum_id"],  # not a typo#
                "enc1": time_datum["datum_id"],
                "enc2": time_datum["datum_id"],
                "i0": sis_time["datum_id"],
                "i0_time": sis_time["datum_id"],
                "im": sis_datum_im["datum_id"],
                "it": sis_datum_it["datum_id"],
            },
        }
        for d in self._dets:
            reading = d.read()
            self._last_bulk["data"].update(
                {k: v["value"] for k, v in reading.items()}
                )
            self._last_bulk["timestamps"].update(
                {k: v["timestamp"] for k, v in reading.items()}
            )

        return NullStatus()

    def collect(self):
        # Create records in the FileStore database.
        # move this to stage because I thinkt hat describe_collect needs the
        # resource id
        # TODO use ophyd.areadectector.filestoer_mixins.resllource_factory here
        if self._last_bulk is None:
            raise Exception(
                "the order of complete and collect is brittle and out "
                "of sync. This device relies on in-order and 1:1 calls "
                "between complete and collect to correctly create and stash "
                "the asset registry documents"
            )
        yield self._last_bulk
        self._last_bulk = None
        self._mode = "idle"

    def collect_asset_docs(self):
        yield from iter(list(self._document_cache))
        self._document_cache.clear()

    def stop(self):
        self._encoder.pc.block_state_reset.put(1)
        pass

    def pause(self):
        "Pausing in the middle of a kickoff nukes the partial dataset."
        self._encoder.pc.block_state_reset.put(1)
        self._sis.stop_all.put(1)
        for d in self._dets:
            if hasattr(d, "settings"):
                d.settings.acquire.put(0)
            if hasattr(d, "cam"):
                d.cam.acquire.put(0)
        self._mode = "idle"
        self.unstage()

    def resume(self):
        self.unstage()
        self.stage()


try:
    # flying_zebra = SRXFlyer1Axis(
    #     zebra,
    #     list(xs for xs in [xs] if xs is not None),
    #     sclr1,
    #     "HOR",
    #     name="flying_zebra",
    # )
    # flying_zebra_y = SRXFlyer1Axis(
    #   zebra, [xs], sclr1, "VER", name="flying_zebra_y"
    # )
    flying_zebra = SRXFlyer1Axis(
        list(xs for xs in [xs] if xs is not None), sclr1, name="flying_zebra"
    )
except Exception as ex:
    print("Cannot connect to Zebra. Continuing without device.\n", ex)
    flying_zebra = None
    # flying_zebra_y = None


# For confocal
# For plans that call xs2,
# should we simply add xs2 to flying_zebra.dets
# and set dir to 'DET2HOR'?
if xs2 is not None:
    # flying_zebra_x_xs2 = SRXFlyer1Axis(
    #     zebra, [xs2], sclr1, "HOR", name="flying_zebra_x_xs2"
    # )
    # flying_zebra_y_xs2 = SRXFlyer1Axis(
    #     zebra, [xs2], sclr1, "VER", name="flying_zebra_y_xs2"
    # )
    flying_zebra_xs2 = SRXFlyer1Axis(
        list(xs2 for xs2 in [xs2] if xs2 is not None),
        sclr1,
        name="flying_zebra_xs2"
    )

else:
    flying_zebra_xs2 = None
    # flying_zebra_y_xs2 = None
# For chip imaging
# flying_zebra_x_xs2 = SRXFlyer1Axis(
#   zebra, xs2, sclr1, 'DET2HOR', name='flying_zebra'
# )
# flying_zebra_y_xs2 = SRXFlyer1Axis(
#   zebra, xs2, sclr1, 'DET2VER', name='flying_zebra'
# )
# flying_zebra = SRXFlyer1Axis(zebra)


def export_zebra_data(zebra, filepath, fast_axis):
    j = 0
    while zebra.pc.data_in_progress.get() == 1:
        print("waiting zebra")
        ttime.sleep(0.1)
        j += 1
        if j > 10:
            print("THE ZEBRA IS BEHAVING BADLY CARRYING ON")
            break

    time_d = zebra.pc.data.time.get()
    if fast_axis == "HOR":
        enc1_d = zebra.pc.data.enc2.get()
        enc2_d = zebra.pc.data.enc1.get()
    elif fast_axis == "DET2HOR":
        enc1_d = zebra.pc.data.enc3.get()
    elif fast_axis == "DET2VER":
        enc1_d = zebra.pc.data.enc4.get()
    else:
        enc1_d = zebra.pc.data.enc1.get()
        enc2_d = zebra.pc.data.enc2.get()

    while len(time_d) == 0 or len(time_d) != len(enc1_d):
        time_d = zebra.pc.data.time.get()
        if fast_axis == "HOR":
            enc1_d = zebra.pc.data.enc2.get()
        else:
            enc1_d = zebra.pc.data.enc1.get()

    size = (len(time_d),)
    with h5py.File(filepath, "w") as f:
        dset0 = f.create_dataset("time", size, dtype="f")
        dset0[...] = np.array(time_d)
        dset1 = f.create_dataset("enc1", size, dtype="f")
        dset1[...] = np.array(enc1_d)
        dset2 = f.create_dataset("enc2", size, dtype="f")
        dset2[...] = np.array(enc2_d)
        f.close()


class ZebraHDF5Handler(HandlerBase):
    HANDLER_NAME = "ZEBRA_HDF51"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self, *, column):
        return self._handle[column][:]


db.reg.register_handler("ZEBRA_HDF51", ZebraHDF5Handler, overwrite=True)
