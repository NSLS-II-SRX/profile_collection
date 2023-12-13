print(f"Loading {__file__}...")

import os
from datetime import datetime
from ophyd.signal import EpicsSignalBase, EpicsSignal, DEFAULT_CONNECTION_TIMEOUT
from bluesky_queueserver import is_re_worker_active, parameter_annotation_decorator

def if_touch_beamline(envvar="TOUCHBEAMLINE"):
    value = os.environ.get(envvar, "false").lower()
    if value in ("", "n", "no", "f", "false", "off", "0"):
        return False
    elif value in ("y", "yes", "t", "true", "on", "1"):
        return True
    else:
        raise ValueError(f"Unknown value: {value}")

def print_now():
    return datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S.%f')

def wait_for_connection_base(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    '''Wait for the underlying signals to initialize or connect'''
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = time.time()
    try:
        self._ensure_connected(self._read_pv, timeout=timeout)
        # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')
    except TimeoutError:
        if self._destroyed:
            raise DestroyedError('Signal has been destroyed')
        raise

def wait_for_connection(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    '''Wait for the underlying signals to initialize or connect'''
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = time.time()
    self._ensure_connected(self._read_pv, self._write_pv, timeout=timeout)
    # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')

EpicsSignalBase.wait_for_connection = wait_for_connection_base
EpicsSignal.wait_for_connection = wait_for_connection
###############################################################################

if if_touch_beamline():
    # Case of real beamline:
    timeout = 10  # seconds
    going = "Going"
else:
    # Case of CI:
    timeout = 10  # seconds
    going = "NOT going"

print(f'\nEpicsSignalBase timeout is {timeout} [seconds]. {going} to touch beamline hardware.\n')

from ophyd.signal import EpicsSignalBase
# EpicsSignalBase.set_default_timeout(timeout=timeout, connection_timeout=timeout)  # old style
EpicsSignalBase.set_defaults(timeout=timeout, connection_timeout=timeout)  # new style

import datetime

def print_now():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S.%f')

import nslsii
import matplotlib as mpl
from IPython.terminal.prompts import Prompts, Token


class SRXPrompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        return [
            (Token.Prompt, "BlueSky@SRX ["),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]

ip = get_ipython()
nslsii.configure_base(ip.user_ns,
                      "srx",
                      publish_documents_with_kafka=True)
ip.log.setLevel('WARNING')

nslsii.configure_olog(ip.user_ns)
ip.prompts = SRXPrompt(ip)



# Custom Matplotlib configs:
mpl.rcParams["axes.grid"] = True  # grid always on


# Comment it out to enable BEC table:
bec.disable_table()


# Disable BestEffortCallback to plot ring current
bec.disable_plots()

from pathlib import Path

import appdirs


try:
    from bluesky.utils import PersistentDict
except ImportError:
    import msgpack
    import msgpack_numpy
    import zict

    class PersistentDict(zict.Func):
        """
        A MutableMapping which syncs it contents to disk.
        The contents are stored as msgpack-serialized files, with one file per item
        in the mapping.
        Note that when an item is *mutated* it is not immediately synced:
        >>> d['sample'] = {"color": "red"}  # immediately synced
        >>> d['sample']['shape'] = 'bar'  # not immediately synced
        but that the full contents are synced to disk when the PersistentDict
        instance is garbage collected.
        """
        def __init__(self, directory):
            self._directory = directory
            self._file = zict.File(directory)
            self._cache = {}
            super().__init__(self._dump, self._load, self._file)
            self.reload()

            # Similar to flush() or _do_update(), but without reference to self
            # to avoid circular reference preventing collection.
            # NOTE: This still doesn't guarantee call on delete or gc.collect()!
            #       Explicitly call flush() if immediate write to disk required.
            def finalize(zfile, cache, dump):
                zfile.update((k, dump(v)) for k, v in cache.items())

            import weakref
            self._finalizer = weakref.finalize(
                self, finalize, self._file, self._cache, PersistentDict._dump)

        @property
        def directory(self):
            return self._directory

        def __setitem__(self, key, value):
            self._cache[key] = value
            super().__setitem__(key, value)

        def __getitem__(self, key):
            return self._cache[key]

        def __delitem__(self, key):
            del self._cache[key]
            super().__delitem__(key)

        def __repr__(self):
            return f"<{self.__class__.__name__} {dict(self)!r}>"

        @staticmethod
        def _dump(obj):
            "Encode as msgpack using numpy-aware encoder."
            # See https://github.com/msgpack/msgpack-python#string-and-binary-type
            # for more on use_bin_type.
            return msgpack.packb(
                obj,
                default=msgpack_numpy.encode,
                use_bin_type=True)

        @staticmethod
        def _load(file):
            return msgpack.unpackb(
                file,
                object_hook=msgpack_numpy.decode,
                raw=False)

        def flush(self):
            """Force a write of the current state to disk"""
            for k, v in self.items():
                super().__setitem__(k, v)

        def reload(self):
            """Force a reload from disk, overwriting current cache"""
            self._cache = dict(super().items())


# using appdirs line for xspress3 development on xf05id2-ws1
# do not commit this
# runengine_metadata_dir = appdirs.user_data_dir(appname="bluesky") / Path("runengine-metadata")
# runengine_metadata_dir = Path('/nsls2/xf05id1/shared/config/runengine-metadata-new')
runengine_metadata_dir = Path('/nsls2/data/srx/legacy/xf05id1/shared/config/runengine-metadata-new')

RE.md = PersistentDict(runengine_metadata_dir)

# Optional: set any metadata that rarely changes.
RE.md["beamline_id"] = "SRX"
RE.md["md_version"] = "1.1"

# from bluesky.utils import ts_msg_hook
# RE.msg_hook = ts_msg_hook

# The following plan stubs are automatically imported in global namespace by 'nslsii.configure_base', 
# but have signatures that are not compatible with the Queue Server. They should not exist in the global
# namespace, but can be accessed as 'bps.one_1d_step' etc. from other plans.
del one_1d_step, one_nd_step, one_shot
