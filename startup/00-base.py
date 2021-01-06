print(f"Loading {__file__}...")

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
nslsii.configure_base(ip.user_ns, "srx")
nslsii.configure_olog(ip.user_ns)
ip.prompts = SRXPrompt(ip)



# Custom Matplotlib configs:
mpl.rcParams["axes.grid"] = True  # grid always on


# Comment it out to enable BEC table:
bec.disable_table()


# Disable BestEffortCallback to plot ring current
bec.disable_plots()


# Temporary fix before it's fixed in ophyd
import logging
logger = logging.getLogger('ophyd')
logger.setLevel('WARNING')

from pathlib import Path

import appdirs


try:
    from bluesky.utils import PersistentDict
except ImportError:
    import msgpack
    import msgpack_numpy
    import zict

    class PersistentDict(zict.Func):
        def __init__(self, directory):
            self._directory = directory
            self._file = zict.File(directory)
            super().__init__(self._dump, self._load, self._file)

        @property
        def directory(self):
            return self._directory

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

# runengine_metadata_dir = appdirs.user_data_dir(appname="bluesky") / Path("runengine-metadata")
runengine_metadata_dir = Path('/nsls2/xf05id1/shared/config/runengine-metadata')

RE.md = PersistentDict(runengine_metadata_dir)

# Optional: set any metadata that rarely changes.
RE.md["beamline_id"] = "SRX"
RE.md["md_version"] = "1.0"

