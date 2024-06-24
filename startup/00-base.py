print(f"Loading {__file__}...")

import copy
import os
import time as ttime
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import nslsii
import redis
from bluesky_queueserver import is_re_worker_active, parameter_annotation_decorator
from IPython.terminal.prompts import Prompts, Token
from ophyd.signal import DEFAULT_CONNECTION_TIMEOUT, EpicsSignal, EpicsSignalBase
from redis_json_dict import RedisJSONDict
from tiled.client import from_profile, from_uri


def if_touch_beamline(envvar="TOUCHBEAMLINE"):
    value = os.environ.get(envvar, "false").lower()
    if value in ("", "n", "no", "f", "false", "off", "0"):
        return False
    elif value in ("y", "yes", "t", "true", "on", "1"):
        return True
    else:
        raise ValueError(f"Unknown value: {value}")


def print_now():
    return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")


def wait_for_connection_base(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    """Wait for the underlying signals to initialize or connect"""
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = ttime.time()
    try:
        self._ensure_connected(self._read_pv, timeout=timeout)
        # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')
    except TimeoutError:
        if self._destroyed:
            raise DestroyedError("Signal has been destroyed")
        raise


def wait_for_connection(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    """Wait for the underlying signals to initialize or connect"""
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = ttime.time()
    self._ensure_connected(self._read_pv, self._write_pv, timeout=timeout)
    # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')


EpicsSignalBase.wait_for_connection = wait_for_connection_base
EpicsSignal.wait_for_connection = wait_for_connection
###############################################################################

if if_touch_beamline():
    # Case of real beamline:
    timeout = 2  # seconds
    going = "Going"
else:
    # Case of CI:
    timeout = 10  # seconds
    going = "NOT going"

print(f"\nEpicsSignalBase timeout is {timeout} [seconds]. {going} to touch beamline hardware.\n")

# EpicsSignalBase.set_default_timeout(timeout=timeout, connection_timeout=timeout)  # old style
EpicsSignalBase.set_defaults(timeout=timeout, connection_timeout=timeout)  # new style


ip = get_ipython()
nslsii.configure_base(
    ip.user_ns,
    "srx",
    publish_documents_with_kafka=True,
)

RE.unsubscribe(0)

# Define tiled catalog
srx_raw = from_profile("nsls2", api_key=os.environ["TILED_BLUESKY_WRITING_API_KEY_SRX"])["srx"]["raw"]
c = srx_raw

discard_liveplot_data = True
descriptor_uids = []

def post_document(name, doc):
    if name == "start":
        doc = copy.deepcopy(doc)
        descriptor_uids.clear()

    if name == "descriptor":
        if discard_liveplot_data and doc["name"].startswith("DONOTSAVE_"):
            descriptor_uids.append(doc["uid"])
            return
    elif name == "event_page" and doc["descriptor"] in descriptor_uids:
        return
    # print(f"==================  name={name!r} doc={doc} type(doc)={type(doc)}")
    ATTEMPTS = 20
    error = None
    for attempt in range(ATTEMPTS):
        try:
            srx_raw.post_document(name, doc)
        except Exception as exc:
            print("Document saving failure:", repr(exc))
            error = exc
        else:
            break
        time.sleep(2)
    else:
        # Out of attempts
        raise error


RE.subscribe(post_document)

ip.log.setLevel("WARNING")

nslsii.configure_olog(ip.user_ns)

# Custom Matplotlib configs:
mpl.rcParams["axes.grid"] = True  # grid always on


# Comment it out to enable BEC table:
bec.disable_table()


# Disable BestEffortCallback to plot ring current
bec.disable_plots()

RE.md = RedisJSONDict(redis.Redis("info.srx.nsls2.bnl.gov"), prefix="")

# Optional: set any metadata that rarely changes.
# RE.md["beamline_id"] = "SRX"
# RE.md["md_version"] = "1.1"


class SRXPrompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        return [
            (
                Token.Prompt,
                f"BlueSky@SRX | Proposal #{RE.md.get('proposal', {}).get('proposal_id', 'N/A')} [",
            ),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]


ip.prompts = SRXPrompt(ip)

# from bluesky.utils import ts_msg_hook
# RE.msg_hook = ts_msg_hook

# The following plan stubs are automatically imported in global namespace by 'nslsii.configure_base',
# but have signatures that are not compatible with the Queue Server. They should not exist in the global
# namespace, but can be accessed as 'bps.one_1d_step' etc. from other plans.
del one_1d_step, one_nd_step, one_shot
