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


# Optional: set any metadata that rarely changes.
RE.md["beamline_id"] = "SRX"


# Custom Matplotlib configs:
mpl.rcParams["axes.grid"] = True  # grid always on


# Comment it out to enable BEC table:
bec.disable_table()


# Disable BestEffortCallback to plot ring current
bec.disable_plots()


# Uncomment the following lines to turn on verbose messages for
# debugging.
# import logging
# ophyd.logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)
