import nslsii


nslsii.configure_base(get_ipython().user_ns, 'srx')
nslsii.configure_olog(get_ipython().user_ns)

#Optional: set any metadata that rarely changes.
RE.md['beamline_id'] = 'SRX'

# Custom Matplotlib configs:
import matplotlib as mpl
mpl.rcParams['axes.grid'] = True      #grid always on

# Uncomment the following lines to turn on verbose messages for
# debugging.
# import logging
# ophyd.logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

from ophyd import PseudoSingle, PseudoPositioner, Signal

from IPython.terminal.prompts import Prompts, Token
class SRXPrompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        return [(Token.Prompt, 'BlueSky@SRX ['),
                (Token.PromptNum, str(self.shell.execution_count)),
                (Token.Prompt, ']: ')]

ip = get_ipython()
ip.prompts = SRXPrompt(ip)
