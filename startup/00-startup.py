from bluesky.standard_config import *
from bluesky.global_state import *
from bluesky import qt_kicker
from bluesky.spec_api import *
from ophyd import PseudoPositioner, PseudoSingle, Signal
import matplotlib.pyplot as plt
import ophyd
from ophyd.commands import *  # imports mov, wh_pos, etc.

qt_kicker.install_qt_kicker()
gs.RE.md['beamline_id'] = 'xf05id'
RE = gs.RE

ophyd.utils.startup.setup()

# import matplotlib and put it in interactive mode.

plt.ion()
#
# # Uncomment the following lines to turn on verbose messages for debugging.
# # import logging
# # ophyd.logger.setLevel(logging.DEBUG)
# # logging.basicConfig(level=logging.DEBUG)

def relabel_motors(dev):
    for chld in dev.signal_names:
        obj = getattr(dev, chld)
        if hasattr(obj, 'user_readback'):
            getattr(obj, 'user_readback').name = obj.name



class FixedPseudoSingle(PseudoSingle):
    """Adds missing methods

    This will need to be removed when Positioner is fixed upstream
    """
    def read(self):
        return {self.name: {'value': self.position,
                            'timestamp': ttime.time()}}
    
    def describe(self):
        return {self.name: {'dtype': 'number',
                            'shape': [],
                            'source': 'computed',
                            'units': 'keV'}}

    def read_configuration(self):
        return {}
    
    def describe_configuration(self):
        return {}
                                

class MagicSetPseudoPositioner(PseudoPositioner):
    def set(self, *args):
        v = self.PseudoPosition(*args)
        return super().set(v)


class PermissiveGetSignal(Signal):
    def get(self, use_monitor=None):
        return super().get()

