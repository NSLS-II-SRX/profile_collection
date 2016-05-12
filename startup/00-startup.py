# Make ophyd listen to pyepics.
from ophyd import setup_ophyd
setup_ophyd()

# Make plots update live while scans run.
from bluesky.utils import install_qt_kicker
install_qt_kicker()


# Subscribe metadatastore to documents.
# If this is removed, data is not saved to metadatastore.
import metadatastore.commands
from bluesky.global_state import gs
gs.RE.subscribe_lossless('all', metadatastore.commands.insert)

# convenience imports
from ophyd.commands import *
from bluesky.callbacks import *
from bluesky.spec_api import *
from bluesky.global_state import gs, abort, stop, resume
from databroker import (DataBroker as db, get_events, get_images,
                        get_table, get_fields, restream, process)
from time import sleep
import numpy as np

RE = gs.RE  # convenience alias
gs.RE.md['beamline_id'] = 'xf05id'


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
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


from ophyd import PseudoSingle, PseudoPositioner, Signal


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

