# Make ophyd listen to pyepics.
from ophyd import setup_ophyd
setup_ophyd()

# Make plots update live while scans run.
from bluesky.utils import install_qt_kicker
install_qt_kicker()
from metadatastore.mds import MDS
# from metadataclient.mds import MDS
from databroker import Broker
from databroker.core import register_builtin_handlers
from filestore.fs import FileStore

# pull from /etc/metadatastore/connection.yaml
mds = MDS({'host': 'xf05id-ca1',
           'database': 'datastore',
           'port': 27017,
           'timezone': 'US/Eastern'}, auth=False)
# mds = MDS({'host': CA, 'port': 7770})

# pull configuration from /etc/filestore/connection.yaml
db = Broker(mds, FileStore({'host': 'xf05id-ca1',
                            'database': 'filestore',
                            'port': 27017,
                            'timezone': 'US/Eastern',
                            }))
register_builtin_handlers(db.fs)

# Subscribe metadatastore to documents.
# If this is removed, data is not saved to metadatastore.

from bluesky.global_state import gs
gs.RE.subscribe_lossless('all', mds.insert)

# convenience imports
from ophyd.commands import *
from bluesky.callbacks import *
from bluesky.spec_api import *
from bluesky.global_state import gs, abort, stop, resume
from time import sleep
import numpy as np

RE = gs.RE  # convenience alias
gs.RE.md['beamline_id'] = 'xf05id'
gs.RE.record_interruptions = True

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
