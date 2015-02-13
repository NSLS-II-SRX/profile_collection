import logging
session_mgr._logger.setLevel(logging.CRITICAL)
from ophyd.userapi import *
from metadataStore import conf as mds_cnf
mds_cnf.mds_config['host'] = 'xf05id-ca1'
