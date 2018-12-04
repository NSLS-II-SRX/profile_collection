import os
import ophyd
from hxntools.detectors.dexela import (HxnDexelaDetector, HDF5PluginWithFileStore, DexelaFileStoreHDF5)


# monkey patch for trailing slash problem (from XPD's profile, see
# https://github.com/NSLS-II-XPD/profile_collection/blob/master/startup/80-areadetector.py#L17-L32)
def _ensure_trailing_slash(path):
    """
    'a/b/c' -> 'a/b/c/'
    EPICS adds the trailing slash itself if we do not, so in order for the
    setpoint filepath to match the readback filepath, we need to add the
    trailing slash ourselves.
    """
    newpath = os.path.join(path, '')
    if newpath[0] != '/' and newpath[-1] == '/':
        # make it a windows slash
        newpath = newpath[:-1]
    return newpath

ophyd.areadetector.filestore_mixins._ensure_trailing_slash = _ensure_trailing_slash


class SRXHDF5PluginWithFileStore(HDF5PluginWithFileStore):
    def stage(self):
        ret = super().stage()
        self._generate_resource({'frame_per_point': 1})
        return ret


class SRXDexelaDetector(HxnDexelaDetector):
    hdf5 = Cpt(SRXHDF5PluginWithFileStore, 'HDF1:',
               read_attrs=[],
               configuration_attrs=[],
               write_path_template='Z:\\%Y\\%m\\%d\\',
               read_path_template='/nsls2/xf05id1/XF05ID1/dexela/%Y/%m/%d/',
               root='/nsls2/xf05id1/XF05ID1/dexela/')


del DexelaFileStoreHDF5.stage

dexela = SRXDexelaDetector('XF:05IDD-ES{Dexela:1}', name='dexela')
dexela.hdf5.filestore_spec = dexela.hdf5._spec
dexela.mode_settings.total_points.put(1)

