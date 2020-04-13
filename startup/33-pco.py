print(f'Loading {__file__}...')


import sys
from ophyd.areadetector import AreaDetector, SingleTrigger, HDF5Plugin
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.plugins import ROIPlugin, ImagePlugin
from ophyd import Component as Cpt
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from nslsii.ad33 import StatsPluginV33


class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    file_number_sync = None

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()


class PCOEdgeCamV33(AreaDetectorCam):
    wait_for_plugins = Cpt(EpicsSignal, 'WaitForPlugins',
                           string=True, kind='config')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs['wait_for_plugins'] = 'Yes'

    def ensure_nonblocking(self):
        self.stage_sigs['wait_for_plugins'] = 'Yes'
        for c in self.parent.component_names:
            cpt = getattr(self.parent, c)
            if cpt is self:
                continue
            if hasattr(cpt, 'ensure_nonblocking'):
                cpt.ensure_nonblocking()


class SRXPCOEdge(SingleTrigger, AreaDetector):
    cam = Cpt(PCOEdgeCamV33, 'cam1:')
    image_plugin = Cpt(ImagePlugin, 'image1:')
    stats1 = Cpt(StatsPluginV33, 'Stats1:')
    stats2 = Cpt(StatsPluginV33, 'Stats2:')
    stats3 = Cpt(StatsPluginV33, 'Stats3:')
    stats4 = Cpt(StatsPluginV33, 'Stats4:')
    stats5 = Cpt(StatsPluginV33, 'Stats5:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    hdf = Cpt(HDF5PluginWithFileStore,
              'HDF1:',
              read_path_template=r'/nsls2/xf05id1/XF05ID1/PCO/%Y/%m/%d/',
              write_path_template=r'Z:\%Y\%m\%d\\',
              root='/nsls2/xf05id1/XF05ID1/',
              )


try:
    pcoedge = SRXPCOEdge('XF05IDD-ES{PCO:1}:', name='pcoedge')
    pcoedge.cam.ensure_nonblocking()
except TimeoutError:
    print('\nCannot connect to PCO Edge. Continuing without device.\n')
except Exception:
    print('\nUnexpected error connecting to PCO Edge.\n',
          sys.exc_info()[0],
          end='\n\n')
