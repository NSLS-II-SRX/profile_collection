from ophyd.areadetector import AreaDetector, SingleTrigger, HDF5Plugin
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.plugins import ROIPlugin, StatsPlugin, ImagePlugin
from ophyd import Component as C
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite

from nslsii.ad33 import SingleTriggerV33, StatsPluginV33, CamV33Mixin

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
    

class SRXPCOEDGE(SingleTrigger, AreaDetector):
    cam = C(PCOEdgeCamV33, 'cam1:')
    image_plugin = C(ImagePlugin, 'image1:')
    stats1 = C(StatsPluginV33, 'Stats1:')
    stats2 = C(StatsPluginV33, 'Stats2:')
    stats3 = C(StatsPluginV33, 'Stats3:')
    stats4 = C(StatsPluginV33, 'Stats4:')
    stats5 = C(StatsPluginV33, 'Stats5:')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    hdf = C(HDF5PluginWithFileStore, 'HDF1:',
            read_path_template=r'/nsls2/xf05id1/XF05ID1/PCO/%Y/%m/%d/',
            write_path_template=r'Z:\%Y\%m\%d\\',
            root='/nsls2/xf05id1/XF05ID1/')


pcoedge = SRXPCOEDGE('XF05IDD-ES{PCO:1}:', name='pcoedge')
pcoedge.cam.ensure_nonblocking()
