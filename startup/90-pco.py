from ophyd.areadetector import AreaDetector, SingleTrigger, HDF5Plugin
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.plugins import ROIPlugin, StatsPlugin, ImagePlugin
from ophyd import Component as C
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite

class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    file_number_sync = None

class SRXPCOEDGECam(SingleTrigger, AreaDetector):
    cam = C(AreaDetectorCam, 'cam1:')
    image_plugin = C(ImagePlugin, 'image1:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    stats5 = C(StatsPlugin, 'Stats5:')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    hdf = C(HDF5PluginWithFileStore, 'HDF1:',
            read_path_template='/data/PCOEDGE/%Y/%m/%d/',
            write_path_template='C:\epicsdata\pcoedge\debug_testing',
            root='/data')

pcoedge = SRXPCOEDGECam('XF05IDD-ES{PCO:1}:',name='pcoedge')