print(f'Loading {__file__}...')


import sys
from ophyd.areadetector import (AreaDetector, ImagePlugin,
                                TIFFPlugin, StatsPlugin,
                                ROIPlugin, TransformPlugin,
                                OverlayPlugin, ProcessPlugin)
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreTIFF)
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.device import Component as Cpt

from ophyd.areadetector.plugins import (ImagePlugin_V33, TIFFPlugin_V33,
                                        ROIPlugin_V33, StatsPlugin_V33)


# BPM Camera
class SRXTIFFPlugin(TIFFPlugin_V33,
                    FileStoreTIFF,
                    FileStoreIterativeWrite):
    ...


class SRXAreaDetectorCam(AreaDetectorCam):
    pool_max_buffers = None


class BPMCam(SingleTrigger, AreaDetector):
    cam = Cpt(SRXAreaDetectorCam, 'cam1:')
    image_plugin = Cpt(ImagePlugin_V33, 'image1:')

    tiff = Cpt(SRXTIFFPlugin, 'TIFF1:',
               write_path_template='/epicsdata/bpm1-cam1/2020/11/07/')
               #write_path_template='/epicsdata/bpm1-cam1/%Y/%m/%d/',
               #root='/epicsdata', reg=db.reg)
               # write_path_template='/nsls2/xf05id1/data/bpm1-cam1/%Y/%m/%d/',
               # root='/nsls2/xf05id1')
    roi1 = Cpt(ROIPlugin_V33, 'ROI1:')
    roi2 = Cpt(ROIPlugin_V33, 'ROI2:')
    roi3 = Cpt(ROIPlugin_V33, 'ROI3:')
    roi4 = Cpt(ROIPlugin_V33, 'ROI4:')
    stats1 = Cpt(StatsPlugin_V33, 'Stats1:')
    stats2 = Cpt(StatsPlugin_V33, 'Stats2:')
    stats3 = Cpt(StatsPlugin_V33, 'Stats3:')
    stats4 = Cpt(StatsPlugin_V33, 'Stats4:')
    # this is flakey?
    # stats5 = Cpt(StatsPlugin_V33, 'Stats5:')


bpmAD = BPMCam('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpmAD', read_attrs=[])
bpmAD.wait_for_connection()

bpmAD.read_attrs = ['stats1', 'stats2', 'stats3', 'stats4', 'tiff']
bpmAD.stats1.read_attrs = ['total']
bpmAD.stats2.read_attrs = ['total']
bpmAD.stats3.read_attrs = ['total']
bpmAD.stats4.read_attrs = ['total']


# HF VLM
# Does this belong here or in microES?
class SRXHFVLMCam(SingleTrigger, AreaDetector):
    cam = Cpt(AreaDetectorCam, 'cam1:')
    image_plugin = Cpt(ImagePlugin, 'image1:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    over1 = Cpt(OverlayPlugin, 'Over1:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    tiff = Cpt(SRXTIFFPlugin, 'TIFF1:',
               write_path_template='/epicsdata/hfvlm/%Y/%m/%d/',
               root='/epicsdata')


try:
    hfvlmAD = SRXHFVLMCam('XF:05IDD-BI:1{Mscp:1-Cam:1}',
                          name='hfvlm',
                          read_attrs=['tiff'])
    hfvlmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
    hfvlmAD.tiff.read_attrs = []
    hfvlmAD.stats1.read_attrs = ['total']
    hfvlmAD.stats2.read_attrs = ['total']
    hfvlmAD.stats3.read_attrs = ['total']
    hfvlmAD.stats4.read_attrs = ['total']
except TimeoutError:
    hfvlmAD = None
    print('\nCannot connect to HF VLM Camera. Continuing without device.\n')
except Exception as ex:
    hfvlmAD = None
    print('\nUnexpected error connecting to HF VLM Camera.\n')
    print(ex, end='\n\n')


# Transmission Camera
# Does this belong here or in microES?
class SRXCam05(SingleTrigger, AreaDetector):
    cam = Cpt(AreaDetectorCam, 'cam1:')
    image_plugin = Cpt(ImagePlugin, 'image1:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    over1 = Cpt(OverlayPlugin, 'Over1:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    tiff = Cpt(SRXTIFFPlugin, 'TIFF1:',
               write_path_template='/epicsdata/cam05/%Y/%m/%d/',
               root='/epicsdata')


try:
    cam05 = SRXCam05('XF:05IDD-BI:1{Cam:5}',
                          name='cam05',
                          read_attrs=['tiff'])
    cam05.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
    cam05.tiff.read_attrs = []
    cam05.stats1.read_attrs = ['total']
    cam05.stats2.read_attrs = ['total']
    cam05.stats3.read_attrs = ['total']
    cam05.stats4.read_attrs = ['total']
except TimeoutError:
    cam05 = None
    print('\nCannot connect to camera 5. Continuing without device.\n')
except Exception as ex:
    cam05 = None
    print('\nUnexpected error connecting to camera 5.\n')
    print(ex, end='\n\n')


# nano-VLM
class SRXnanoVLMCam(SingleTrigger, AreaDetector):
    cam = Cpt(AreaDetectorCam, 'cam1:')
    image_plugin = Cpt(ImagePlugin, 'image1:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    over1 = Cpt(OverlayPlugin, 'Over1:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    tiff = Cpt(SRXTIFFPlugin, 'TIFF1:',
               write_path_template='/nsls2/xf05id1/XF05ID1/nanoVLM/%Y/%m/%d/',
               read_path_template='/nsls2/xf05id1/XF05ID1/nanoVLM/%Y/%m/%d/',
               root='/nsls2/xf05id1/XF05ID1/nanoVLM')


try:
    nano_vlm = SRXnanoVLMCam('XF:05ID1-ES{PG-Cam:1}',
                          name='nano_vlm',
                          read_attrs=['tiff'])
    nano_vlm.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
    nano_vlm.tiff.read_attrs = []
    nano_vlm.stats1.read_attrs = ['total']
    nano_vlm.stats2.read_attrs = ['total']
    nano_vlm.stats3.read_attrs = ['total']
    nano_vlm.stats4.read_attrs = ['total']
except TimeoutError:
    nano_vlm = None
    print('\nCannot connect to nanoVLM Camera. Continuing without device.\n')
except Exception as ex:
    nano_vlm = None
    print('\nUnexpected error connecting to nanoVLM Camera.\n')
    print(ex, end='\n\n')
