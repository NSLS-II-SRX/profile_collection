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


### BPM Camera
class SRXTIFFPlugin(TIFFPlugin, FileStoreTIFF,
                    FileStoreIterativeWrite):
    file_number_sync = None


class BPMCam(SingleTrigger, AreaDetector):
    cam = Cpt(AreaDetectorCam, '')
    image_plugin = Cpt(ImagePlugin, 'image1:')

    # tiff = C(SRXTIFFPlugin, 'TIFF1:',
    #          #write_path_template='/epicsdata/bpm1-cam1/2016/2/24/')
    #          #write_path_template='/epicsdata/bpm1-cam1/%Y/%m/%d/',
    #          #root='/epicsdata', reg=db.reg)
    #          write_path_template='/nsls2/xf05id1/data/bpm1-cam1/%Y/%m/%d/',
    #          root='/nsls2/xf05id1')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    # this is flakey?
    # stats5 = C(StatsPlugin, 'Stats5:')


bpmAD = BPMCam('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpmAD', read_attrs=[])
bpmAD.read_attrs = ['stats1', 'stats2', 'stats3', 'stats4']
bpmAD.stats1.read_attrs = ['total']
bpmAD.stats2.read_attrs = ['total']
bpmAD.stats3.read_attrs = ['total']
bpmAD.stats4.read_attrs = ['total']


### HF VLM
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
               # write_path_template='/nsls2/xf05id1/data/hfvlm/%Y/%m/%d/',
               # root='/nsls2/xf05id1')

try:
    hfvlmAD = SRXHFVLMCam('XF:05IDD-BI:1{Mscp:1-Cam:1}', name='hfvlm', read_attrs=['tiff'])
    hfvlmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
    hfvlmAD.tiff.read_attrs = []
    hfvlmAD.stats1.read_attrs = ['total']
    hfvlmAD.stats2.read_attrs = ['total']
    hfvlmAD.stats3.read_attrs = ['total']
    hfvlmAD.stats4.read_attrs = ['total']
except TimeoutError:
    hfvlmAD = None
    print('\nCannot connect to HF VLM Camera. Continuing without device.\n')
except:
    hfvlmAD = None
    print('\nUnexpected error connecting to HF VLM Camera.\n', sys.exc_info()[0], end='\n\n')

