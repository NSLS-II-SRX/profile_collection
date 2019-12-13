print(f'Loading {__file__}...')


from ophyd.areadetector import (AreaDetector, ImagePlugin,
                                TIFFPlugin, StatsPlugin, 
                                ROIPlugin, TransformPlugin,
                                OverlayPlugin)
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreTIFF)
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.cam import AreaDetectorCam


### Triggers for HFM/BPM1/HF VLM cameras
hfm_cam = EpicsSignal('XF:05IDA-BI:1{FS:1-Cam:1}Acquire_RBV',
                        write_pv='XF:05IDA-BI:1{FS:1-Cam:1}Acquire',
                        name='hfm_cam_trigger')
hfm_tot1 = EpicsSignal('XF:05IDA-BI:1{FS:1-Cam:1}Stats1:Total_RBV',
                        name='hfm_tot1')
bpm1_cam = EpicsSignal('XF:05IDA-BI:1{BPM:1-Cam:1}Acquire_RBV',
                        write_pv='XF:05IDA-BI:1{Mir:1-Cam:1}Acquire',
                        name='hfm_cam_trigger')
bpm1_tot1 = EpicsSignal('XF:05IDA-BI:1{BPM:1-Cam:1}Stats1:Total_RBV',
                         name='bpm1_tot1')
hfvlm_cam = EpicsSignal('XF:05IDD-BI:1{Mscp:1-Cam:1}cam1:Acquire',
                        write_pv='XF:05IDD-BI:1{Mscp:1-Cam:1}cam1:Acquire',
                        name='hfvlm_cam_trigger')
hfvlm_cam_tiff = EpicsSignal('XF:05IDD-BI:1{Mscp:1-Cam:1}TIFF1:Capture',
                        write_pv='XF:05IDD-BI:1{Mscp:1-Cam:1}TIFF1:Capture',
                        name='hfvlm_cam_tiff')


### BPM Camera
class SRXTIFFPlugin(TIFFPlugin, FileStoreTIFF,
                    FileStoreIterativeWrite):
    file_number_sync = None


class BPMCam(SingleTrigger, AreaDetector):
    cam = C(AreaDetectorCam, '')
    image_plugin = C(ImagePlugin, 'image1:')

    # tiff = C(SRXTIFFPlugin, 'TIFF1:',
    #          #write_path_template='/epicsdata/bpm1-cam1/2016/2/24/')
    #          #write_path_template='/epicsdata/bpm1-cam1/%Y/%m/%d/',
    #          #root='/epicsdata', reg=db.reg)
    #          write_path_template='/nsls2/xf05id1/data/bpm1-cam1/%Y/%m/%d/',
    #          root='/nsls2/xf05id1')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    # this is flakey?
    # stats5 = C(StatsPlugin, 'Stats5:')
    # pass  # is this needed?


bpmAD = BPMCam('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpmAD', read_attrs=[])
bpmAD.read_attrs = ['stats1', 'stats2', 'stats3', 'stats4']
bpmAD.stats1.read_attrs = ['total']
bpmAD.stats2.read_attrs = ['total']
bpmAD.stats3.read_attrs = ['total']
bpmAD.stats4.read_attrs = ['total']


### HF VLM
# Does this belong here or in microES?
class SRXHFVLMCam(SingleTrigger, AreaDetector):
    cam = C(AreaDetectorCam, '')
    image_plugin = C(ImagePlugin, 'image1:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    over1 = C(OverlayPlugin, 'Over1:')
    trans1 = C(TransformPlugin, 'Trans1:')
    tiff = C(SRXTIFFPlugin, 'TIFF1:',
             #write_path_template='/epicsdata/hfvlm/%Y/%m/%d/',
             #root='/epicsdata',
             write_path_template='/nsls2/xf05id1/data/hfvlm/%Y/%m/%d/',
             root='/nsls2/xf05id1')

hfvlmAD = SRXHFVLMCam('XF:05IDD-BI:1{Mscp:1-Cam:1}', name='hfvlm', read_attrs=['tiff'])
hfvlmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
hfvlmAD.tiff.read_attrs = []
hfvlmAD.stats1.read_attrs = ['total']
hfvlmAD.stats2.read_attrs = ['total']
hfvlmAD.stats3.read_attrs = ['total']
hfvlmAD.stats4.read_attrs = ['total']

