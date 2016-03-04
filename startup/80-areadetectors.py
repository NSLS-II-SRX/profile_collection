from ophyd.areadetector import (AreaDetector, PixiradDetectorCam, ImagePlugin,
                                TIFFPlugin, StatsPlugin, HDF5Plugin,
                                ProcessPlugin)
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.device import BlueskyInterface
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreHDF5IterativeWrite,
                                                 FileStoreTIFFSquashing,
                                                 FileStoreTIFF)
from ophyd import Signal
from ophyd import Component as C

class SRXTIFFPlugin(TIFFPlugin, FileStoreTIFF,
                    FileStoreIterativeWrite):
    pass

class BPMCam(SingleTrigger, AreaDetector):
    cam = C(AreaDetectorCam, '')
    image_plugin = C(ImagePlugin, 'image1:')

    tiff = C(SRXTIFFPlugin, 'TIFF1:',
             write_path_template='/epicsdata/bpm1-cam1/2016/2/24/')
             #write_path_template='/epicsdata/bpm1-cam1/%Y/%m/%d/')

    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    # this is flakey?
    # stats5 = C(StatsPlugin, 'Stats5:')
    pass
                               
bpmAD = BPMCam('XF:05IDA-BI:1{BPM:1-Cam:1}', name='bpmAD', read_attrs=['tiff'])
bpmAD.read_attrs = ['tiff', 'stats1', 'stats2', 'stats3', 'stats4']
bpmAD.tiff.read_attrs = []
bpmAD.stats1.read_attrs = ['total']
bpmAD.stats2.read_attrs = ['total']
bpmAD.stats3.read_attrs = ['total']
bpmAD.stats4.read_attrs = ['total']

class SRXPixirad(SingleTrigger,AreaDetector):
    det = C(PixiradDetectorCam, 'cam1:')
    image = C(ImagePlugin, 'image1:')
    stats1 = C(StatsPlugin, 'Stats1:')
    hdf = C(HDF5Plugin, 'HDF1:')
    pass

pixi = SRXPixirad('XF:05IDD-ES:1{Det:Pixi}', name='pixi', read_attrs=['stats1','hdf'])
pixi.stats1.read_attrs = ['total']
