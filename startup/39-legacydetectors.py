from ophyd.areadetector import (AreaDetector, PixiradDetectorCam, ImagePlugin,
                                TIFFPlugin, StatsPlugin, HDF5Plugin,
                                ProcessPlugin, ROIPlugin, TransformPlugin,
                                OverlayPlugin)
from ophyd.areadetector.plugins import PluginBase
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.device import BlueskyInterface
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.filestore_mixins import (FileStoreIterativeWrite,
                                                 FileStoreHDF5IterativeWrite,
                                                 FileStoreTIFFSquashing,
                                                 FileStoreTIFF,
                                                 FileStoreHDF5, new_short_uid,
                                                 FileStoreBase
                                                 )
from ophyd import Signal
from ophyd import Component as C
from hxntools.detectors.merlin import MerlinDetector
from hxntools.handlers import register
import itertools

register(db)


class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    file_number_sync = None

class FileStoreBulkReadable(FileStoreIterativeWrite):

    def _reset_data(self):
        self._datum_uids.clear()
        self._point_counter = itertools.count()

    def bulk_read(self, timestamps):
        # if you see this, delete this line but report to DAMA
        raise Exception("should not be here")
        image_name = self.image_name

        uids = [self.generate_datum(self.image_name, ts, {}) for ts in timestamps]

        #clear so unstage will not save the images twice:
        self._reset_data()
        return {image_name: uids}

    @property
    def image_name(self):
        return self.parent._image_name


class SRXPixirad(SingleTrigger,AreaDetector):

    det = C(PixiradDetectorCam, 'cam1:')
    image = C(ImagePlugin, 'image1:')
    roi1 = C(ROIPlugin, 'ROI1:')
    roi2 = C(ROIPlugin, 'ROI2:')
    roi3 = C(ROIPlugin, 'ROI3:')
    roi4 = C(ROIPlugin, 'ROI4:')
    stats1 = C(StatsPlugin, 'Stats1:')
    stats2 = C(StatsPlugin, 'Stats2:')
    stats3 = C(StatsPlugin, 'Stats3:')
    stats4 = C(StatsPlugin, 'Stats4:')
    tiff = C(SRXTIFFPlugin, 'TIFF1:',
             #write_path_template='/epicsdata/pixirad/%Y/%m/%d/',
             #root='/epicsdata')
             write_path_template='/nsls2/xf05id1/data/pixirad/%Y/%m/%d/',
             root='/nsls2/xf05id1')

#pixi = SRXPixirad('XF:05IDD-ES:1{Det:Pixi}', name='pixi', read_attrs=['stats1','stats2','stats3','stats4','tiff'])
#pixi.stats1.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.stats2.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.stats3.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.stats4.read_attrs = ['total','centroid','sigma_x','sigma_y']
#pixi.tiff.read_attrs = []

