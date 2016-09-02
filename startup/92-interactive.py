import matplotlib.widgets as mwidgets
import matplotlib.dates as mdates
from collections import defaultdict
from cycler import cycler as cy
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import bluesky.interactive as bsi   
        
def plot_crab(hdr):
    fig, ax = plt.subplots()
    tab = db.get_table(hdr, stream_name='primary')
    cols = tab.columns
    readbacks = [k for k in cols if 'readback' in k and 'elevation' not in k]
    setpoint = [k for k in cols if 'setpoint' in k and 'elevation' not in k]
    cyl = cy('color', ['r', 'g', 'b', 'k'])
    finite_cy_iter = iter(cyl)
    dd = defaultdict(lambda : next(finite_cy_iter))
    side_map = {'upper': 'r', 'lower': 'b'}
    end_map = {'ds': 2, 'us': 3}
    val_map = {'setpoint': ':', 'readback':'-'}

    for nm in setpoint + readbacks:
        _, end, side, val = mtr = nm.rsplit('_')
        ax.plot('time', nm, data=tab, color=side_map[side], 
                lw=end_map[end], ls=val_map[val])
    end_h = [mpatches.Patch(color=v, label=k) for k,v in side_map.items()]
    side_h = [mlines.Line2D([],[], color='k', lw=v, label=k)
           for k,v in end_map.items()]
    val_h = [mlines.Line2D([],[], color='k', ls=v, label=k)
           for k,v in val_map.items()]
    ax.legend(handles=end_h + side_h + val_h, loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))


def display_vlm(ax, image_plugin, stage, overlay_read, pixel_scale):
     im_sz = image_plugin.array_size.get()[:2]
     vlm = image_plugin.array_data.get().reshape(im_sz)

     x = stage.x.position
     y = stage.y.position

     bp_x = overlay_read.position_x.get()
     bp_y = overlay_read.position_y.get()

     left_edge = x - bp_x * pixel_scale
     right_edge = x + (im_sz[1] - bp_x) * pixel_scale

     top_edge = y - bp_y * pixel_scale
     bottom_edge = y + (im_sz[0] - bp_y) * pixel_scale

     ax.imshow(vlm, cmap='gray', interpolation='none',
               extent=[left_edge, right_edge, bottom_edge, top_edge])
     ax.axhline(y)
     ax.axvline(x)

class SRXPlanner(bsi.OuterProductWidget):
    def __init__(self, *args, write_overlay, pixel_scale, stage, 
                 overlay_read, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_overlay = write_overlay
        self.pixel_scale = pixel_scale
        self.stage = stage
        self.overlay_read = overlay_read

    def _onselect(self, pt1, pt2):
        super()._onselect(pt1, pt2)
        x1, y1 = pt1.xdata, pt1.ydata
        x2, y2 = pt2.xdata, pt2.ydata

        x = self.stage.x.position
        y = self.stage.y.position

        bp_x = self.overlay_read.position_x.get()
        bp_y = self.overlay_read.position_y.get()

        left_edge = (x1 - x) / self.pixel_scale + bp_x
        width = (x2 - x1) / self.pixel_scale

        top_edge = (y1 - y) / self.pixel_scale + bp_y
        height = (y2 - y1) / self.pixel_scale

        self.write_overlay.position_x.put(left_edge)
        self.write_overlay.size_x.put(width)
        self.write_overlay.position_y.put(top_edge)
        self.write_overlay.size_y.put(height)


def attach_planner(ax, stage, write_overlay, pixel_scale, 
                   read_overlay):
    return SRXPlanner(ax, [], stage.x, stage.y, 10, 10, 
                      write_overlay=write_overlay,
                      pixel_scale=pixel_scale, stage=stage, 
                      overlay_read=read_overlay)
