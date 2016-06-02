import matplotlib.widgets as mwidgets
import matplotlib.dates as mdates
from collections import defaultdict
from cycler import cycler as cy
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
   
class ROIPlanCreator(object):
    def __init__(self, ax, step_size, plan_kwargs=None):
        if plan_kwargs is None:
            plan_kwargs = {}
        self.plan_kwargs = plan_kwargs
        self.step_size = step_size
        self.ax = ax
        self.widget = mwidgets.RectangleSelector(
            self.ax, self._onselect, useblit=True, interactive=True)
        self._pt1 = self._pt2 = None
    
    def _onselect(self, pt1, pt2):
        print('triggered')
        self._pt1 = pt1
        self._pt2 = pt2

    @property
    def last_plan(self):
        # TODO deal with ordering issues
        x1, y1 = self._pt1.xdata, self._pt1.ydata
        x2, y2 = self._pt2.xdata, self._pt2.ydata
        dx = x2 - x1
        dy = y2 - y1
        
        inp = dict(self.plan_kwargs)
        inp.update({'xstart': x1, 'xnumstep': max(dx // self.step_size, 1),
               'ystart': y1, 'ynumstep': max(dy // self.step_size, 1),
               'xstepsize': self.step_size,
               'ystepsize': self.step_size})
        print(inp)
        return hf2dxrf(**inp)
        
def plot_crab(hdr):
    fig, ax = plt.subplots()
    tab = db.get_table(hdr)
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
