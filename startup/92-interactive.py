import matplotlib.widgets as mwidgets

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
        
    
