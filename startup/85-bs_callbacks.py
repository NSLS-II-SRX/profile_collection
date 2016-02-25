# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:30:06 2016

@author: xf05id1
"""

from bluesky.callbacks import LivePlot
i0_baseline = 1.53e-7

class NormalizeLivePlot(LivePlot):
    def __init__(self, *args, norm_key=None,  **kwargs):
        super().__init__(*args, **kwargs)
        if norm_key is None:
            raise RuntimeError("norm key is required kwarg")
        self._norm = norm_key
      
        
    def event(self, doc):
        "Update line with data from this Event."
        try:
            if self.x is not None:
                # this try/except block is needed because multiple event streams
                # will be emitted by the RunEngine and not all event streams will
                # have the keys we want
                new_x = doc['data'][self.x]
            else:
                new_x = doc['seq_num']
            new_y = doc['data'][self.y]
            new_norm = doc['data'][self._norm]
        except KeyError:
            # wrong event stream, skip it
            return
        self.y_data.append(new_y / abs(new_norm-i0_baseline))
        self.x_data.append(new_x)
        self.current_line.set_data(self.x_data, self.y_data)
        # Rescale and redraw.
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()