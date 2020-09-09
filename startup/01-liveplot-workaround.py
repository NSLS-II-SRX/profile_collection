from bluesky.callbacks.mpl_plotting import LivePlot, QtAwareCallback
from functools import partial
import matplotlib.pyplot as plt
import threading
from bluesky.callbacks.core import CallbackBase, get_obj_fields, make_class_safe
from ophyd.sim import det1, det2, motor

class HackLivePlot(LivePlot):
    """
    Build a function that updates a plot from a stream of Events.

    Note: If your figure blocks the main thread when you are trying to
    scan with this callback, call `plt.ion()` in your IPython session.

    Parameters
    ----------
    y : str
        the name of a data field in an Event
    x : str, optional
        the name of a data field in an Event, or 'seq_num' or 'time'
        If None, use the Event's sequence number.
        Special case: If the Event's data includes a key named 'seq_num' or
        'time', that takes precedence over the standard 'seq_num' and 'time'
        recorded in every Event.
    legend_keys : list, optional
        The list of keys to extract from the RunStart document and format
        in the legend of the plot. The legend will always show the
        scan_id followed by a colon ("1: ").  Each
    xlim : tuple, optional
        passed to Axes.set_xlim
    ylim : tuple, optional
        passed to Axes.set_ylim
    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.
    fig : Figure, optional
        deprecated: use ax instead
    epoch : {'run', 'unix'}, optional
        If 'run' t=0 is the time recorded in the RunStart document. If 'unix',
        t=0 is 1 Jan 1970 ("the UNIX epoch"). Default is 'run'.
    All additional keyword arguments are passed through to ``Axes.plot``.

    Examples
    --------
    >>> my_plotter = LivePlot('det', 'motor', legend_keys=['sample'])
    >>> RE(my_scan, my_plotter)
    """
    def __init__(self, y, x=None, *, legend_keys=None, xlim=None, ylim=None,
                 epoch='run', fig_factory=None, **kwargs):
        # don't use super to "skip" a level!
        QtAwareCallback.__init__(self, use_teleporter=kwargs.pop('use_teleporter', None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        def setup():
            # Run this code in start() so that it runs on the correct thread.
            nonlocal y, x, legend_keys, xlim, ylim, epoch, kwargs
            import matplotlib.pyplot as plt
            with self.__setup_lock:
                if self.__setup_event.is_set():
                    return
                self.__setup_event.set()
            if fig_factory is None:
                ax_factory = plt.subplots

            fig, ax  = fig_factory()

            self.ax = ax

            if legend_keys is None:
                legend_keys = []
            self.legend_keys = ['scan_id'] + legend_keys
            if x is not None:
                self.x, *others = get_obj_fields([x])
            else:
                self.x = 'seq_num'
            self.y, *others = get_obj_fields([y])
            self.ax.set_ylabel(y)
            self.ax.set_xlabel(x or 'sequence #')
            if xlim is not None:
                self.ax.set_xlim(*xlim)
            if ylim is not None:
                self.ax.set_ylim(*ylim)
            self.ax.margins(.1)
            self.kwargs = kwargs
            self.lines = []
            self.legend = None
            self.legend_title = " :: ".join([name for name in self.legend_keys])
            self._epoch_offset = None  # used if x == 'time'
            self._epoch = epoch

        self._LivePlot__setup = setup


class SRXJustPlotSomething(LivePlot):
    def __init__(self, y='', x=None, *, legend_keys=None, xlim=None, ylim=None,
                 ax=None, fig=None, epoch='run', **kwargs):
        super().__init__(y, use_teleporter=kwargs.pop('use_teleporter', None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        def setup():
            # Run this code in start() so that it runs on the correct thread.
            # nonlocal y, x, legend_keys, xlim, ylim, ax, fig, epoch, kwargs
            nonlocal legend_keys, xlim, ylim, ax, fig, epoch, kwargs
            import matplotlib.pyplot as plt
            with self.__setup_lock:
                if self.__setup_event.is_set():
                    return
                self.__setup_event.set()
            if fig is not None:
                if ax is not None:
                    raise ValueError("Values were given for both `fig` and `ax`. "
                                     "Only one can be used; prefer ax.")
                warnings.warn("The `fig` keyword arugment of LivePlot is "
                              "deprecated and will be removed in the future. "
                              "Instead, use the new keyword argument `ax` to "
                              "provide specific Axes to plot on.")
                ax = fig.gca()
            if ax is None:
                fig, ax = plt.subplots()
            self.ax = ax

            if legend_keys is None:
                legend_keys = []
            self.legend_keys = ['scan_id'] + legend_keys
            if x is not None:
                self.x, *others = get_obj_fields([x])
            else:
                self.x = 'seq_num'
            self.y, *others = get_obj_fields([y])
            self.ax.set_ylabel(y)
            self.ax.set_xlabel(x or 'sequence #')
            if xlim is not None:
                self.ax.set_xlim(*xlim)
            if ylim is not None:
                self.ax.set_ylim(*ylim)
            self.ax.margins(.1)
            self.kwargs = kwargs
            self.lines = []
            self.legend = None
            self.legend_title = " :: ".join([name for name in self.legend_keys])
            self._epoch_offset = None  # used if x == 'time'
            self._epoch = epoch

        self.__setup = setup


def test_fig():
    # x = np.linspace(0, 2*np.pi, num=100)
    # y = np.sin(x)
    # plotme = SRXJustPlotSomething(title='Fitting')
    # plotme.ax.plot(x, y, '*', label='Raw Data')
    # plotme.ax.set_title('Scan testing')
    # plotme.ax.legend()

    cb = SRXJustPlotSomething()

    @subs_decorator(cb)
    def _plan():
        yield from scan([det1, det2], motor, -5, 5, num=11)

    yield from _plan()

    h = db[-1]
    x = np.array(list(h.table()['motor']))
    y = np.array(list(h.table()['det2']))

    cb.ax.cla()
    cb.ax.plot(x, y)
    x = np.linspace(0, 2*np.pi, num=100)
    y = np.sin(x)
    cb.ax.plot(x, y)
    cb.ax.set_xlabel('X axis')
    cb.ax.set_ylabel('Y axis')

