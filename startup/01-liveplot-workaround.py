from bluesky.callbacks.mpl_plotting import LivePlot, QtAwareCallback
from functools import partial
import matplotlib.pyplot as plt
import threading
from bluesky.callbacks.core import CallbackBase, get_obj_fields, make_class_safe
from ophyd.sim import det1, det2, motor
import numpy as np


import logging
logger = logging.getLogger('bluesky')


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



@make_class_safe(logger=logger)
class SRX1DFlyerPlot(QtAwareCallback):
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
    def __init__(self, y, x=None, xstart=0, xstep=1, xlabel=None, *, legend_keys=None, xlim=None, ylim=None,
                 ax=None, fig=None, epoch='run', **kwargs):
        super().__init__(use_teleporter=kwargs.pop('use_teleporter', None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        self._xstart = xstart
        self._xstep = xstep
        self._xind = 0
        self._xlabel = xlabel

        def setup():
            # Run this code in start() so that it runs on the correct thread.
            nonlocal y, x, legend_keys, xlim, ylim, ax, fig, epoch, kwargs
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
            if (self._xlabel is None):
                self.ax.set_xlabel(x or 'sequence #')
            else:
                self.ax.set_xlabel(self._xlabel)
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

    def start(self, doc):
        self.__setup()
        # The doc is not used; we just use the signal that a new run began.
        self._epoch_offset = doc['time']  # used if self.x == 'time'
        self.x_data, self.y_data = [], []
        label = " :: ".join(
            [str(doc.get(name, name)) for name in self.legend_keys])
        kwargs = ChainMap(self.kwargs, {'label': label})
        self.current_line, = self.ax.plot([], [], **kwargs)
        self.lines.append(self.current_line)
        legend = self.ax.legend(loc=0, title=self.legend_title)
        try:
            # matplotlib v3.x
            self.legend = legend.set_draggable(True)
        except AttributeError:
            # matplotlib v2.x (warns in 3.x)
            self.legend = legend.draggable(True)
        super().start(doc)

    def event(self, doc):
        "Unpack data from the event and call self.update()."
        # This outer try/except block is needed because multiple event
        # streams will be emitted by the RunEngine and not all event
        # streams will have the keys we want.
        try:
            # This inner try/except block handles seq_num and time, which could
            # be keys in the data or accessing the standard entries in every
            # event.
            try:
                new_x = doc['data'][self.x]
            except KeyError:
                if self.x in ('time', 'seq_num'):
                    new_x = doc[self.x]
                else:
                    raise
            new_y = doc['data'][self.y]
        except KeyError:
            # wrong event stream, skip it
            return

        # Special-case 'time' to plot against against experiment epoch, not
        # UNIX epoch.
        if self.x == 'time' and self._epoch == 'run':
            new_x -= self._epoch_offset

        #overright the x value
        new_x = self._xstart + self._xstep * self._xind
        self._xind = self._xind + 1

        self.update_caches(new_x, new_y)
        self.update_plot()
        super().event(doc)

    def update_caches(self, x, y):
        self.y_data.append(y)
        self.x_data.append(x)

    def update_plot(self):
        self.current_line.set_data(self.x_data, self.y_data)
        # Rescale and redraw.
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()

    def stop(self, doc):
        if not self.x_data:
            print('LivePlot did not get any data that corresponds to the '
                  'x axis. {}'.format(self.x))
        if not self.y_data:
            print('LivePlot did not get any data that corresponds to the '
                  'y axis. {}'.format(self.y))
        if len(self.y_data) != len(self.x_data):
            print('LivePlot has a different number of elements for x ({}) and'
                  'y ({})'.format(len(self.x_data), len(self.y_data)))
        super().stop(doc)


@make_class_safe(logger=logger)
class SRX1DTSFlyerPlot(QtAwareCallback):
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

    def __init__(
        self,
        y,
        x=None,
        xstart=0,
        xstep=1,
        xlabel=None,
        *,
        legend_keys=None,
        xlim=None,
        ylim=None,
        ax=None,
        fig=None,
        epoch="run",
        **kwargs
    ):
        super().__init__(use_teleporter=kwargs.pop("use_teleporter", None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        self._xstart = xstart
        self._xstep = xstep
        self._xind = 0
        self._xlabel = xlabel
        self.x_data, self.y_data = [], []

        def setup():
            # Run this code in start() so that it runs on the correct thread.
            nonlocal y, x, legend_keys, xlim, ylim, ax, fig, epoch, kwargs
            import matplotlib.pyplot as plt

            with self.__setup_lock:
                if self.__setup_event.is_set():
                    return
                self.__setup_event.set()
            if fig is not None:
                if ax is not None:
                    raise ValueError(
                        "Values were given for both `fig` and `ax`. " "Only one can be used; prefer ax."
                    )
                warnings.warn(
                    "The `fig` keyword arugment of LivePlot is "
                    "deprecated and will be removed in the future. "
                    "Instead, use the new keyword argument `ax` to "
                    "provide specific Axes to plot on."
                )
                ax = fig.gca()
            if ax is None:
                fig, ax = plt.subplots()
            self.ax = ax

            if legend_keys is None:
                legend_keys = []
            self.legend_keys = ["scan_id"] + legend_keys
            if x is not None:
                self.x, *others = get_obj_fields([x])
            else:
                self.x = "seq_num"
            self.y, *others = get_obj_fields([y])
            self.ax.set_ylabel(y)
            if self._xlabel is None:
                self.ax.set_xlabel(x or "sequence #")
            else:
                self.ax.set_xlabel(self._xlabel)
            if xlim is not None:
                self.ax.set_xlim(*xlim)
            if ylim is not None:
                self.ax.set_ylim(*ylim)
            self.ax.margins(0.1)
            self.kwargs = kwargs
            self.lines = []
            self.legend = None
            self.legend_title = " :: ".join([name for name in self.legend_keys])
            self._epoch_offset = None  # used if x == 'time'
            self._epoch = epoch

        self.__setup = setup

    def start(self, doc):
        self.__setup()
        # The doc is not used; we just use the signal that a new run began.
        self._epoch_offset = doc["time"]  # used if self.x == 'time'
        self.clear_caches()
        label = " :: ".join([str(doc.get(name, name)) for name in self.legend_keys])
        kwargs = ChainMap(self.kwargs, {"label": label})
        (self.current_line,) = self.ax.plot([], [], **kwargs)
        self.lines.append(self.current_line)
        legend = self.ax.legend(loc=0, title=self.legend_title)
        try:
            # matplotlib v3.x
            self.legend = legend.set_draggable(True)
        except AttributeError:
            # matplotlib v2.x (warns in 3.x)
            self.legend = legend.draggable(True)
        super().start(doc)

    def event_page(self, doc):
        try:
            # This inner try/except block handles seq_num and time, which could
            # be keys in the data or accessing the standard entries in every
            # event.
            try:
                new_x = doc["data"][self.x]
            except KeyError:
                if self.x in ("time", "seq_num"):
                    new_x = doc[self.x]
                else:
                    raise
            new_y = doc["data"][self.y]
            new_x_ind = doc["data"]["index_count"]
        except KeyError:
            # wrong event stream, skip it
            return

        new_x, new_y = np.asarray(new_x), np.asarray(new_y)

        # Special-case 'time' to plot against against experiment epoch, not
        # UNIX epoch.
        if self.x == "time" and self._epoch == "run":
            new_x -= self._epoch_offset

        # override the x value
        new_x = np.asarray(new_x_ind)  # Index
        new_x = new_x * self._xstep + self._xstart

        self.update_caches(new_x, new_y)
        self.update_plot()

        super().event_page(doc)

    def clear_caches(self):
        self.x_data = np.array([])
        self.y_data = np.array([])

    def update_caches(self, x, y):
        if y.size:
            self.y_data = np.append(self.y_data, y)
            self.x_data = np.append(self.x_data, x)

    def update_plot(self):
        self.current_line.set_data(self.x_data, self.y_data)
        # Rescale and redraw.
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()

    def stop(self, doc):
        if not len(self.x_data):
            print("LivePlot did not get any data that corresponds to the " "x axis. {}".format(self.x))
        if not len(self.y_data):
            print("LivePlot did not get any data that corresponds to the " "y axis. {}".format(self.y))
        if len(self.y_data) != len(self.x_data):
            print(
                "LivePlot has a different number of elements for x ({}) and"
                "y ({})".format(len(self.x_data), len(self.y_data))
            )
        super().stop(doc)


@make_class_safe(logger=logger)
class TSLiveGrid(QtAwareCallback):
    """Plot gridded 2D data in a "heat map".
    This assumes that readings are placed on a regular grid and can be placed
    into an image by sequence number. The seq_num is used to determine which
    pixel to fill in.
    For non-gridded data with arbitrary placement, use
    :func:`bluesky.callbacks.mpl_plotting.LiveScatter`.
    This simply wraps around a `AxesImage`.
    Parameters
    ----------
    raster_shape : tuple
        The (row, col) shape of the raster
    I : str
        The field to use for the color of the markers
    clim : tuple, optional
       The color limits
    cmap : str or colormap, optional
       The color map to use
    xlabel, ylabel : str, optional
       Labels for the x and y axis
    extent : scalars (left, right, bottom, top), optional
       Passed through to :meth:`matplotlib.axes.Axes.imshow`
    aspect : str or float, optional
       Passed through to :meth:`matplotlib.axes.Axes.imshow`
    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.
    x_positive: string, optional
        Defines the positive direction of the x axis, takes the values 'right'
        (default) or 'left'.
    y_positive: string, optional
        Defines the positive direction of the y axis, takes the values 'up'
        (default) or 'down'.
    See Also
    --------
    :class:`bluesky.callbacks.mpl_plotting.LiveScatter`.
    """

    def __init__(
        self,
        raster_shape,
        I,
        *,  # noqa: E741
        clim=None,
        cmap="viridis",
        xlabel="x",
        ylabel="y",
        extent=None,
        aspect="equal",
        ax=None,
        x_positive="right",
        y_positive="up",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()
        self.x_data, self.y_data = [], []

        def setup(doc):
            # Run this code in start() so that it runs on the correct thread.
            nonlocal raster_shape, I, clim, cmap, xlabel, ylabel, extent  # noqa: E741
            nonlocal aspect, ax, x_positive, y_positive, kwargs
            with self.__setup_lock:
                if self.__setup_event.is_set():
                    return
                self.__setup_event.set()
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            if ax is None:
                fig, ax = plt.subplots()
            ax.cla()
            self.I = I  # noqa: E741
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect(aspect)
            self.ax = ax
            self._Idata = np.ones(raster_shape) * np.nan
            self._norm = mcolors.Normalize()
            if clim is not None:
                self._norm.vmin, self._norm.vmax = clim
            self.clim = clim
            self.cmap = cmap
            self.raster_shape = raster_shape
            self.im = None
            self.extent = extent
            self.aspect = aspect
            self.x_positive = x_positive
            self.y_positive = y_positive

            self.snake = doc["scan"]["snake"]
            self.nx, self.ny = doc["scan"]["shape"]

            self._index = 0
            self._start = 0

        self.__setup = setup

    def start(self, doc):
        self.__setup(doc)
        if self.im is not None:
            raise RuntimeError("Can not re-use LiveGrid")
        self._Idata = np.ones(self.raster_shape) * np.nan
        # The user can control origin by specific 'extent'.
        extent = self.extent
        # origin must be 'lower' for the plot to fill in correctly
        # (the first voxel filled must be closest to what mpl thinks
        # is the 'lower left' of the image)
        im = self.ax.imshow(
            self._Idata,
            norm=self._norm,
            cmap=self.cmap,
            interpolation="none",
            extent=extent,
            aspect=self.aspect,
            origin="lower",
        )

        # make sure the 'positive direction' of the axes matches what
        # is defined in axes_positive
        xmin, xmax = self.ax.get_xlim()
        if (xmin > xmax and self.x_positive == "right") or (xmax > xmin and self.x_positive == "left"):
            self.ax.set_xlim(xmax, xmin)
        elif (xmax >= xmin and self.x_positive == "right") or (xmin >= xmax and self.x_positive == "left"):
            self.ax.set_xlim(xmin, xmax)
        else:
            raise ValueError('x_positive must be either "right" or "left"')

        ymin, ymax = self.ax.get_ylim()
        if (ymin > ymax and self.y_positive == "up") or (ymax > ymin and self.y_positive == "down"):
            self.ax.set_ylim(ymax, ymin)
        elif (ymax >= ymin and self.y_positive == "up") or (ymin >= ymax and self.y_positive == "down"):
            self.ax.set_ylim(ymin, ymax)
        else:
            raise ValueError('y_positive must be either "up" or "down"')

        self.im = im
        self.ax.set_title("scan {uid} [{sid}]".format(sid=doc["scan_id"], uid=doc["uid"][:6]))
        self.snaking = doc.get("snaking", (False, False))

        cb = self.ax.figure.colorbar(im, ax=self.ax)
        cb.set_label(self.I)
        super().start(doc)

    def event_page(self, doc):
        try:
            # This inner try/except block handles seq_num and time, which could
            # be keys in the data or accessing the standard entries in every
            # event.
            new_y = doc["data"][self.I]
            new_x_ind = doc["data"]["index_count"]
        except KeyError:
            # wrong event stream, skip it
            return

        new_x, new_y = np.asarray(new_x_ind), np.asarray(new_y)
        if len(new_y):
            self.y_data = np.append(self.y_data, new_y)
            self.x_data = np.append(self.x_data, new_x)

            n_total = self.nx * self.ny
            if len(self.y_data) > n_total:
                image_data = self.y_data[:n_total]
            else:
                image_data = np.pad(self.y_data, (0, n_total - len(self.y_data)), constant_values=np.nan)
            image_data = image_data.reshape([self.ny, self.nx])

            if self.snake:
                ind = np.arange(1, self.ny, 2)
                image_data[ind] = np.fliplr(image_data[ind])

            self.update(image_data)

        super().event_page(doc)

    def clear_caches(self):
        self.x_data = np.array([])
        self.y_data = np.array([])

    def update(self, I):  # noqa: E741
        # self._Idata[self._index, :] = I
        self._Idata = I
        if self.clim is None:
            self.im.set_clim(np.nanmin(self._Idata), np.nanmax(self._Idata))

        self.im.set_array(self._Idata)
        self.ax.figure.canvas.draw_idle()


class HackLiveFlyerPlot(QtAwareCallback):
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
    def __init__(self, y, x=None, xstart=0, xstep=1, xlabel=None, *, legend_keys=None, xlim=None, ylim=None,
                 epoch='run', fig_factory=None, **kwargs):
        super().__init__(use_teleporter=kwargs.pop('use_teleporter', None))
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()

        self._xstart = xstart
        self._xstep = xstep
        self._xind = 0
        self._xlabel = xlabel

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

        self.__setup = setup

    def start(self, doc):
        self.__setup()
        # The doc is not used; we just use the signal that a new run began.
        self._epoch_offset = doc['time']  # used if self.x == 'time'
        self.x_data, self.y_data = [], []
        label = " :: ".join(
            [str(doc.get(name, name)) for name in self.legend_keys])
        kwargs = ChainMap(self.kwargs, {'label': label})
        self.current_line, = self.ax.plot([], [], **kwargs)
        self.lines.append(self.current_line)
        self._xind = 0
        legend = self.ax.legend(loc=0, title=self.legend_title)
        try:
            # matplotlib v3.x
            self.legend = legend.set_draggable(True)
        except AttributeError:
            # matplotlib v2.x (warns in 3.x)
            self.legend = legend.draggable(True)
        super().start(doc)

    def event(self, doc):
        "Unpack data from the event and call self.update()."
        # This outer try/except block is needed because multiple event
        # streams will be emitted by the RunEngine and not all event
        # streams will have the keys we want.
        try:
            # This inner try/except block handles seq_num and time, which could
            # be keys in the data or accessing the standard entries in every
            # event.
            try:
                new_x = doc['data'][self.x]
            except KeyError:
                if self.x in ('time', 'seq_num'):
                    new_x = doc[self.x]
                else:
                    raise
            new_y = doc['data'][self.y]
        except KeyError:
            # wrong event stream, skip it
            return

        # Special-case 'time' to plot against against experiment epoch, not
        # UNIX epoch.
        if self.x == 'time' and self._epoch == 'run':
            new_x -= self._epoch_offset

        #overright the x value
        new_x = self._xstart + self._xstep * self._xind
        self._xind = self._xind + 1

        self.update_caches(new_x, new_y)
        self.update_plot()
        super().event(doc)

    def update_caches(self, x, y):
        self.y_data.append(y)
        self.x_data.append(x)

    def update_plot(self):
        self.current_line.set_data(self.x_data, self.y_data)
        # Rescale and redraw.
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()

    def stop(self, doc):
        if not self.x_data:
            print('LivePlot did not get any data that corresponds to the '
                  'x axis. {}'.format(self.x))
        if not self.y_data:
            print('LivePlot did not get any data that corresponds to the '
                  'y axis. {}'.format(self.y))
        if len(self.y_data) != len(self.x_data):
            print('LivePlot has a different number of elements for x ({}) and'
                  'y ({})'.format(len(self.x_data), len(self.y_data)))
        super().stop(doc)
