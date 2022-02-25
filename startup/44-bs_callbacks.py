print(f'Loading {__file__}...')


import time as ttime
from bluesky.callbacks import CallbackBase,LivePlot
from xray_vision.backend.mpl.cross_section_2d import CrossSection

# Try-except only hear for testing. Can be removed after validation.
def cb_print_scaninfo(name, doc):
    if name == "start":
        try:
            str1 = f"Scan Type:  {doc['scan']['type']}"
        except KeyError:
            str1 = f"Scan Type:  {doc['plan_name'].upper()}"
        except:
            str1 = ''
        str2 = f"Scan ID:    {doc['scan_id']}"
        str3 = f"Start Time: {doc['time_str']}"
        banner([str1, str2, str3])
    elif name == "stop":
        try:
            start_doc = db[doc['run_start']].start
        except:
            start_doc = None

        if start_doc is not None:
            try:
                str1 = f"Scan Type:    {start_doc['scan']['type']}"
            except KeyError:
                str1 = f"Scan Type:  {doc['plan_name'].upper()}"
            except:
                str1 = ''
            str2 = f"Scan ID:      {start_doc['scan_id']}"
            str3 = f"Stop Time:    {ttime.ctime(doc['time'])}"
            str4 = f"  Total Time: {doc['time'] - start_doc['time']:.2f} s"
            banner([str1, str2, str3, str4])
        else: 
            str3 = f"Stop Time:    {ttime.ctime(doc['time'])}"
            banner([str3])


RE.subscribe(cb_print_scaninfo)


i0_baseline = 7.24e-10


# LivePlot for XANES measurements
#   Need to remove i0_baseline (not using current anymore and we won't have
#     1e-10 counts
#   Can we incorporate all the detector channels, instead of only one?
class NormalizeLivePlot(HackLivePlot):
    def __init__(self, *args, norm_key=None,  **kwargs):
        super().__init__(*args, **kwargs)
        if norm_key is None:
            raise RuntimeError("norm key is required kwarg")
        self._norm = norm_key

    def event(self, doc):
        "Update line with data from this Event."
        try:
            if self.x is not None:
                # this try/except block is needed because multiple event
                # streams will be emitted by the RunEngine and not all event
                # streams will have the keys we want
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


# I don't think anything below here is used...
# Can be removed once this is confirmed.
def make_live_image(image_axes, key):
    """
    Example
    p--------

    fig, ax = plt.subplots()
    image_axes = ax.imshow(np.zeros((476, 512)), vmin=0, vmax=2)
    cb = make_live_image(image_axes, 'pixi_image_array_data')
    RE(Count([pixi]), subs={'event': [cb]})
    """
    def live_image(name, doc):
        if name != 'event':
            return
        image_axes.set_data(doc['data'][key].reshape(476, 512))
    return live_image


class SRXLiveImage(CallbackBase):
    """
    Stream 2D images in a cross-section viewer.

    Parameters
    ----------
    field : string name of data field in an Event

    Note
    ----
    Requires a matplotlib fix that is not released as of this writing. The
    relevant commit is a951b7.
    """
    def __init__(self, field):
        super().__init__()
        self.field = field
        fig = plt.figure()
        self.cs = CrossSection(fig)
        self.cs._fig.show()

    def event(self, doc):
        uid = doc['data'][self.field]
        data = db.reg.retrieve(uid)
        self.cs.update_image(data)
        self.cs._fig.canvas.draw_idle()
