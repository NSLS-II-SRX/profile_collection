from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI
RM = REManagerAPI()

print("queue server plans")

def get_current_position():

	roi = {}

	sx, sy = nano_stage.sx.position, nano_stage.sy.position #scanners (used for scanning at HXN)
	x, y, z = nano_stage.topx.position, nano_stage.y.position, nano_stage.z.position #coarse motors
	topx, topz = nano_stage.topx.position, nano_stage.topz.position #top
	th = nano_stage.th.position

	roi = {
	"nano_stage.sx": sx, "nano_stage.sy": sy,
	"nano_stage.topx": x, "nano_stage.y": y, "nano_stage.z": z,
	# "nano_stage.topx": topx, "nano_stage.topz": topz,
	"nano_stage.topx": topx,
	"nano_stage.th": th
	}

	return roi

def recover_and_scan_nano(label, roi_positions, mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t):
    print(f"{label} running")
    for key, value in roi_positions.items():
        print(f"{key=}\t{value=}")
        yield from bps.mov(eval(key), value) #recover positions
        print(f"{key} moved to {value :.3f}")
    
    print(f"mot1_s={mot1_s}")
    print(f"mot1_e={mot1_e}")
    print(f"mot1_n={mot1_n}")
    print(f"mot2_s={mot2_s}")
    print(f"mot2_e={mot2_e}")
    print(f"mot2_n={mot2_n}")
    print(f"exp_t={exp_t}")
    # yield from nano_scan_and_fly(mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t,extra_dets = [eval(extra_dets)])
    yield from nano_scan_and_fly(mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t)


def recover_and_scan_coarse(label, roi_positions, mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t):

	print(f"{label} running")
	for key, value in roi_positions.items():
	
		yield from bps.mov(eval(key), value) #recover positions
		print(f"{key} moved to {value :.3f}")
	
	#yield from coarse_scan_and_fly(mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t,extra_dets = [eval(extra_dets)])
	yield from coarse_scan_and_fly(mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t)

def send_nano_plan_to_queue(label, mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t):

	roi = get_current_position()
	RM.item_add((BPlan("recover_and_scan",label, roi, mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t))) #2D flyscan


def send_coarse_plan_to_queue(label, mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t):

	roi = get_current_position()
	RM.item_add((BPlan("recover_and_scan_coarse",label, roi, mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t))) #2D flyscan


def download_qs_history(file_name="qs_history", *, format="text"):
    """
    Downloads history from the Queue Server and saves it as a text or
    or a JSON file. The text file is nicely formatted and human readable.
    JSON file is human readable, but also machine-readable.

    Parameters
    ----------
    file_name: str
        File name without extension. The function adds ``.txt`` extension
        to the name of a text file and ``.json`` extension to the name of
        a JSON file.
    format: str ('text' or 'json')
        The string that specifies the format of the output file.
    """
    from bluesky_queueserver_api.zmq import REManagerAPI
    import pprint
    import json
    RM = REManagerAPI()

    resp = RM.history_get()
    history = resp["items"]
    if format.lower() == "text":
        file_name = file_name + ".txt"
        with open(file_name, "wt") as f:
            for n, plan in enumerate(history):
                f.write("=" * 80)
                s = f"PLAN {n + 1}"
                f.write(f"\n{s: ^80}\n")
                f.write("=" * 80)
                f.write(f"\n{pprint.pformat(plan, width=80)}\n")
    elif format.lower() == "json":
        file_name = file_name + ".json"
        with open(file_name, "wt") as f:
            json.dump(history, f, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {format!r}")
    print(f"QS history was saved to the file {file_name!r}")
