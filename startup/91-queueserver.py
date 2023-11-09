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
	"nano_stage.x": x, "nano_stage.y": y, "nano_stage.z": z,
	"nano_stage.topx": topx, "nano_stage.topz": topz,
	"nano_stage.th": th
	}

	return roi

def recover_and_scan_nano(label, roi_positions, mot1_s, mot1_e, mot1_n, mot2_s, mot2_e, mot2_n, exp_t):

	print(f"{label} running")
	for key, value in roi_positions.items():
	
		yield from bps.mov(eval(key), value) #recover positions
		print(f"{key} moved to {value :.3f}")
	
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
