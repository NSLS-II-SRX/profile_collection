from bluesky_queueserver_api import BPlan, BFunc
from bluesky_queueserver_api.zmq import REManagerAPI
import pprint

RM = REManagerAPI()

def add_plan_with_roi_positions(plan_name, label, params):
    try:
        func = BFunc("get_current_position")

        print("Requesting ROI positions ...")
        resp = RM.function_execute(func)

        print("Waiting for positions ...")
        task_uid = resp["task_uid"]
        RM.wait_for_completed_task(task_uid)
        print("Request complete ...")

        resp = RM.task_result(task_uid)
        if not resp["result"]["success"]:
            msg = resp["result"]["msg"]
            print(f"Failed to execute the function: {msg}")
        else:
            roi_positions = resp["result"]["return_value"]
            print(f"ROI positions:\n{pprint.pformat(roi_positions)}")

        plan_params = dict(roi_positions=roi_positions, label=label, **params)
        plan = BPlan(plan_name, **plan_params)

        print("Adding plan to the queue ...")
        RM.item_add(plan)
        print(f"Plan {plan_name!r} was added to the queue. Plan parameters:\n{pprint.pformat(plan_params)}")

    except Exception as ex:
        print(f"Failed to execute operation: {ex}")
        raise


# plan_name = "recover_and_scan_nano"
plan_name = "recover_and_scan_coarse"

params = {"mot1_s": -1, "mot1_e": 1, "mot1_n": 21, "mot2_s": -2, "mot2_e": 2, "mot2_n": 3, "exp_t": 0.05}

label = "my_scan"

add_plan_with_roi_positions(plan_name, label, params)