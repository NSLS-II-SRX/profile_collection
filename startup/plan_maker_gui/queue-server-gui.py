#2023-2.0-py310

import os
import sys
import collections
import json
import numpy as np
import pprint

from epics import caget, caput
from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QLabel, QErrorMessage
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot

from bluesky_queueserver_api import BPlan, BFunc
from bluesky_queueserver_api.zmq import REManagerAPI

ui_path = os.path.dirname(os.path.abspath(__file__))

RM = None

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


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(os.path.join(ui_path,'send_to_queue.ui'), self)

        self.pb_nano_send_to_queue.clicked.connect(self.send_nanofly_to_queue)
        self.pb_coarse_send_to_queue.clicked.connect(self.send_coarsefly_to_queue)
        self.pb_nano_makeplan.clicked.connect(self.make_nanofly_plan)
        self.pb_coarse_makeplan.clicked.connect(self.make_coarsefly_plan)
        self.pb_copy_plan.clicked.connect(self.copy_scan_plan)


    def send_nanofly_to_queue(self):
        x_s = self.dsb_nanox_start.value()
        x_e = self.dsb_nanox_end.value()
        x_n = self.sb_nano_xn.value()
        y_s = self.dsb_nanoy_start.value()
        y_e = self.dsb_nanoy_end.value()
        y_n = self.sb_nano_yn.value()
        exp = self.dsb_nano_exp_time.value()
        scan_label = self.le_nano_label.text()

        params = {"mot1_s": x_s, "mot1_e": x_e, "mot1_n": x_n, "mot2_s": y_s, "mot2_e": y_e, "mot2_n": y_n, "exp_t": exp}
        plan_name = "recover_and_scan_nano"
        add_plan_with_roi_positions(plan_name, scan_label, params)

    def send_coarsefly_to_queue(self):
        x_s = self.dsb_coarsex_start.value()
        x_e = self.dsb_coarsex_end.value()
        x_n = self.sb_coarse_xn.value()
        y_s = self.dsb_coarsey_start.value()
        y_e = self.dsb_coarsey_end.value()
        y_n = self.sb_coarse_yn.value()

        scan_label = self.le_coarse_label.text()
        exp = self.dsb_coarse_exp_time.value()

        params = {"mot1_s": x_s, "mot1_e": x_e, "mot1_n": x_n, "mot2_s": y_s, "mot2_e": y_e, "mot2_n": y_n, "exp_t": exp}
        plan_name = "recover_and_scan_coarse"
        add_plan_with_roi_positions(plan_name, scan_label, params)

    def make_nanofly_plan(self):
        x_s = self.dsb_nanox_start.value()
        x_e = self.dsb_nanox_end.value()
        x_n = self.sb_nano_xn.value()
        y_s = self.dsb_nanoy_start.value()
        y_e = self.dsb_nanoy_end.value()
        y_n = self.sb_nano_yn.value()
        exp = self.dsb_nano_exp_time.value()

        scan_label = self.le_nano_label.text()

        nano_plan = f"send_nano_plan_to_queue('{scan_label}', {x_s}, {x_e}, {x_n}, {y_s}, {y_e}, {y_n}, {exp})"
        self.le_plan_to_copy.setText(nano_plan)
        self.te_scan_details.clear()
        scan_time = ((x_n*y_n*exp)+(x_n*3.8))/60
        x_step = (abs(x_e)+abs(x_s))/(x_n-1)
        y_step = (abs(y_e)+abs(y_s))/(y_n-1)

        scan_details_text = f"{scan_time = :.2f} minutes, {x_step = :.2f} um,{y_step = :.2f} um"

        self.te_scan_details.setText(scan_details_text)
        self.copy_scan_plan()


    def make_coarsefly_plan(self):
        x_s = self.dsb_coarsex_start.value()
        x_e = self.dsb_coarsex_end.value()
        x_n = self.sb_coarse_xn.value()
        y_s = self.dsb_coarsey_start.value()
        y_e = self.dsb_coarsey_end.value()
        y_n = self.sb_coarse_yn.value()

        scan_label = self.le_coarse_label.text()
        exp = self.dsb_coarse_exp_time.value()

        x_pos = np.around(caget("XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr.RBV"),2)
        y_pos = np.around(caget("XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.RBV"),2)

        coarse_plan = f"send_coarse_plan_to_queue('{scan_label}', {x_pos+x_s}, {x_pos+x_e}, {x_n}, {y_pos+y_s}, {y_pos+y_e}, {y_n}, {exp})"
        self.le_plan_to_copy.setText(coarse_plan)
        self.te_scan_details.clear()
        scan_time = ((x_n*y_n*exp)+(x_n*3.8))/60
        x_step = (abs(x_e)+abs(x_s))/(x_n-1)
        y_step = (abs(y_e)+abs(y_s))/(y_n-1)

        scan_details_text = f"{scan_time = :.2f} minutes, {x_step = :.2f} um,{y_step = :.2f} um"

        self.te_scan_details.setText(scan_details_text)
        self.copy_scan_plan()

    def copy_scan_plan(self):

        self.le_plan_to_copy.selectAll()
        self.le_plan_to_copy.copy()


if __name__ == "__main__":
    RM = REManagerAPI()

    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())


