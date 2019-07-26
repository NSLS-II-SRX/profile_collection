# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:05:35 2016

@author: xf05id1
"""
import time

_DEFAULT_FILEDIR = ""


def textout(
    scan=-1,
    header=[],
    userheader={},
    column=[],
    usercolumn={},
    usercolumnname=[],
    output=True,
    filename_add="",
    filedir=None,
):
    """
    scan: can be scan_id (integer) or uid (string). defaul = -1 (last scan run)
          default = -1
    header: a list of items that exist in the event data to be put into the header
    userheader: a dictionary defined by user to put into the hdeader
    column: a list of items that exist in the event data to be put into the column data
    output: print all header fileds. if output = False, only print the ones that were able to be written
            default = True

    """
    if filedir is None:
        filedir = _DEFAULT_FILEDIR
    scanh = db[scan]
    #    print(scanh.start)
    events = list(
        scanh.events(scanh, fill=False, stream_name="primary")
    )  # fill=False so it does not look for the metadata in filestorage with reference (hdf5 here)

    # convert time stamp to localtime
    # timestamp=scanhh.start['time']
    # scantime=time.localtime(timestamp)

    # filedir=userdatadir

    if filename_add != "":
        filename = "scan_" + str(scanh.start["scan_id"]) + "_" + filename_add
    else:
        filename = "scan_" + str(scanh.start["scan_id"])

    #    print(filedir)
    #    print(filename)

    f = open(filedir + filename, "w")

    staticheader = (
        "# XDI/1.0 MX/2.0\n"
        + "# Beamline.name: "
        + scanh.start.beamline_id
        + "\n"
        + "# Facility.name: NSLS-II\n"
        + "# Facility.ring_current:"
        + str(events[0]["data"]["ring_current"])
        + "\n"
        + "# Scan.start.uid: "
        + scanh.start.uid
        + "\n"
        + "# Scan.start.time: "
        + str(scanh.start.time)
        + "\n"
        + "# Scan.start.ctime: "
        + time.ctime(scanh.start.time)
        + "\n"
        + "# Mono.name: Si 111\n"
    )
    # +'# bpm.cam.exposure_time: '+str(events[0].descriptor.configuration['bpmAD']['data']['bpmAD_cam_acquire_time'])+'\n'  \
    # +'# Undulator.elevation: '+str(scanh.start.undulator_setup['elevation'])+'\n'  \
    # +'# Undulator.tilt: '+str(scanh.start.undulator_setup['tilt'])+'\n'  \
    # +'# Undulator.taper: '+str(scanh.start.undulator_setup['taper'])+'\n'

    f.write(staticheader)

    for item in header:
        if item in events[0].data.keys():
            f.write("# " + item + ": " + str(events[0]["data"][item]) + "\n")
            if output is True:
                print(item + " is written")
        else:
            print(item + " is not in the scan")

    for key in userheader:
        f.write("# " + key + ": " + str(userheader[key]) + "\n")
        if output is True:
            print(key + " is written")

    for idx, item in enumerate(column):
        if item in events[0].data.keys():
            f.write("# Column." + str(idx + 1) + ": " + item + "\n")

    f.write("# ")
    for item in column:
        if item in events[0].data.keys():
            f.write(str(item) + "\t")

    for item in usercolumnname:
        f.write(item + "\t")

    f.write("\n")
    f.flush()

    idx = 0
    for event in events:
        for item in column:
            if item in events[0].data.keys():
                # f.write(str(event['data'][item])+'\t')
                f.write("{0:8.6g}  ".format(event["data"][item]))
        for item in usercolumnname:
            try:
                # f.write(str(usercolumn[item][idx])+'\t')
                f.write("{0:8.6g}  ".format(usercolumn[item][idx]))
            except KeyError:
                idx += 1
                f.write("{0:8.6g}  ".format(usercolumn[item][idx]))
        idx = idx + 1
        f.write("\n")

    f.close()
