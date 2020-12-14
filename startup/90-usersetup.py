print(f'Loading {__file__}...')

import os
# import os.path
import time as ttime
import shutil


### Proposal information put into the metadata
# proposal_num = None
# proposal_title = None
# PI_lastname = None
# saf_num = None

proposal_num = 307426
proposal_title = 'SRX Beamline Commissioning'
PI_lastname = 'Kiss'
saf_num = 305921

# proposal_num = 307426
# proposal_title = 'SRX Beamline Commissioning'
# PI_lastname = 'Hidas'
# saf_num = 305921

# proposal_num = 306378
# proposal_title = 'Investigating the charge distribution of Ni-ricj LiNi1-x-yCosMnyO2 cathode in high-energy resources'
# PI_lastname = 'Zuli'
# saf_num = 306149


cycle = '2020_cycle3'

# Set user data in bluesky
RE.md['proposal']  = {'proposal_num': str(proposal_num),
                      'proposal_title': str(proposal_title),
                      'PI_lastname': str(PI_lastname),
                      'saf_num': str(saf_num),
                      'cycle': str(cycle)}

if 'TOUCHBEAMLINE' in os.environ and os.environ['TOUCHBEAMLINE'] == 1:
    # Set user data in scanbroker
    scanrecord.update_metadata()

# User data directory and simple scripts
userdatadir = '/nsls2/xf05id1/experiments/' + str(cycle) + '/' + str(saf_num) + '_' + str(PI_lastname) + '/'
scriptdir = '/nsls2/xf05id1/shared/src/bluesky_scripts/'

# Create the user directory
try:
    os.makedirs(userdatadir, exist_ok=True)
except Exception as e:
    print(e)
    raise OSError('Cannot create directory: ' + userdatadir)

# Create the log file
userlogfile = userdatadir + 'logfile' + str(saf_num) + '.txt'
if not os.path.exists(userlogfile):
    userlogf = open(userlogfile, 'w')
    userlogf.close()

# Copy over the example scripts
for f in ['simple_batch.py','fly_batch.py']:
    if (not os.path.exists(userdatadir + f)):
        shutil.copyfile(scriptdir + f, userdatadir + f)

# Update the symbolic link to the data
try:
    os.unlink('/nsls2/xf05id1/shared/current_user_data')
except FileNotFoundError:
    print("[W] Previous user data directory link did not exist!")

os.symlink(userdatadir, '/nsls2/xf05id1/shared/current_user_data')


def get_stock_md(scan_md):
    # Should this be ChainMap(scan_md, {...})?
    # This should also be put into baseline, and not start document
    scan_md['beamline_status']  = {'energy':  energy.energy.position}
    scan_md['initial_sample_position'] = {'hf_stage_x': hf_stage.x.position,
                                          'hf_stage_y': hf_stage.y.position,
                                          'hf_stage_z': hf_stage.z.position}
    scan_md['wb_slits'] = {'v_gap' : slt_wb.v_gap.position,
                           'h_gap' : slt_wb.h_gap.position,
                           'v_cen' : slt_wb.v_cen.position,
                           'h_cen' : slt_wb.h_cen.position}
    scan_md['hfm'] = {'y' : hfm.y.position,
                      'bend' : hfm.bend.position}
    scan_md['ssa_slits'] = {'v_gap' : slt_ssa.v_gap.position,
                            'h_gap' : slt_ssa.h_gap.position,
                            'v_cen' : slt_ssa.v_cen.position,
                            'h_cen' : slt_ssa.h_cen.position}


def logscan(scantype):
    h = db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    userlogf = open(userlogfile, 'a')
    userlogf.write(str(scan_id) + '\t' + uid + '\t' + scantype + '\n')
    userlogf.close()


def logscan_event0info(scantype, event0info = []):
    h = db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    userlogf = open(userlogfile, 'a')
    userlogf.write(str(scan_id) + '\t' + uid + '\t' + scantype)
    events = list(db.get_events(h, stream_name='primary'))

    for item in event0info:
        userlogf.write('\t' + item + '=' + str(events[0]['data'][item]) + '\t')
    userlogf.write('\n')
    userlogf.close()


def logscan_detailed(scantype):
    h = db[-1]
    scan_id = h.start['scan_id']
    uid = h.start['uid']

    userlogf = open(userlogfile, 'a')
    userlogf.write(str(scan_id) + '\t' + uid + '\t' + scantype + '\t' + str(h['start']['scan_input']) + '\n')
    userlogf.close()


def scantime(scanid, printresults=True):
    '''
    input: scanid
    return: start and stop time stamps as strings
    '''
    start_str = 'scan start: ' + ttime.ctime(db[scanid].start['time'])
    stop_str  = 'scan stop : ' + ttime.ctime(db[scanid].stop['time'])
    totaltime = db[scanid].stop['time'] - db[scanid].start['time']
    scannumpt = len(list(db.get_events(db[scanid])))

    if (printresults):
        print(start_str)
        print(stop_str)
        print('Total time: ', totaltime, ' s')
        print('Number of points: ', scannumpt)
        print('Scan time per point: ', totaltime/scannumpt, ' s')
    return db[scanid].start['time'], db[scanid].stop['time'], start_str, stop_str


def timestamp_batchoutput(filename='timestamplog.text', initial_scanid=None, final_scanid=None):
    with open(filename, 'w') as f:
        for scanid in range(initial_scanid, final_scanid+1):
            f.write(str(scanid) + '\n')
            try:
                _, _, start_t, stop_t = scantime(scanid)
                f.write(start_t)
                f.write('\n')
                f.write(stop_t)
                f.write('\n')
            except:
                f.write('Scan did no finish correctly.\n')


def scantime_batchoutput(filename='scantimelog.txt', scanlist=[]):
    with open(filename, 'w') as f:
        f.write('scanid\tstartime(s)\tstoptime(s)\tstartime(date-time)\tstoptime(date-time)\n')
        for i in scanlist:
            starttime_s, endtime_s, starttime, endtime = scantime(i, printresults=False)
            f.write(str(i) + '\t' + str(starttime_s) + '\t' + str(endtime_s) + '\t' \
                    + starttime[12::] + '\t' + endtime[12::] + '\n')
