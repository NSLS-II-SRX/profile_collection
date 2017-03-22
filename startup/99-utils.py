import os
import numpy as np
from ophyd import EpicsSignal
from bluesky.plans import relative_scan
from bluesky.callbacks import LiveFit,LiveFitPlot
from bluesky.callbacks.scientific import PeakStats
import lmfit

def cryofill(wait_time_after_v19_claose = 60*10):
    cryo_v19_possp = EpicsSignal('XF:05IDA-UT{Cryo:1-IV:19}Pos-SP', name='cryov19_possp')
    cryo_v19_possp.set(100)
    while abs(cryo_v19.get() - 1) > 0.05:
        cryo_v19_possp.set(100)
        time.sleep(2)
    
    time.sleep(5)
    while (cryo_v19.get() - 0) > 0.05:
        print('cryo cooler still refilling')
        time.sleep(5)
    cryo_v19_possp.set(0)
    print('waiting for', wait_time_after_v19_claose, 's', 'before taking data...')
    time.sleep(wait_time_after_v19_claose)    

def breakdown(batch_dir=None, batch_filename=None,xstart=None,ystart=None,\
    xsteps=None,ysteps=None,xstepsize=None,ystepsize=None,zposition=None,\
    acqtime=None,numrois=None,xbasestep=39,ybasestep=39):
    '''
    helper function for hf2dxrf_xybath
    takes a large range with uniform step size and breaks it into chunks

    batch_dir (string): directory for the input batch file
    batch_filename (string): text file name that defines the set points for batch scans
    xstart (float): starting x position
    ystart (float): starting y position
    xsteps (int): steps in X
    ysteps (int): steps in Y
    xstepsize (float): scan step in X
    ystepsize (float): scan step in Y
    zposition (float or list of floats): position(s) in z
    acqtime (float): acquisition time
    numrois (int): number or ROIs
    xbasestep (int): number of X steps in each atomic sub-scan
    ybasestep (int): number of Y steps in each atomic sub-scan
    '''
    xchunks=np.ceil((xsteps+1)/(xbasestep+1))
    ychunks=np.ceil((ysteps+1)/(ybasestep+1))
    xoverflow=np.mod((xsteps+1),(xbasestep+1))-1
    yoverflow=np.mod((ysteps+1),(ybasestep+1))-1
    print('xdimension = '+str(xchunks))
    print('ydimension = '+str(ychunks))
    print('xoverflow = '+str(xoverflow))
    print('yoverflow = '+str(yoverflow))

    if zposition is None:
        zposition=[hf_stage.z.position]
    if zposition.__class__ is not list:
        zposition=[zposition]
    mylist=list()
    for k in zposition:
        for j in range(0,int(ychunks),1):
            for i in range(0,int(xchunks),1):
                xs= xstart+(xbasestep+1)*i*xstepsize
                ys= ystart+(ybasestep+1)*j*ystepsize
                if (ychunks > 1):
                    if ((j==ychunks-1) and (yoverflow >= 0)):
                        ysteps=yoverflow
                    else:
                        ysteps=ybasestep
                if (xchunks>1):
                    if((i==xchunks-1) and (xoverflow >= 0)):
                        xsteps=xoverflow
                    else:
                        xsteps=xbasestep
                
                mylist.append([k,xs,xsteps,xstepsize,ys,ysteps,ystepsize,\
                acqtime,numrois])
    if batch_dir is None:
        batch_dir = os.getcwd()
        print("No batch_dir was assigned, using the current directory")
    else:
        if not os.path.isdir(batch_dir):
            raise Exception(\
            "Please provide a valid batch_dir for the batch file path.")
    if batch_filename is None:
        raise Exception(\
        "Please provide a batch file name, e.g. batch_file = 'xrf_batch_test.txt'.")
    batchfile = batch_dir+'/'+batch_filename
                                                                           
    with open(batchfile, 'w') as batchf:
        for item in mylist:
            for entry in item:
                batchf.write('%s '%entry)
            batchf.write('\n')
    return mylist

def xybatch_grid(xstart, xstepsize, xnumstep, ystart, ystepsize, ynumstep):
    xylist = []
    for j in np.linspace(ystart, ystart+ystepsize*ynumstep, ynumstep+1):
        for i in np.linspace(xstart, xstart+xstepsize*xnumstep, xnumstep+1):
            xylist.append([i, j])
    return xylist

def gaussian(x, A, sigma, x0):
    return A*np.exp(-(x - x0)**2/(2 * sigma**2))

def peakup_dcm():
    e_value=energy.energy.get()[1]
    det = [sclr1]
    ps = PeakStats(dcm.c2_pitch.name,i0.name)
    shut_b.put(1)
    
    #if e_value < 10.:
    #    sclr1.preset_time.put(0.1)
    #    RE(scan([sclr1], dcm.c2_pitch, -19.335, -19.305, 31), [ps])
    #else:
    #    sclr1.preset_time.put(1.)
    #    RE(scan([sclr1], dcm.c2_pitch, -19.355, -19.310, 46), [ps])
    if e_value < 10.:
        sclr1.preset_time.put(0.1)
    else:
        sclr1.preset_time.put(1.)
    RE(scan([sclr1], dcm.c2_pitch, -19.250, -19.200, 51), [ps])


    #RE(relative_scan([sclr1], dcm.c2_pitch, -0.01, 0.01, 21), [ps])
    dcm.c2_pitch.move(ps.cen)


def retune_undulator():
    energy.detune.put(0.)
