#import os
#import numpy as np
#
#def breakdown(batch_dir=None, batch_filename=None,xstart=None,ystart=None,\
#    xsteps=None,ysteps=None,xstepsize=None,ystepsize=None,zposition=None,\
#    acqtime=None,numrois=None,xbasestep=39,ybasestep=39):
#    '''
#    helper function for hf2dxrf_xybath
#    takes a large range with uniform step size and breaks it into chunks
#
#    batch_dir (string): directory for the input batch file
#    batch_filename (string): text file name that defines the set points for batch scans
#    xstart (float): starting x position
#    ystart (float): starting y position
#    xsteps (int): steps in X
#    ysteps (int): steps in Y
#    xstepsize (float): scan step in X
#    ystepsize (float): scan step in Y
#    zposition (float or list of floats): position(s) in z
#    acqtime (float): acquisition time
#    numrois (int): number or ROIs
#    xbasestep (int): number of X steps in each atomic sub-scan
#    ybasestep (int): number of Y steps in each atomic sub-scan
#    '''
#    xchunks=np.ceil(xsteps/xbasestep)
#    ychunks=np.ceil(ysteps/ybasestep)
#    xoverflow=np.mod(xsteps,xbasestep)
#    yoverflow=np.mod(ysteps,ybasestep)
#    print(ychunks)
#
#    if zposition is None:
#        zposition=[hf_stage.z.position]
#    if zposition.__class__ is not list:
#        zposition=[zposition]
#    mylist=list()
#    for k in zposition:
#        for j in range(0,int(ychunks),1):
#            for i in range(0,int(xchunks),1):
#                xs= xstart+(xbasestep+1)*i*xstepsize
#                ys= ystart+(ybasestep+1)*j*ystepsize
#                if (ychunks > 1):
#                    if (j==ychunks-1):
#                        ysteps=yoverflow
#                    else:
#                        ysteps=ybasestep
#                if (xchunks>1):
#                    if(i==xchunks-1):
#                        xsteps=xoverflow
#                    else:
#                        xsteps=xbasestep
#                
#                mylist.append([k,xs,xsteps,xstepsize,ys,ysteps,ystepsize,\
#                acqtime,numrois])
#    if batch_dir is None:
#        batch_dir = os.getcwd()
#        print("No batch_dir was assigned, using the current directory")
#    else:
#        if not os.path.isdir(batch_dir):
#            raise Exception(\
#            "Please provide a valid batch_dir for the batch file path.")
#    if batch_filename is None:
#        raise Exception(\
#        "Please provide a batch file name, e.g. batch_file = 'xrf_batch_test.txt'.")
#    batchfile = batch_dir+'/'+batch_filename
#                                                                           
#    with open(batchfile, 'w') as batchf:
#        for item in mylist:
#            for entry in item:
#                batchf.write('%s '%entry)
#            batchf.write('\n')
#    return mylist
