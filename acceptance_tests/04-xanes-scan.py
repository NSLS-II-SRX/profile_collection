
print('4. testing integrated scan functions:')
print('testing xanes')
xanes(erange = [7112-30, 7112-20, 7112+30], 
            estep = [2, 5],  
            harmonic = None,            
            acqtime=0.2, roinum=1, i0scale = 1e8, itscale = 1e8,samplename='test',filename='test')

