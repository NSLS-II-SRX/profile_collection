print(f'Loading {__file__}...')

from ophyd import EpicsSignal

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
    
    
    
