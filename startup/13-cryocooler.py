print(f'Loading {__file__}...')


from ophyd import EpicsSignal


### Signals
cryo_v19 = EpicsSignal('XF:05IDA-UT{Cryo:1-IV:19}Sts-Sts', name='cryo_v19')
cryo_v19_possp = EpicsSignal('XF:05IDA-UT{Cryo:1-IV:19}Pos-SP', name='cryov19_possp')


# Is this to fill the cryocooler? This is currently on autofill
def cryofill(wait_time_after_v19_close=60*10):
    cryo_v19_possp.set(100)
    while abs(cryo_v19.get() - 1) > 0.05:
        cryo_v19_possp.set(100)
        yield from bps.sleep(2)
    
    yield from bps.sleep(5)
    while (cryo_v19.get() - 0) > 0.05:
        print('cryo cooler still refilling')
        yield from bps.sleep(5)
    cryo_v19_possp.set(0)
    print('Waiting for ', wait_time_after_v19_close, ' s ', ' before taking data...')
    yield from bps.sleep(wait_time_after_v19_close)    

