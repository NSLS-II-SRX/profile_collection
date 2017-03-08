from ophyd import EpicsSignal,EpicsSignalRO,Device
from ophyd import Component as Cpt

class SRXScanRecord(Device):

    class OneScan(Device):
        p1s = Cpt(EpicsSignal, 'P1-S')
        p2s = Cpt(EpicsSignal, 'P2-S')
        p1i = Cpt(EpicsSignal, 'P1-I')
        p2i = Cpt(EpicsSignal, 'P2-I')
        p1stp = Cpt(EpicsSignal, 'P1-STP')
        p2stp = Cpt(EpicsSignal, 'P2-STP')
        p1npts = Cpt(EpicsSignalRO, 'P1-NPTS')
        p2npts = Cpt(EpicsSignalRO, 'P2-NTPS')
        p1e = Cpt(EpicsSignalRO, 'P1-E')
        p2e = Cpt(EpicsSignalRO, 'P2-E')
        p1ena = Cpt(EpicsSignal, 'P1-ENA')
        p2ena = Cpt(EpicsSignal, 'P2-ENA')
        curpt = Cpt(EpicsSignal, 'CURPT')
        npts = Cpt(EpicsSignalRO, 'NPTS')
        tpp = Cpt(EpicsSignal, 'TPP')
        ena = Cpt(EpicsSignal, 'ENA')
        acq = Cpt(EpicsSignal, 'ACQ')

#    scans = [ Cpt(OneScan,'Scan'+str(i)+':') for i in range(0,8) ]
    scan0 = Cpt(OneScan,'Scan0:')
    scan1 = Cpt(OneScan,'Scan1:')
    scan2 = Cpt(OneScan,'Scan2:')
    scan3 = Cpt(OneScan,'Scan3:')
    scan4 = Cpt(OneScan,'Scan4:')
    scan5 = Cpt(OneScan,'Scan5:')
    scan6 = Cpt(OneScan,'Scan6:')
    scan7 = Cpt(OneScan,'Scan7:')

    current_scan = Cpt(EpicsSignal,'Scan:CUR')
    time_remaining = Cpt(EpicsSignal,'Scan:REMTIME')
    scanning = Cpt(EpicsSignal, 'Scan:ENA')

scanrecord = SRXScanRecord('XF:05IDA-CT{IOC:ScanBroker01}')
scanrecord.scan0.p1s.put(25.0)
scanrecord.scan0.p2s.put(20.0)
scanrecord.scan0.p1i.put(0.010)
scanrecord.scan0.p2i.put(0.010)
scanrecord.scan0.p1stp.put(3)
scanrecord.scan0.p2stp.put(3)
scanrecord.scan0.ena.put(1)
scanrecord.scan0.acq.put(1)
