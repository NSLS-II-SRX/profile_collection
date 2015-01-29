from ophyd.userapi.scan_api import Scan, AScan, DScan, Count

scan = Scan()
ascan = AScan()
ascan.default_triggers = [sclr_trig]
ascan.default_detectors = [sclr_ch1, sclr_ch2, sclr_ch3, sclr_ch4, sclr_ch5,
                           sclr_ch6]
dscan = DScan()

# Use ct as a count which is a single scan.

ct = Count()
