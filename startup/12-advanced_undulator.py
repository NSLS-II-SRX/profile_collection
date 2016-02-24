
from __future__ import print_function #Python 2.7 compatibility


#--------------------------------------------------------------
def und_combined_motion(_tilt_microrad, _taper_microm, _elev_microm):

    _ugap = 6.8

    yUU0 = _ugap/2
    yUL0 = _ugap/2
    yDU0 = _ugap/2
    yDL0 = _ugap/2
    
    real_u_elev0 = 0.02675 #mm

    undLen = 1.220 #1.5 #[m] Distance between Undulator Motion "Axes"
    halfUndLen = 0.5*undLen

    tilt_halfUndLen_mm = _tilt_microrad*halfUndLen*1.e-03
    quart_taper_mm = 0.25*_taper_microm*1.e-03

    #relative motion in laboratory frame    
    dyUU_mm =  - tilt_halfUndLen_mm - quart_taper_mm #[mm] Upstream Upper displacement 
    dyUL_mm =  - tilt_halfUndLen_mm + quart_taper_mm #[mm] Upstream Lower displacement
    dyDU_mm =    tilt_halfUndLen_mm + quart_taper_mm #[mm] Downstream Upper displacement
    dyDL_mm =    tilt_halfUndLen_mm - quart_taper_mm #[mm] Downstream Lower displacement

    #
    yUU_mm = yUU0 + dyUU_mm
    yUL_mm = yUL0 - dyUL_mm
    yDU_mm = yDU0 + dyDU_mm
    yDL_mm = yDL0 - dyDL_mm

    elev_mm = real_u_elev0 + _elev_microm*1.e-03

    print('yUU=', round(yUU_mm, 6), 'yUL=', round(yUL_mm, 6), 'yDU=', round(yDU_mm, 6), 'yUL=', round(yDL_mm, 6), 'elev=', elev_mm)
    print('elevation =', round(elev_mm, 6))

    #ud_crab_plan(pu, us_u, us_l, ds_u, ds_l, other_dets)    
    #uplan_taper_tilt=ud_crab_plan(pu, yUU_mm, yUL_mm, yDU_mm, yDL_mm, [])
    #gs.RE(uplan_taper_tilt)
    #pu.elevation.set(elev_mm)
    
    #print('done moving the undulator')

    #resKPP = 0
    return yUU_mm, yUL_mm, yDU_mm, yDL_mm, elev_mm

#def undSpecKPP(_tilt_microrad, _taper_microm, _elev_microm) will return resKPP for optimization


#*********************************Entry
if __name__ == "__main__":

    #test call
    #undSpecKPP(-50, -50, 300)
    print('load advanced undulator module')
