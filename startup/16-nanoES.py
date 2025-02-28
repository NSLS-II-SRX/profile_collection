print(f'Loading {__file__}...')


from ophyd import (Device, EpicsMotor, EpicsSignal, EpicsSignalRO,
                   PVPositionerPC)
from ophyd import Component as Cpt


# nano-KB mirrors
class SRXNanoKBCoarse(PVPositionerPC):
    setpoint = Cpt(EpicsSignal, 'SPOS')  # XF:05IDD-ES:1{Mir:nKBv-Ax:PC}SPOS
    readback = Cpt(EpicsSignalRO, 'RPOS')  # XF:05IDD-ES:1{Mir:nKBh-Ax:PC}RPOS


class SRXNanoKB(Device):
    # XF:05IDD-ES:1{nKB:vert-Ax:Y}Mtr.RBV
    v_y = Cpt(EpicsMotor, 'vert-Ax:Y}Mtr')
    # XF:05IDD-ES:1{nKB:vert-Ax:PC}RPOS
    v_pitch = Cpt(SRXNanoKBCoarse,
                  'XF:05IDD-ES:1{nKB:vert-Ax:PC}',
                  name='nanoKB_v_pitch',
                  add_prefix=())  
    # XF:05IDD-ES:1{nKB:horz-Ax:PC}Mtr.RBV
    v_pitch_um = Cpt(EpicsMotor, 'vert-Ax:PC}Mtr')
    # XF:05IDD-ES:1{nKB:vert-Ax:PFPI}Mtr.RBV
    v_pitch_fine = Cpt(EpicsMotor,
                       "XF:05IDD-ES:1{nKB:vert-Ax:PFPI}Mtr",
                       name="v_pitch_fine",
                       add_prefix="")
    # XF:05IDD-ES:1{nKB:horz-Ax:X}Mtr.RBV
    h_x = Cpt(EpicsMotor, 'horz-Ax:X}Mtr')
    # XF:05IDD-ES:1{nKB:horz-Ax:PC}RPOS
    h_pitch = Cpt(SRXNanoKBCoarse,
                  'XF:05IDD-ES:1{nKB:horz-Ax:PC}',
                  name='nanoKB_h_pitch',
                  add_prefix=())
    # XF:05IDD-ES:1{nKB:vert-Ax:PC}Mtr.RBV
    h_pitch_um = Cpt(EpicsMotor, 'horz-Ax:PC}Mtr')
    # XF:05IDD-ES:1{nKB:horz-Ax:PFPI}Mtr.RBV
    h_pitch_fine = Cpt(EpicsMotor,
                       "XF:05IDD-ES:1{nKB:horz-Ax:PFPI}Mtr",
                       name="h_pitch_fine",
                       add_prefix="")


nanoKB = SRXNanoKB('XF:05IDD-ES:1{nKB:', name='nanoKB')


# High flux sample stages
class SRXNanoStage(Device):
    # x = Cpt(EpicsMotor, 'sx}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr.RBV
    y = Cpt(EpicsMotor, 'sy}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.RBV
    z = Cpt(EpicsMotor, 'sz}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sz}Mtr.RBV
    sx = Cpt(EpicsMotor, 'ssx}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr.RBV
    sy = Cpt(EpicsMotor, 'ssy}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssy}Mtr.RBV
    sz = Cpt(EpicsMotor, 'ssz}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:ssz}Mtr.RBV
    th = Cpt(EpicsMotor, 'th}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV
    topx = Cpt(EpicsMotor, 'xth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.RBV
    topz = Cpt(EpicsMotor, 'zth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.RBV


nano_stage = SRXNanoStage('XF:05IDD-ES:1{nKB:Smpl-Ax:', name='nano_stage')


# nanoVLM motion
class SRXNanoVLMStage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')  # XF:05IDD-ES:1{nKB:VLM-Ax:X}Mtr.RBV
    y = Cpt(EpicsMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')


nano_vlm_stage = SRXNanoVLMStage('XF:05IDD-ES:1{nKB:VLM-Ax:',
                                 name='nano_vlm_stage')


# SDD motion
class SRXNanoDet(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')  # XF:05IDD-ES:1{nKB:Det-Ax:X}Mtr.RBV
    y = Cpt(EpicsMotor, 'Y}Mtr')  # XF:05IDD-ES:1{nKB:Det-Ax:Y}Mtr.RBV
    z = Cpt(EpicsMotor, 'Z}Mtr')  # XF:05IDD-ES:1{nKB:Det-Ax:Z}Mtr.RBV


nano_det = SRXNanoDet('XF:05IDD-ES:1{nKB:Det-Ax:', name='nano_det')


# Lakeshore temperature monitors
class SRXNanoTemp(Device):
    temp_nanoKB_horz = Cpt(EpicsSignalRO, '2}T:C-I')
    temp_nanoKB_vert = Cpt(EpicsSignalRO, '1}T:C-I')
    temp_nanoKB_base = Cpt(EpicsSignalRO, '4}T:C-I')
    temp_microKB_base = Cpt(EpicsSignalRO, '3}T:C-I')


temp_nanoKB = SRXNanoTemp('XF:05IDD-ES{LS:1-Chan:', name='temp_nanoKB')

# NanoKB inteferometer monitors
class SRXNanoKBInterferometer(Device):
    posX = Cpt(EpicsSignalRO, 'Chan0}Pos-I')
    posY = Cpt(EpicsSignalRO, 'Chan1}Pos-I')
    posZ = Cpt(EpicsSignalRO, 'Chan2}Pos-I')


nanoKB_interferometer = SRXNanoKBInterferometer('XF:05IDD-ES:1{FPS:1-', name='nanoKB_interferometer')


# Nanostage inteferometer monitors
class SRXNanoStageInterferometer(Device):
    posX = Cpt(EpicsSignalRO, 'POS_0')
    posY = Cpt(EpicsSignalRO, 'POS_1')
    posZ = Cpt(EpicsSignalRO, 'POS_2')


nano_stage_interferometer = SRXNanoStageInterferometer('XF:05IDD-ES:1{PICOSCALE:1}', name='nano_stage_interferometer')


def move_to_scanner_center(timeout=10):
    """
    Move the scanner stages to the center (zero).

    Parameters
    ----------
    timeout : integer
        Amount of time in seconds to wait

    Yields
    ------
    msg : Msg to move motors to zero
    """
    return (yield from mv(nano_stage.sx, 0,
                          nano_stage.sy, 0,
                          nano_stage.sz, 0,
                          timeout=timeout))


def reset_scanner_velocity():
    """
    Reset the scanner stages to their nominal speeds
    """
    for d in [nano_stage.topx, nano_stage.topz, nano_stage.y, nano_stage.z]:
        d.velocity.set(500)  # um/s
    for d in [nano_stage.sx, nano_stage.sy, nano_stage.sz]:
        d.velocity.set(100)  # um/s
    nano_stage.th.velocity.set(10_000)  # mdeg/s


def center_scanner():
    """
    Move the coarse and scanner stages such that the scanner is set to
    zero and the coarse stages translate to keep the sample in position
    """
    del_sx = nano_stage.sx.user_readback.get()
    del_sy = nano_stage.sy.user_readback.get()
    del_sz = nano_stage.sz.user_readback.get()

    yield from mv(nano_stage.sx, 0)
    yield from mvr(nano_stage.topx, del_sx)

    yield from mv(nano_stage.sy, 0)
    yield from mvr(nano_stage.y, del_sy)

    yield from mv(nano_stage.sz, 0)
    yield from mvr(nano_stage.z, del_sz)


def mv_along_axis(z_end):
    """
    Move the sample along the focused beam axis
    """
    cur_x = nano_stage.topx.user_readback.get()
    cur_y = nano_stage.y.user_readback.get()
    cur_z = nano_stage.z.user_readback.get()
    print('Moving z along axis.')
    print(f'\tCurrent locations are: {cur_x}, {cur_y}, {cur_z}')

    ratio_xz = 0.004875
    ratio_yz = 0.0067874

    delta_z = z_end - cur_z
    print(f'\tMove z to {z_end}')

    delta_x = ratio_xz * delta_z
    print(f'\tMove x by {delta_x}')

    delta_y = ratio_yz * delta_z
    print(f'\tMove y by {delta_y}')

    yield from mvr(nano_stage.topx, delta_x)
    yield from mvr(nano_stage.y, delta_y)
    yield from mv(nano_stage.z, z_end)


def get_defocused_beam_parameters(hor_size=None,
                                  ver_size=None,
                                  z_end=None):
    """
    Determine the new sample position and horizontal and vertical beam
    size given one of the parameters in microns.
    """
    
    # Nominal constants for defocus math in um
    n_ver_focus = 0.5
    n_ver_focal_length = 310000
    n_ver_acceptance = 600
    n_hor_focus = 0.5
    n_hor_focal_length = 130000
    n_hor_acceptance = 300

    def get_delta_z(size, focus, focal_length, acceptance):
        return focal_length * ((size - focus) / (acceptance - focus))

    def get_new_size(delta_z, focus, focal_length, acceptance):
        return focus + ((delta_z / focal_length) * (acceptance - focus))

    # Defocus math
    if z_end is None:
        curr_z = nano_stage.z.user_readback.get()
        # curr_z = 0
        if hor_size is not None:
            if hor_size < n_hor_focus:
                warn_str = ('WARNING: Requested horizontal size of '
                            + f'{hor_size} μm is less than nominal '
                            + f'horizontal focus of {n_hor_focus} μm.')
                print(warn_str)
            delta_z = get_delta_z(hor_size, n_hor_focus, n_hor_focal_length, n_hor_acceptance)
            ver_size = get_new_size(delta_z, n_ver_focus, n_ver_focal_length, n_ver_acceptance)
        elif ver_size is not None:
            if ver_size < n_ver_focus:
                warn_str = ('WARNING: Requested vertical size of '
                            + f'{ver_size} μm is less than nominal '
                            + f'vertical focus of {n_ver_focus} μm.')
                print(warn_str)
            delta_z = get_delta_z(ver_size, n_ver_focus, n_ver_focal_length, n_ver_acceptance)
            hor_size = get_new_size(delta_z, n_hor_focus, n_hor_focal_length, n_hor_acceptance)
        else:
            raise ValueError('Must define hor_size, ver_size, or z_end.')
        z_end = curr_z + delta_z
    else:
        curr_z = nano_stage.z.user_readback.get()
        # curr_z = 0
        delta_z = z_end - curr_z
        ver_size = get_new_size(delta_z, n_ver_focus, n_ver_focal_length, n_ver_acceptance)
        hor_size = get_new_size(delta_z, n_hor_focus, n_hor_focal_length, n_hor_acceptance)

    print((f'Move the sample from z = {curr_z:.0f} μm by {delta_z:.0f}'
           + f' μm to a new z = {curr_z + delta_z:.0f} μm'))
    print(('The new focal size will be approximately:'
           + f'\n\tV = {ver_size:.2f} μm\n\tH = {hor_size:.2f} μm'))
    intensity_factor = (n_ver_focus * n_hor_focus) / (ver_size * hor_size) 
    print(('Defocused intensity will be about '
           + f'{intensity_factor * 100:.1f} % of focused intensity.'))

    return curr_z, delta_z, z_end, ver_size, hor_size


def defocus_beam(hor_size=None,
                 ver_size=None,
                 z_end=None,
                 follow_with_vlm=True,
                 follow_with_sdd=True):
    """
    Move the sample, VLM, and SDD to new positions for a specified
    defocused X-ray beam.
    """
    
    # Get defocus parameters
    (curr_z,
     delta_z,
     z_end,
     ver_size,
     hor_size) = get_defocused_beam_parameters(hor_size=hor_size,
                                               ver_size=ver_size,
                                               z_end=z_end)

    # Move sample
    print('Moving sample to defocus X-ray beam...')
    yield from mv_along_axis(np.round(z_end))
    
    # Move VLM
    if follow_with_vlm:
        print('Moving VLM to new position...')
        curr_vlm_z = nano_vlm_stage.z.user_readback.get()
        print(f'\tCurrent VLM location is z = {curr_vlm_z:.3f} mm')
        # yield from mvr did not work???
        yield from mov(nano_vlm_stage.z,
                       np.round(curr_vlm_z + (delta_z / 1000), 3)) # in mm
        print(f'\tMove VLM by {delta_z / 1000:.3f} mm')

    # Move SDD along projected z axis
    if follow_with_sdd:
        print('Moving SDD to new position...')
        curr_det_x = nano_det.x.user_readback.get()
        curr_det_z = nano_det.z.user_readback.get()
        print(f'\tCurrent locations are: x = {curr_det_x:.3f} mm, z = {curr_det_z:.3f} mm')

        sdd_rot = 30 # deg
        R = np.array([[np.cos(np.radians(sdd_rot)), np.sin(np.radians(sdd_rot))],
                      [-np.sin(np.radians(sdd_rot)), np.cos(np.radians(sdd_rot))]])
        
        # Determine deltas in um. delta_x is 0
        delta_sdd_x, delta_sdd_z = R @ [0, delta_z]

        print(f'\tMove z by {delta_sdd_z / 1000:.3f} mm')
        print(f'\tMove x by {delta_sdd_x / 1000:.3f} mm.')

        # Cautious move. Always move sdd inboard first
        if delta_sdd_x < 0:
            # yield from mvr did not work???
            yield from mov(nano_det.x,
                           np.round(curr_det_x + (delta_sdd_x / 1000), 3)) # in mm
            yield from mov(nano_det.z,
                           np.round(curr_det_z + (delta_sdd_z / 1000), 3)) # in mm
        else:
            yield from mov(nano_det.z,
                           np.round(curr_det_z + (delta_sdd_z / 1000), 3)) # in mm
            yield from mov(nano_det.x,
                           np.round(curr_det_x + (delta_sdd_x / 1000), 3)) # in mm
