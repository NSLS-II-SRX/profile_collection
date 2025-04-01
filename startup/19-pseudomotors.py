print(f'Loading {__file__}...')


import numpy as np

from ophyd import (
    EpicsSignal,
    EpicsSignalRO,
    EpicsMotor,
    Signal,
    PseudoPositioner,
    PseudoSingle,
)

from ophyd.pseudopos import pseudo_position_argument, real_position_argument
from ophyd import Component as Cpt


# Dumb way to overwrite the hard-coded Signal class limits
class LimitedSignal(Signal):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal_limits = (0, 0)
    
    @property
    def limits(self):
        return self._signal_limits
    
    # Not as setter to help avoid non-explicit calls
    def _set_signal_limits(self, new_limits):
        new_limits = tuple(new_limits)
        if len(new_limits) != 2:
            err_str = ('Length of new limits must be 2, not '
                        + f'{len(new_limits)}.')
            raise ValueError(err_str)

        self._signal_limits = new_limits


class ProjectedTopStage(PseudoPositioner):

    # Pseudo axes
    projx = Cpt(PseudoSingle)
    projz = Cpt(PseudoSingle)

    # Real axes. From XRXNanoStage class definition.
    topx = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.RBV
    topz = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.RBV

    # Configuration signals
    th = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV')  # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV
    velocity_x = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.VELO')
    velocity_z = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.VELO')
    acceleration_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.ACCL')
    acceleration_z = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.ACCL')
    motor_egu_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.EGU')
    motor_egu_z = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.EGU')

    # Create projected signals to read
    velocity = Cpt(LimitedSignal, None, add_prefix=(), kind='config')
    acceleration = Cpt(LimitedSignal, None, add_prefix=(), kind='config')
    motor_egu = Cpt(LimitedSignal, None, add_prefix=(), kind='config')

    def __init__(self,
                 *args,
                 projected_axis=None,
                 **kwargs):

        super().__init__(*args, **kwargs)
        
        # Store projected axis for determining projected velocity
        if projected_axis is None:
            err_str = "Must define projected_axis as 'x' or 'z'."
            raise ValueError(err_str)
        elif str(projected_axis).lower() not in ['x', 'z']:
            err_str = ("ProjectedTopStage axis only supported for 'x' "
                       + f"or 'z' projected axis not {projected_axis}.")
            raise ValueError(err_str)
        self._projected_axis = str(projected_axis).lower()

        # Define defualt projected signals
        velocity = min([self.velocity_x.get(),
                        self.velocity_z.get()])
        acceleration = min([self.acceleration_x.get(),
                            self.acceleration_z.get()])
        if self.motor_egu_x.get() == self.motor_egu_z.get():
            motor_egu = self.motor_egu_x.get()
        else:
            err_str = (f'topx motor_egu of {self.motor_egu_x.get()} does '
                       + 'not match topz motor_egu of '
                       + f'{self.motor_egu_z.get()}')
            raise AttributeError(err_str)

        self.velocity.set(velocity)
        self.acceleration.set(acceleration)
        self.motor_egu.set(motor_egu)
        self.motor_egu._set_signal_limits((None, None))

        # Set velocity limits
        velocity_limits = (
            max([self.velocity_x.low_limit,
                 self.velocity_z.low_limit]),
            min([self.velocity_x.high_limit,
                 self.velocity_z.high_limit])
        )
        self.velocity._set_signal_limits(velocity_limits)

        # Set acceleration limits
        acceleration_limits = (
            max([self.acceleration_x.low_limit,
                 self.acceleration_z.low_limit]),
            min([self.acceleration_x.high_limit,
                 self.acceleration_z.high_limit])
        )
        self.acceleration._set_signal_limits(acceleration_limits)

        # Set up alias for flyer readback
        if self._projected_axis == 'x':
            self.user_readback = self.projx.readback
        else:
            self.user_readback = self.projz.readback

    # Convenience function to get rotation matrix between 
    # rotated top stage axes and projected lab axes
    def R(self):
        th = self.th.get()
        th = np.radians(th / 1000) # to radians
        return np.array([[np.cos(th), np.sin(th)],
                         [-np.sin(th), np.cos(th)]])
    

    # Function to change component motor velocities
    def set_component_velocities(self,
                                 topx_velocity=None,
                                 topz_velocity=None):
        
        bool_flags = sum([topx_velocity is None,
                          topz_velocity is None])

        if bool_flags == 1:
            err_str = ('Must specify both topx_velocity and '
                       + 'topz_velocity or neither.')
            raise ValueError(err_str)
        elif bool_flags == 2:
            # Determine component velocities from projected
            velocity = self.velocity.get()
            if self._projected_axis == 'x':
                velocity_vector = [velocity, 0]
            else:
                velocity_vector = [0, velocity]

            (topx_velocity,
             topz_velocity) = np.abs(self.R() @ velocity_vector)
        
        if topx_velocity < self.topx.velocity.low_limit:
            topx_velocity = self.topx.velocity.low_limit
        if topz_velocity < self.topz.velocity.low_limit:
            topz_velocity = self.topz.velocity.low_limit
        
        # In the background is a set_and_wait. Returning status object may not be necessary
        self.velocity_x.set(topx_velocity)
        # print(f'{topx_velocity=}')
        self.velocity_z.set(topz_velocity)
        # print(f'{topz_velocity=}')
        # print('finished changing velocities')

    
    # Wrap move function with stage_sigs-like behavior
    def move(self, *args, **kwargs):
        # Get starting velocities
        start_topx_velocity = self.velocity_x.get()
        start_topz_velocity = self.velocity_z.get()
        
        # Set component velocities based on internal velocity signal
        # print('setting velocities')
        self.set_component_velocities()

        # Move like normal
        # print('starting move')
        mv_st = super().move(*args, **kwargs)
        mv_st.wait()
        # print('move done')

        # Reset component velocities to original values
        # print('resetting velocities')
        self.set_component_velocities(
                    topx_velocity=start_topx_velocity,
                    topz_velocity=start_topz_velocity)
        
        # Must return move status object!!
        return mv_st


    def _forward(self, projx, projz):
        #     # |topx|   | cos(th)  sin(th)| |projx|
        #     # |topz| = |-sin(th)  cos(th)| |projz|
        return self.R().T @ [projx, projz]

    
    def _inverse(self, topx, topz):
        #     # |projx|   |cos(th)  -sin(th)| |topx|
        #     # |projz| = |sin(th)   cos(th)| |topz|
        return self.R() @ [topx, topz]


    @pseudo_position_argument
    def forward(self, p_pos):

        if self._projected_axis == 'x':
            projx = p_pos.projx
            self.projz.sync() # Ignore setpoint value
            projz = p_pos.projz
        else:
            projz = p_pos.projz
            self.projx.sync()
            projx = p_pos.projx
        
        topx, topz = self._forward(projx, projz)
        return self.RealPosition(topx=topx, topz=topz)


    @real_position_argument
    def inverse(self, r_pos):
        topx = r_pos.topx
        topz = r_pos.topz
        projx, projz = self._inverse(topx, topz)
        return self.PseudoPosition(projx=projx, projz=projz)


projx = ProjectedTopStage(name='projected_top_x', projected_axis='x')
projz = ProjectedTopStage(name='projected_top_z', projected_axis='z')


# class RotatedScannerStage(PseudoPositioner):

#     # Pseudo axes
#     rotx = Cpt(PseudoSingle)
#     roty = Cpt(PseudoSingle)

#     # # Real axes. From XRXNanoStage class definition.
#     # realx = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr.RBV
#     # realy = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssy}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.RBV

#     # # Configuration signals
#     # velocity_x = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr.VELO')
#     # velocity_y = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssy}Mtr.VELO')
#     # acceleration_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr.ACCL')
#     # acceleration_y = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssy}Mtr.ACCL')
#     # motor_egu_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssx}Mtr.EGU')
#     # motor_egu_y = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:ssy}Mtr.EGU')

#     # Real axes. From XRXNanoStage class definition.
#     realx = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sx}Mtr.RBV
#     realy = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.RBV

#     # Configuration signals
#     velocity_x = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.VELO')
#     velocity_y = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.VELO')
#     acceleration_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.ACCL')
#     acceleration_y = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.ACCL')
#     motor_egu_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.EGU')
#     motor_egu_y = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:sy}Mtr.EGU')

#     # Create projected signals to read
#     velocity = Cpt(LimitedSignal, None, add_prefix=(), kind='config')
#     acceleration = Cpt(LimitedSignal, None, add_prefix=(), kind='config')
#     motor_egu = Cpt(LimitedSignal, None, add_prefix=(), kind='config')

#     # Internal scan rotation
#     scan_rotation = Cpt(LimitedSignal, None, add_prefix=(), kind='config')

#     def __init__(self,
#                  *args,
#                  projected_axis='x',
#                  # linked_rotation=None,
#                  **kwargs):

#         super().__init__(*args, **kwargs)

#         # Store projected axis for determining projected velocity
#         if projected_axis is None:
#             err_str = "Must define projected_axis as 'x' or 'y'."
#             raise ValueError(err_str)
#         elif str(projected_axis).lower() not in ['x', 'y']:
#             err_str = ("ProjectedTopStage axis only supported for 'x' "
#                        + f"or 'y' projected axis not {projected_axis}.")
#             raise ValueError(err_str)
#         self._projected_axis = str(projected_axis).lower()

#         # Define defualt projected signals
#         velocity = min([self.velocity_x.get(),
#                         self.velocity_y.get()])
#         acceleration = min([self.acceleration_x.get(),
#                             self.acceleration_y.get()])
#         if self.motor_egu_x.get() == self.motor_egu_y.get():
#             motor_egu = self.motor_egu_x.get()
#         else:
#             err_str = (f'topx motor_egu of {self.motor_egu_x.get()} does '
#                        + 'not match topz motor_egu of '
#                        + f'{self.motor_egu_y.get()}')
#             raise AttributeError(err_str)

#         self.velocity.set(velocity)
#         self.acceleration.set(acceleration)
#         self.motor_egu.set(motor_egu)
#         self.motor_egu._set_signal_limits((None, None))

#         # Rotations
#         self.scan_rotation.set(0)
#         self.scan_rotation._set_signal_limits((-90, 90))

#         # Set velocity limits
#         velocity_limits = (
#             max([self.velocity_x.low_limit,
#                  self.velocity_y.low_limit]),
#             min([self.velocity_x.high_limit,
#                  self.velocity_y.high_limit])
#         )
#         self.velocity._set_signal_limits(velocity_limits)

#         # Set acceleration limits
#         acceleration_limits = (
#             max([self.acceleration_x.low_limit,
#                  self.acceleration_y.low_limit]),
#             min([self.acceleration_x.high_limit,
#                  self.acceleration_y.high_limit])
#         )
#         self.acceleration._set_signal_limits(acceleration_limits)

#         # Set up alias for flyer readback
#         if self._projected_axis == 'x':
#             self.user_readback = self.rotx.readback
#         else:
#             self.user_readback = self.roty.readback

#     # Convenience function to get rotation matrix between 
#     # rotated top stage axes and projected lab axes
#     def R(self):
#         th = self.scan_rotation.get()
#         th = np.radians(th) # to radians
#         return np.array([[np.cos(th), np.sin(th)],
#                          [-np.sin(th), np.cos(th)]])  

#     # Function to change component motor velocities
#     def set_component_velocities(self,
#                                  realx_velocity=None,
#                                  realy_velocity=None):

#         print('Setting component velocities.')
#         print(f'Starting values are \n\t{self.velocity_x.get()=}\n\t{self.velocity_y.get()=}')
        
#         bool_flags = sum([realx_velocity is None,
#                           realy_velocity is None])

#         if bool_flags == 1:
#             err_str = ('Must specify both realx_velocity and '
#                        + 'realy_velocity or neither.')
#             raise ValueError(err_str)
#         elif bool_flags == 2:
#             # Determine component velocities from projected
#             velocity = self.velocity.get()
#             if self._projected_axis == 'x':
#                 velocity_vector = [velocity, 0]
#             else:
#                 velocity_vector = [0, velocity]

#             (realx_velocity,
#              realy_velocity) = np.abs(self.R() @ velocity_vector)
        
#         if realx_velocity < self.realx.velocity.low_limit:
#             realx_velocity = self.realx.velocity.low_limit
#         if realy_velocity < self.realy.velocity.low_limit:
#             realy_velocity = self.realy.velocity.low_limit
        
#         # In the background is a set_and_wait.
#         # Returning status object may not be necessary
#         self.velocity_x.set(realx_velocity)
#         self.velocity_y.set(realy_velocity)

#         print('Finished component velocities.')
#         print(f'Values are \n\t{self.velocity_x.get()=}\n\t{self.velocity_y.get()=}')

    
#     # Wrap move function with stage_sigs-like behavior
#     def move(self, *args, **kwargs):
#         # Get starting velocities
#         start_realx_velocity = self.velocity_x.get()
#         start_realy_velocity = self.velocity_y.get()
#         # print(f'{self.realx.velocity.get()=}')
#         # print(f'{self.realy.velocity.get()=}')
        
#         # Set component velocities based on internal velocity signal
#         self.set_component_velocities()
#         # print(f'{self.realx.velocity.get()=}')
#         # print(f'{self.realy.velocity.get()=}')

#         # Move like normal
#         mv_st = super().move(*args, **kwargs)
#         mv_st.wait()

#         # Reset component velocities to original values
#         self.set_component_velocities(
#                     realx_velocity=start_realx_velocity,
#                     realy_velocity=start_realy_velocity)

#         # print(f'{self.realx.velocity.get()=}')
#         # print(f'{self.realy.velocity.get()=}')
        
#         # Must return move status object!!
#         return mv_st


#     def _forward(self, rotx, roty):
#         #     # |realx|   | cos(th)  sin(th)| |rotx|
#         #     # |realy| = |-sin(th)  cos(th)| |roty|
#         return self.R().T @ [rotx, roty]

    
#     def _inverse(self, realx, realy):
#         #     # |rotx|   |cos(th)  -sin(th)| |realx|
#         #     # |rotz| = |sin(th)   cos(th)| |realy|
#         return self.R() @ [realx, realy]


#     @pseudo_position_argument
#     def forward(self, p_pos):

#         if self._projected_axis == 'x':
#             rotx = p_pos.rotx
#             self.roty.sync() # Ignore setpoint value
#             roty = p_pos.roty
#         else:
#             roty = p_pos.roty
#             self.rotx.sync()
#             rotx = p_pos.rotx
        
#         realx, realy = self._forward(rotx, roty)
#         return self.RealPosition(realx=realx, realy=realy)


#     @real_position_argument
#     def inverse(self, r_pos):
#         realx = r_pos.realx
#         realy = r_pos.realy
#         rotx, roty = self._inverse(realx, realy)
#         return self.PseudoPosition(rotx=rotx, roty=roty)


# rotx = RotatedScannerStage(name='rot_sx', projected_axis='x')
# roty = RotatedScannerStage(name='rot_sy', projected_axis='y')