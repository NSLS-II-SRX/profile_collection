print(f'Loading {__file__}...')

import numpy as np
import time as ttime

from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from matplotlib import patches
from matplotlib.collections import PatchCollection


def search_and_analyze_base(search_args=[],
                            search_kwargs={},
                            search_function=None,
                            search_defocus_distance=0,
                            search_prep_function=None,
                            search_scan_id=None,
                            data_key='xs_fluor',
                            data_slice=None,
                            data_cutoff=None,
                            normalize=False, # Liveplot does not normalize values
                            data_processing_function=None,
                            integrating_function=np.sum,
                            feature_type='points',
                            attempt_edges=False,
                            plot_analysis_rois=True,
                            move_for_analysis=True,
                            analysis_prep_function=None,
                            analysis_motors=None,
                            analysis_function=None,
                            analysis_args=[],
                            analysis_kwargs={},
                            wait_time=0,
                            ):
    """
    Generalized base function for searching an area, identifying
    features, and then anlayzing found features.

    Parameters
    ----------
    search_args : iterable, optional
        Arguments given to search function. Default is an empty list.
    search_kwargs : dictionary, optional
        Keyword arguments given to search function. Default is an empty
        dictionary.
    search_function : function, optional
        Function called to search area. This function should be some
        sort of mapping function which acquires a signal useful for
        discriminating features (e.g., coarse_scan_and_fly). This
        parameter is only optional if the search_scan_id parameter is
        specified.
    search_defocus_distance : float, optional
        Relative distance in μm to move the sample to defocus the X-ray
        beam for searching. Defualt is 0 or no defocusing.
    search_prep_function : function, optional
        Function called prior to the search function to adjust any
        parameters outside the search function itself. This function 
        cannot have any inputs. Default is to not call any function.
    search_scan_id : int, optional
        Scan ID of previously acquired function to use as the search
        function. If given, only the search function and search prep
        function are disabled. Default is None and search function will
        be called instead.
    data_key : str, optional
        Key used in tiled to retrieve data
        (e.g., bs_run['stream0']['data']['xs_fluor']). Default is
        'xs_fluor'.
    data_slice : slice or iterable of slices, optional
        Information how to slice the data beyond the first two spatial
        dimensions. If data is greater than 3D, an iterable for how to
        slice each additional axis should be provided. If data key is
        'xs_fluor', only one slice is required for energy bins as each
        detector channel will be included automatically. By default
        this value looks for the first roi information in the xpress3.
    data_cutoff : float
        Number used to specify significant regions. All values greater
        than or equal will be considered.
    normalize : bool, optional
        Flag to normalize data according to scaler values. These values
        will look for 'i0' and then 'im' data keys. Normalized data
        will no longer match the liveplot values. By default this is
        set to False.
    data_processing_function : function, optional
        Function to be called on the data for any additional processing
        (e.g., median filtering to remove noise or dark-field
        subtraction). This function can only have data as the input.
        Default is to not call any function.
    integration_function : function, optional
        Function called on data to integrate region defined by data
        slicing. This function must have axis as a keyword argument
        (e.g., numpy.sum or numpy.max). Default is numpy.sum.
    feature_type : {'points', 'regions'}, optional
        Tag to indicate which type of feature to identify. 'points'
        will identify local points at least 2 pixels apart by calling
        the skimage.features.peak_local_max on significant regions.
        'regions' will identify contiguous significant pixels by
        calling skimage.measure.label on the data. Default is 'points'.
    attempt_edges : bool, optional
        Flag to adjust regions of interest to within analysis motor
        limits. If False, out of bounds regions of interest will be
        ignored instead. Default is False. 
    plot_analysis_rois : bool, optional
        Flag to plot regions of interest over data.
    move_for_analysis : bool, optional
        Flag to indicate if search motors should be moved to regions of
        interest before calling the analysis function. Default is True.
    analysis_prep_function : function, optional
        Function called prior to the analysis functions to adjust any
        parameters outside the analysis function itself. This function 
        cannot have any inputs and should probably revert any changes
        made in the search prep function. Default is to not call any
        function.
    analysis_motors : iterable, optional
        Iterable of motors used for analysis (e.g., [nano_stage.sx,
        nano_stage.sy]). This should be set to None for analysis
        functions that do not use any motors along the same dimensions
        as the search motors. Default is None.
    analysis_args : iterable, optional
        Arguments given to analysis function. If analysis motors is not
        None, the first six values might be adjusted to follow regions
        of interest following the standard SRX input (i.e., [xstart,
        xend, xnum, ystart, yend, ynum]). Default is an empty list.
    analysis_kwargs : dict, optional
        Keyword arguments given to analysis function. Default is an empty
        dictionary.
    wait_time : float, optional
        Time in seconds to wait after the search function and after
        each analysis function call. Default is 0.


    Raises
    ------
    """

    # Temp
    move_backlash = 25

    # Check for functions
    if ((search_function is None and search_scan_id is None)
         or analysis_function is None):
        err_str = 'Must provide function for both search and analysis.'
        raise ValueError(err_str)

    # Catch weird analysis
    if not move_for_analysis and analysis_motors is None:
        err_str = ('Applying a static analysis method without moving '
                   + 'there seems silly. Maybe consider moving to the '
                   + 'ROI by setting move_for_analysis to True?')
        raise RuntimeError(err_str)

    # Integrated range of data
    if data_slice is None:
        if 'xs' in globals() and hasattr(xs, 'channel01'):
            min_x = xs.channel01.mcaroi01.min_x.get()
            size_x = xs.channel01.mcaroi01.size_x.get()
            data_slice = slice(min_x, min_x + size_x)
        else:
            err_str = ('Data slice not provided and one could not be '
                       + 'constructed from XRF roi infomation.')
            raise RuntimeError(err_str)
    elif (not isinstance(data_slice, slice)
          and not all([isinstance(x, slice) for x in data_slice])):
        err_str = ('Data slice must be slice object or iterable of '
                   + 'slice objects for greater than 3D data.')
        raise TypeError(err_str)

    # Masking data
    if data_cutoff is None:
        err_str = 'Must define search cutoff value.'
        raise ValueError(err_str)
    elif data_cutoff <=1:
        err_str = 'Search cutoff value must be greater than zero.'
        raise ValueError(err_str)

    # Ensuring correction feature type
    if (not isinstance(feature_type, str)
        or feature_type.lower() not in ['points', 'regions']):
        err_str = ("Feature type must be either 'points' or 'regions'"
                    + f" not {feature_type}.")
        raise TypeError(err_str)
    else:
        feature_type = feature_type.lower()

    # Defocus X-ray beam!
    starting_z = nano_stage.z.user_readback.get()
    if search_defocus_distance != 0:
        if search_defocus_distance < 0:
            err_str = ('Negative defocus would bringing the sample '
                       + 'closer for a convergent beam. This is not '
                       + 'advised.')
            raise ValueError(err_str)        

        # Move motors
        print('Defocusing X-ray beam for search...')
        yield from defocus_beam(
                z_end=starting_z + search_defocus_distance)

    # Actually search!
    if search_scan_id is None:
        # Additional search preparation
        if search_prep_function is not None:
            print('Preparing for search...')
            yield from search_prep_function()

        # Actually search!
        yield from search_function(*search_args,
                                   **search_kwargs)
        bs_run = c[-1]
    # print(f'Running search function:')
    # print(f'\tWith args in {search_args}')
    # print(f'\tWith kwargs in {search_kwargs}')
    else:
        bs_run = c[search_scan_id]
    
    # Retrieve data from tiled
    data = _get_processed_data(bs_run,
                               data_key=data_key,
                               data_slice=data_slice,
                               normalize=normalize,
                               data_processing_function=data_processing_function,
                               integrating_function=integrating_function)
    
    # Construct ROIs
    roi_mask = data >= data_cutoff
    rois = _get_rois(data,
                     roi_mask,
                     feature_type=feature_type)

    # Find motors
    (fast_motor,
     slow_motor,
     fast_values,
     slow_values) = _generate_positions_and_motors(bs_run)

    if search_defocus_distance != 0:
        # Save parameters for offset corrections
        defocused_x = nano_stage.topx.user_readback.get()
        defocused_y = nano_stage.y.user_readback.get()
        
        print('Refocusing X-ray beam for analysis...')
        yield from defocus_beam(z_end=starting_z)
        
        # New parameters for offset corrections
        focused_x = nano_stage.topx.user_readback.get()
        focused_y = nano_stage.y.user_readback.get()

        # Adjust offsets in positions
        offset_x = focused_x - defocused_x
        offset_y = focused_y - defocused_y
        fast_offset, slow_offset = 0, 0
        # if fast_motor in [nano_stage.topx, nano_stage.sx]:
        if fast_motor == nano_stage.topx:
            fast_offset = offset_x
        #elif fast_motor in [nano_stage.y, nano_stage.sy]:
        elif fast_motor == nano_stage.y:
            fast_offset = offset_y
        # if slow_motor in [nano_stage.topx, nano_stage.sx]:
        if slow_motor == nano_stage.topx:
            slow_offset = offset_x
        # elif slow_motor in [nano_stage.y, nano_stage.sy]:
        elif slow_motor == nano_stage.y:
            slow_offset = offset_y

        fast_values += fast_offset
        slow_values += slow_offset
        # fast_values -= fast_offset
        # slow_values -= slow_offset

    # Iterate through and adjust rois
    analysis_args_list = []
    new_positions_list = []
    valid_rois = []
    fixed_rois = []
    for roi_index, roi in enumerate(rois):
        output = _generate_analysis_args(roi,
                                         analysis_args,
                                         fast_values,
                                         slow_values,
                                         analysis_motors,
                                         move_for_analysis,
                                         attempt_edges,
                                         feature_type)
        analysis_args_list.append(output[0])
        new_positions_list.append(output[1])
        valid_rois.append(output[2])
        fixed_rois.append(output[3])
    
    # Reset new_positions for easier tracking
    if not move_for_analysis:
        new_positions_list = None

    # Plot found regions and areas for analysis
    if plot_analysis_rois:
        _plot_analysis_args(bs_run.start['scan_id'],
                            data,
                            rois,
                            analysis_args_list,
                            valid_rois,
                            fixed_rois,
                            fast_values,
                            slow_values,
                            new_positions_list,
                            feature_type=feature_type,
                            analysis_motors=analysis_motors)
    
    if wait_time > 0:
        print(f'Waiting {wait_time} sec before starting ROI analysis...')
        ttime.sleep(wait_time)

    # Additional analysis preparation
    if analysis_prep_function is not None:
        print('Preparing for analysis...')
        yield from analysis_prep_function()
    
    # Go through found and valid rois
    if len(rois) > 0:
        print(f'Starting ROI analysis for {len(rois)} ROIs!')
        for roi_index, roi in enumerate(rois):
            if not valid_rois[roi_index]:
                warn_str = (f'WARNING: ROI {roi_index} range is '
                            + 'outside of analysis motor limits.'
                            + '\nSkipping this ROI.')
                print(warn_str)
                continue
            elif fixed_rois[roi_index]:
                warn_str = (f'WARNING: ROI {roi_index} range was '
                            + 'tuncated to fit within motor limits.')
                print(warn_str)

            # Move to new location with search motors, if called
            if move_for_analysis:
                (fast_position,
                slow_position) = new_positions_list[roi_index]

                # Actually move!
                print((f'Moving to new position for ROI {roi_index}:'
                    + f'\n\t{fast_motor.name} = {fast_position}'
                    + f'\n\t{slow_motor.name} = {slow_position}'))
                # Backlash
                if move_backlash > 0:
                    yield from mov(fast_motor,
                                fast_position - move_backlash,
                                slow_motor,
                                slow_position - move_backlash
                                )
                # Move
                yield from mov(fast_motor, fast_position,
                            slow_motor, slow_position
                            )
                # print((f'Moving to new position for ROI {roi_index}:'
                #       + f'\n\tx={fast_position}'
                #       + f'\n\ty={slow_position}'))  

            # Actually search!
            print(f'Starting analysis of ROI {roi_index}!')
            yield from analysis_function(*analysis_args_list[roi_index],
                                        **analysis_kwargs)
            # print(f'Running analysis of ROI {roi_index}:')
            # print(f'\tWith args in {analysis_args_list[roi_index]}')
            # print(f'\tWith kwargs in {analysis_kwargs}')
            
            if wait_time > 0 and roi_index != len(rois) - 1:
                print(f'Waiting {wait_time} sec before proceeding...')
                ttime.sleep(wait_time)
    else:
        print('No ROIs found in search area!')


def _get_processed_data(bs_run,
                        data_key='xs_fluor',
                        data_slice=None,
                        normalize=False,
                        data_processing_function=None,
                        integrating_function=np.sum,
                        ):
    
    print((f"Retrieving {data_key} data from scan "
           + f"{bs_run.start['scan_id']}."))
    
    if data_key == 'xs_fluor' and isinstance(data_slice, slice):
        data_slice = (slice(None), data_slice)

    data = bs_run['stream0']['data'][data_key][:, :, (data_slice)] # I would normally unpack this, but apparently that is only for python 3.11
    if normalize:
        for sclr_key in ['i0', 'im']:
            if sclr_key in bs_run['stream0']['data']:
                data /= bs_run['stream0']['data'][sclr_key][:]
                normalized = True
                break
        if not normalized:
            warn_str = ("WARNING: Could not find expected scaler value"
                        + " with key 'i0' or 'im'.\n"
                        + "Proceeding without changes.")
            print(warn_str)
    
    # User-defined data processing
    if data_processing_function is not None:
        # e.g., median_filter, and/or subtract dark-field from XRD
        data = data_processing_function(data) 

    if hasattr(data_slice, '__len__'):
        axis = tuple(-np.arange(1, len(data_slice) + 1, 1)[::-1])
    else:
        axis = -1

    data = integrating_function(data, axis=axis) # e.g., numpy.sum or numpy.max

    return data


def _get_rois(data,
              roi_mask,
              feature_type='points'):

    if feature_type == 'points':
        rois = peak_local_max(data,
                              labels=roi_mask,
                              min_distance=2, # At least some padding
                              num_peaks_per_label=np.inf)

    elif feature_type == 'regions':
        rois = regionprops(label(roi_mask))
    
    return rois
    

def _generate_positions_and_motors(bs_run):
    
    # Get motors
    if bs_run.start['scan']['type'] != 'XRF_FLY':
        err_str = ("Only XRF_Fly scans are currently implemented not "
                   + f"{bs_run.start['scan']['type']}")
        raise NotImplementedError(err_str)

    # Get motors
    motor_names = [getattr(nano_stage, cpt).name
                    for cpt in nano_stage.component_names]
    fast_motor = getattr(nano_stage,
            nano_stage.component_names[
                motor_names.index(
                    bs_run.start['scan']['fast_axis']['motor_name'])])
    slow_motor = getattr(nano_stage,
            nano_stage.component_names[
                motor_names.index(
                    bs_run.start['scan']['slow_axis']['motor_name'])])
        
    # Get values
    # Assume standard SRX input
    scan_input = bs_run.start['scan']['scan_input']
    fast_values = np.linspace(*scan_input[:2], int(scan_input[2]))
    slow_values = np.linspace(*scan_input[3:5], int(scan_input[5]))

    return fast_motor, slow_motor, fast_values, slow_values


def _generate_analysis_args(roi,
                            analysis_args,
                            fast_values,
                            slow_values,
                            analysis_motors,
                            move_for_analysis,
                            attempt_edges,
                            feature_type):

    # Get ROIs in real values
    if feature_type == 'points':
        roi_fast = fast_values[roi[1]]
        # roi_slow = slow_values[::-1][roi[0]]
        roi_slow = slow_values[roi[0]]
    elif feature_type == 'regions':
        roi_fast = fast_values[roi.slice[1]]
        # roi_slow = slow_values[::-1][roi.slice[0]]
        roi_slow = slow_values[roi.slice[0]]

    # Modify analysis_args
    VALID_ROI_RANGE = True
    FIXED_ROI_RANGE = False
    if analysis_motors is not None:
        fstart, fend, fnum, sstart, send, snum = analysis_args[:6]
        other_args = analysis_args[6:]

        new_analysis_args = []
        new_positions = [] # Will be [None, None] if not move_for_analysis
        for i, (start, end, num, roi_vals) in enumerate(
                                ((fstart, fend, fnum, roi_fast),
                                 (sstart, send, snum, roi_slow))):
            ext = end - start
            step = ext / (num - 1)
            roi_start = np.min(roi_vals)
            roi_end = np.max(roi_vals)
                
            # Handle move differences
            if move_for_analysis:
                roi_ext = roi_end - roi_start
                roi_cen = roi_start + (roi_ext / 2)
                new_positions.append(np.round(roi_cen, 3))

                new_ext = ext + roi_ext
                new_start = -new_ext / 2
                new_end = new_ext / 2
                
            else:
                new_start = roi_start - (ext / 2)
                new_end = roi_end + (ext / 2)
                new_positions.append(None)

            # Check args are within scan limits                
            motor = analysis_motors[i]
            _safety_factor = 0.95
            if motor.low_limit != motor.high_limit:
                # print(f'Start {new_start} and End {new_end}')
                if new_end < motor.low_limit * _safety_factor:
                    # Cannot fix this situation. Should be very rare.
                    VALID_ROI_RANGE = False
                elif new_start < motor.low_limit * _safety_factor:
                    if attempt_edges:
                        # Closest value within limit maintaining step
                        # print('Fixing ROI')
                        # print(f'Old start is {new_start}')
                        new_start = -np.arange(
                                    -new_end,
                                    -motor.low_limit * _safety_factor,
                                    step)[-1]
                        FIXED_ROI_RANGE = True
                        # print(f'New start is {new_start}')
                    else:
                        VALID_ROI_RANGE = False

                if new_start > motor.high_limit * _safety_factor:
                    # Cannot fix this situation. Should be very rare.
                    VALID_ROI_RANGE = False
                elif new_end > motor.high_limit * _safety_factor:
                    if attempt_edges:
                        # Closest value within limit maintaining step
                        # print('Fixing ROI')
                        # print(f'Old end is {new_end}')
                        new_end = np.arange(
                                    new_start,
                                    motor.high_limit * _safety_factor,
                                    step)[-1]
                        FIXED_ROI_RANGE = True
                        # print(f'New end is {new_end}')
                    else:
                        VALID_ROI_RANGE = False

            # Get new number of points
            new_start = np.round(new_start, 3)
            new_end = np.round(new_end, 3)
            new_num = (new_end - new_start) / step + 1
            # Paranoid check
            if new_num - np.round(new_num) > 0.001:
                warn_str = ('WARNING: Something funny happened with '
                            + 'the new step number')
                print(f'{new_start=}')
                print(f'{new_end=}')
                print(f'{step=}')
                print(f'{new_num=}')
                print(warn_str)
            new_num = int(new_num)

            # Update the new args
            new_analysis_args += [new_start,
                                  new_end,
                                  new_num]
        
        # Update to full new_analysis_args
        new_analysis_args += other_args
    
    else: # Static analysis
        new_analysis_args = analysis_args
        if feature_type == 'points':
            new_positions = [np.round(roi_fast, 3),
                             np.round(roi_slow, 3)]
        elif feature_type == 'regions':
            new_positions = [np.round(np.mean(roi_fast), 3),
                             np.round(np.mean(roi_slow), 3)]
    
    return (new_analysis_args,
            new_positions,
            VALID_ROI_RANGE,
            FIXED_ROI_RANGE)


def _plot_analysis_args(scan_id,
                        data,
                        rois,
                        analysis_args_list,
                        valid_rois,
                        fixed_rois,
                        x_vals,
                        y_vals,
                        new_positions_list=None,
                        feature_type='points',
                        analysis_motors=None):

    fig, ax = plt.subplots()
    x_step = np.median(np.diff(x_vals))
    y_step = np.median(np.diff(y_vals))
    # extent = [np.min(x_vals) - (x_step / 2),
    #           np.max(x_vals) + (x_step / 2),
    #           np.min(y_vals) - (y_step / 2),
    #           np.max(y_vals) + (y_step / 2)]
    # Invert y axis
    extent = [np.min(x_vals) - (x_step / 2),
              np.max(x_vals) + (x_step / 2),
              np.max(y_vals) + (y_step / 2),
              np.min(y_vals) - (y_step / 2)]
    im = ax.imshow(data, extent=extent)
    fig.colorbar(im, ax=ax)
    
    ax.set_aspect('equal')
    ax.set_xlabel('Fast Axis [μm]')
    ax.set_ylabel('Slow Axis [μm]')
    ax.set_title(f'scan{scan_id}: Found ROIs')

    colors = []
    for valid, fixed in zip(valid_rois, fixed_rois):
        if valid:
            if fixed:
                colors.append('yellow')
            else:
                colors.append('lime')
        else:
            colors.append('red')

    # Plot found ROIs
    if feature_type == 'points':
        x_plot = x_vals[rois[:, 1]]
        # y_plot = y_vals[::-1][rois[:, 0]]
        y_plot = y_vals[rois[:, 0]]

        ax.scatter(x_plot,
                   y_plot,
                   c=colors,
                   marker='+',
                   s=100,
                   label='ROIs')
    
    elif feature_type == 'regions':
        rect_list = []
        for ind, roi in enumerate(rois):
            x_plot = x_vals[roi.slice[1]]
            # y_plot = y_vals[::-1][roi.slice[0]]
            y_plot = y_vals[roi.slice[0]]

            rect = patches.Rectangle(
                        (np.min(x_plot) - (x_step / 2),
                         np.min(y_plot) - (y_step / 2)),
                        np.max(x_plot) - np.min(x_plot) + x_step,
                        np.max(y_plot) - np.min(y_plot) + y_step,
                        linewidth=1.5,
                        linestyle='--',
                        edgecolor=colors[ind],
                        facecolor='none')
            rect_list.append(rect)
        pc = PatchCollection(rect_list,
                             match_original=True,
                             label='ROIs')
        ax.add_collection(pc)

    # Plot mapped regions around ROIs
    if analysis_motors is not None:
        rect_list = []
        for ind, args in enumerate(analysis_args_list):
            fast_step = (args[1] - args[0] - 1) / args[2]
            slow_step = (args[4] - args[3] - 1) / args[5]

            if new_positions_list is not None:
                fast_move, slow_move = new_positions_list[ind]
            else:
                fast_move, slow_move = 0, 0

            rect = patches.Rectangle(
                        (args[0] - (fast_step / 2) + fast_move,
                         args[3] - (slow_step / 2) + slow_move),
                        args[1] - args[0] + fast_step,
                        args[4] - args[3] + slow_step,
                        linewidth=2,
                        linestyle='-',
                        edgecolor=colors[ind],
                        facecolor='none')
            rect_list.append(rect)
        pc = PatchCollection(rect_list,
                             match_original=True,
                             label='Analysis')
        ax.add_collection(pc)

    # Plot static center of regions. Should be uncommon
    elif feature_type == 'regions':
        ax.scatter(np.asarray(new_positions_list)[:, 0],
                   np.asarray(new_positions_list)[:, 1],
                   c=colors,
                   marker='+',
                   s=100,
                   label='Analysis')
    
    # ax.legend()
    fig.show()





# Convenience Wrappers

def coarse_xrf_search_and_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=coarse_scan_and_fly,
                data_key='xs_fluor',
                move_for_analysis=False,
                analysis_motors=[nano_stage.topx, nano_stage.y],
                analysis_function=coarse_scan_and_fly
                )

def coarse_xrf_search_and_nano_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=coarse_scan_and_fly,
                data_key='xs_fluor',
                move_for_analysis=True,
                analysis_motors=[nano_stage.sx, nano_stage.sy],
                analysis_function=nano_scan_and_fly
                )

def coarse_xrf_search_and_xanes_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=coarse_scan_and_fly,
                data_key='xs_fluor',
                move_for_analysis=True,
                analysis_motors=None,
                analysis_function=xanes_plan
                )

def nano_xrf_search_and_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=nano_scan_and_fly,
                data_key='xs_fluor',
                move_for_analysis=False,
                analysis_motors=[nano_stage.sx, nano_stage.sy],
                analysis_function=nano_scan_and_fly
                )

def nano_xrf_search_and_xanes_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=nano_scan_and_fly,
                data_key='xs_fluor',
                move_for_analysis=True,
                analysis_motors=None,
                analysis_function=xanes_plan
                )



# Example nested scan
# def coarse_nno_xrf_search_xanes_analysze():

#     # Sub_function
#     def sub_search(*args):
#         yield from nano_xrf_search_and_xanes_analyze(search_args=args,
#                                                     data_cutoff=2000,
#                                                     # data_key='xs_fluor',
#                                                     feature_type='regions',
#                                                     # move_for_analysis=True,
#                                                     analysis_args=[[X-50, X-10, X+25, X+150],
#                                                                     [2, 1, 3],
#                                                                     0.25],
#                                                     wait_time=1)


#     yield from search_and_analyze_base(search_args=[-1160, -1100, 21, 1900, 1960, 21, 0.05],
#                                        search_function=coarse_scan_and_fly,
#                                        data_cutoff=200,
#                                        data_key='xs_fluor',
#                                        search_defocus_distance=1200,
#                                        feature_type='regions',
#                                        move_for_analysis=True,
#                                        analysis_args=[-2.5, 2.5, 21, -2.5, 2.5, 21, 0.1],
#                                        analysis_function=sub_search,
#                                        analysis_motors=[nano_stage.sx, nano_stage.sy],
#                                        wait_time=1)



