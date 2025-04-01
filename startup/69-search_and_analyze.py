print(f'Loading {__file__}...')

import numpy as np
import time as ttime

from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from matplotlib import patches
from matplotlib.collections import PatchCollection


# Assumption of [start, stop, num] for function arguments is built into several places.

def search_and_analyze_base(search_args=[],
                            search_kwargs={},
                            search_function=None,
                            search_motors=None,
                            search_scan_id=None,
                            search_prep_function=None,
                            defocus_distances=None,
                            data_key='xs_fluor',
                            data_slice=None,
                            data_cutoff=None,
                            normalize=False, # Liveplot does not normalize values
                            data_processing_function=None,
                            integrating_function=np.sum,
                            feature_type='points',
                            point_method='com',
                            fix_edges=False,
                            max_num_rois=None,
                            plot_analysis_rois=True,
                            analysis_args=[],
                            analysis_kwargs={},
                            analysis_function=None,
                            analysis_motors=None,
                            analysis_prep_function=None,
                            wait_time=0,
                            ):
    """
    Generalized base function for searching an area, identifying
    features, and then anlayzing found features.

    Parameters
    ----------
    search_args : iterable, optional
        Arguments given to search function. For motion along an axis,
        arguments should be [start, stop, end] repeated for each new
        axis of motion corresponding to the order of search_motors.
        Default is an empty list.
    search_kwargs : dictionary, optional
        Keyword arguments given to search function. Default is an empty
        dictionary.
    search_function : function, optional
        Function called to search along a set of motors. Data should be
        acquired to discriminate features (e.g., coarse_scan_and_fly).
        This parameter is only optional if the search_scan_id parameter
        is specified.
    search_motors : iterable, optional
        Iterable of motors used for search (e.g., [nano_stage.sx,
        nano_stage.sy]) matching the order of the search arguments.
        This should only be set to None if search_scan_id is specified.
        Default is None.
    search_scan_id : int, optional
        Scan ID of previously acquired function to use as the search
        function. If given, only the search function and search prep
        function are disabled. Default is None and search function will
        be called instead.
    search_prep_function : function, optional
        Function called prior to the search function to adjust any
        parameters outside the search function itself. This function 
        cannot have any inputs. Default is to not call any function.    
    defocus_distances : iterable of length 2, optional
        Relative distances in μm to move the sample from the current 
        position to defocus the X-ray beam. If only one value is given,
        it will be used for only the search defocus distance. Default
        is None which will be changed to (0, 0) or no defocusing for
        both search and analysis.
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
    point_method : {'multiple', 'center', 'com', 'max'}, optional
        Method used for determine single points from regions. This is
        used for both determining points when feaure_type is set to
        'points' and is used to determine the static position to move
        the motors if feature_type is 'regions' and the search and
        analysis motors are not the same. If 'mulitple' is chosen with
        a feature_type of 'regions', the value will defualt to
        'center'. 'com' is center of mass. Default value is 'center'.
    fix_edges : bool, optional
        Flag to adjust regions of interest to within analysis motor
        limits. If False, out of bounds regions of interest will be
        ignored instead. Default is False.
    max_num_rois : int, optional
        Maximum number of ROIs to be analyzed from the search data.
        Default is None or no limit.
    plot_analysis_rois : bool, optional
        Flag to plot regions of interest over data.
    analysis_function :
        Function called to analyze ROIs in the dataset. 
    analysis_args : iterable, optional
        Arguments given to analysis function. For motion along an axis
        used during search, arguments should be [start, stop, num]
        repeated for each new axis of motion corresponding to the order
        of the analysis motors. Default is an empty list.
    analysis_kwargs : dict, optional
        Keyword arguments given to analysis function. Default is an empty
        dictionary.
    analysis_motors : iterable, optional
        Iterable of motors used for analysis (e.g., [nano_stage.sx,
        nano_stage.sy]) matching the order of the analysis arguments.
        This should be set to None for analysis functions that do not
        use any motors along the same dimensions as the search motors.
        Default is None.
    analysis_prep_function : function, optional
        Function called prior to the analysis functions to adjust any
        parameters outside the analysis function itself. This function 
        cannot have any inputs and should probably revert any changes
        made in the search prep function. Default is to not call any
        function.
    wait_time : float, optional
        Time in seconds to wait after the search function and after
        each analysis function call. Default is 0.

    Raises
    ------
    """

    # Hard-coded but easily changed parameters
    move_backlash = 50
    motors_with_backlash = [nano_stage.topx, nano_stage.y] # this can include more motors...
    # Must be given in x, y order.
    defocus_offset_motors = [nano_stage.topx, nano_stage.y]

    # Check for functions
    if ((search_function is None and search_scan_id is None)
         or analysis_function is None):
        err_str = 'Must provide function for both search and analysis.'
        raise ValueError(err_str)

    # # Catch weird analysis
    if search_motors is None and search_scan_id is None:
        err_str('Search motors must be provided unless call a '
                + 'previously used search scan.')
        raise ValueError(err_str)

    # Adjusting slicing
    if data_slice is None:
        if data_key == 'xs_fluor':
            if 'xs' in globals() and hasattr(xs, 'channel01'):
                min_x = xs.channel01.mcaroi01.min_x.get()
                size_x = xs.channel01.mcaroi01.size_x.get()
                data_slice = slice(min_x, min_x + size_x)
            else:
                err_str = ('Data slice not provided and one could not'
                        + ' be constructed from XRF roi infomation.')
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
    
    # Adjusting defocus distances
    start_z = nano_stage.z.user_readback.get()
    if defocus_distances is None:
        defocus_distances = (0, 0)
    elif isinstance(defocus_distances, (int, float)):
        defocus_distances = (defocus_distances, 0)
    elif (hasattr(defocus_distances, '__len__')
          and len(defocus_distances) == 2
          and not isinstance(defocus_distances, (str, dict))):
        pass
    else:
        err_str = ('Cannot handle defocuses distances of type '
                   + f'{type(defocus_distances)}. Should be iterable '
                   + 'of length 2.')
        raise TypeError(err_str)

    # Ensuring correct feature type
    if (not isinstance(feature_type, str)
        or feature_type.lower() not in ['points', 'regions']):
        err_str = ("Feature type must be either 'points' or 'regions'"
                    + f" not {feature_type}.")
        raise TypeError(err_str)
    else:
        feature_type = feature_type.lower()
    
    # Ensure correct point method
    if (not isinstance(point_method, str)
        or point_method.lower() not in ['multiple', 'center', 'com', 'max']):
        err_str = ("Point method must be either 'multiple', 'center', "
                   + f"'com', or 'max' not {point_method}.")
        raise TypeError(err_str)
    else:
        point_method = point_method.lower()
        if feature_type == 'regions' and point_method == 'multiple':
            warn_str = ("WARNING: 'multiple' point method cannot be "
                        + "used simultaneously with 'regions' feature "
                        + "type. 'center' point method will be used "
                        + "instead.")
            print(warn_str)
            point_method = 'center'
    
    # Check max_roi_num
    if max_num_rois is not None and not isinstance(max_num_rois, int):
        if isinstance(max_num_rois, float) and max_num_rois == int(max_num_rois):
            max_num_rois = int(max_num_rois)
        else:
            raise TypeError("'max_num_rois' must be integer.")

    # Sort axes
    search_axes, analysis_axes = None, None
    if search_motors is not None:
        search_axes = [_get_motor_axis(motor)
                       for motor in search_motors]
    if analysis_motors is not None:
        analysis_axes = [_get_motor_axis(motor)
                         for motor in analysis_motors]

    # Or get search motors from previous scan
    if search_scan_id is not None:
        search_bs_run = c[search_scan_id]
        search_uid = search_bs_run.start['uid']
        # Attempt to get search motors and args from start
        if search_motors or len(search_args) is None:
            search_motors = _get_motors_from_tiled(search_bs_run)
            search_axes = [_get_motor_axis(motor)
                           for motor in search_motors]
            search_args = search_bs_run.start['scan']['scan_input']
    
    # Must have search motors by this point
    if search_motors is None:
        err_str = ('Search motors not provided or cannot be determined.'
                   + 'These must be provided or consider a different '
                   + 'type of scan.')
        raise ValueError(err_str)

    # Defocus X-ray beam!
    # start_x = nano_stage.topx.user_readback.get() # hard-coded from mv_along_axis
    # start_y = nano_stage.y.user_readback.get() # hard-coded from mv_along_axis
    start_positions = [motor.user_readback.get() for motor in defocus_offset_motors]
    print(f'{start_positions=}')

    search_offsets = [0, 0]
    if defocus_distances[0] != 0:
        if defocus_distances[0] < 0:
            err_str = ('Negative defocus would bringing the sample '
                       + 'closer for a convergent beam. This is not '
                       + 'advised.')
            raise ValueError(err_str)        

        # Move motors
        print('Defocusing X-ray beam for search...')
        start_positions = [motor.user_readback.get() for motor in defocus_offset_motors]
        yield from defocus_beam(z_end=start_z + defocus_distances[0])
        search_positions = [motor.user_readback.get() for motor in defocus_offset_motors]
        search_offsets = [start - search for (start, search) in zip(start_positions, search_positions)]
        
    # Actually search!
    # search_x = nano_stage.topx.user_readback.get() # hard-coded from mv_along_axis
    # search_y = nano_stage.y.user_readback.get() # hard-coded from mv_along_axis
    
    print(f'{search_positions=}')
    if search_scan_id is None:
        # Additional search preparation
        if search_prep_function is not None:
            print('Preparing for search...')
            yield from search_prep_function()

        # Adjust search arguments
        # offset_x = start_x - search_x
        # offset_y = start_y - search_y
        # Offsets are applied for any motor along the same axis
        if search_axes is not None:
            for i in range(len(search_axes)):
                args = np.asarray(search_args[3 * i : 3 * i + 2])
                if search_axes[i] == 'x':
                    search_args[3 * i : 3 * i + 2] = np.round(args + search_offsets[0], 3)
                elif search_axes[i] == 'y':
                    search_args[3 * i : 3 * i + 2] = np.round(args + search_offsets[1], 3)
        
        # Apply backlash for search
        search_backlash_args = []
        search_backlash_str = 'Adding backlash to search motors:'
        for i, s_motor in enumerate(search_motors):
            if s_motor in motors_with_backlash:
                search_backlash_args.extend([s_motor, search_args[3 * i] - move_backlash])
                search_backlash_str += f'\n\t{s_motor.name} = {search_args[3 * i] - move_backlash}'
        if len(search_backlash_args) > 0:
            print(search_backlash_str)
            yield from mov(*search_backlash_args)

        # Actually search!
        # no guarantee the search function will return a uid...
        search_uid = yield from search_function(*search_args,
                                                **search_kwargs)

        # print(search_uid)
        # print(type(search_uid))

        # search_bs_run = c[search_uid]
        search_bs_run = c[-1]
    
    # Retrieve data from tiled
    data = _get_processed_data(search_bs_run,
                               data_key=data_key,
                               data_slice=data_slice,
                               normalize=normalize,
                               data_processing_function=data_processing_function,
                               integrating_function=integrating_function)
    
    # Check and fix data shape
    if len(search_motors) > data.ndim:
        err_str = ('More search motors were called there are seach axes'
                   + ' in the data. Either too many search motors have'
                   + ' been indicated or something went very wrong.')
        raise RuntimeError(err_str)
    elif data.ndim > len(search_motors):
        if 1 in data.shape:
            data = data.squeeze()
        else:
            err_str = ('More search dimensions were found in the data '
                       + 'than search motors have been given. Either a'
                       + ' search motor is missing or something went '
                       + 'very wrong.')
            raise RuntimeError(err_str)
    else: # data.ndim matches search motors (majority of cases)
        remove_search_index, remove_search_args_index = [], []
        for s_ind in range(len(search_motors)):
            # Are there any motors that do not move?
            if data.shape[::-1][s_ind] == 1: # Reversed to match XRF_FLY data indices (Slow X Fast)
                print(s_ind)
                data_ndims = data.ndim
                # Is that motor called during analysis?
                if search_motors[s_ind] not in analysis_motors:
                    print('Search_motor not in analysis!')
                    # If not, associated values can be removed from search dimensions
                    remove_search_index.append(s_ind)
                    remove_search_args_index.extend(
                            list(range(3 * s_ind, 3 * (s_ind + 1))))

        # Remove search values that do not move and are not required for analysis
        if len(remove_search_index) > 0:
            for ind in sorted(remove_search_index, reverse=True):
                del search_motors[ind]
                del search_axes[ind]
                data = data.squeeze(axis=data_ndims - ind - 1) # Reversed to match XRF_FLY data indices (Slow X Fast)
            for ind in sorted(remove_search_args_index, reverse=True):
                del search_args[ind]
            
    # Construct ROIs
    roi_mask = data >= data_cutoff
    rois, roi_ints = _get_rois(data,
                        roi_mask,
                        feature_type=feature_type)

    # Refocus and update positions
    analysis_offsets = [0, 0]
    if defocus_distances[0] != defocus_distances[1]:
        print('Refocusing X-ray beam for analysis...')
        start_positions = [motor.user_readback.get() for motor in defocus_offset_motors]
        yield from defocus_beam(z_end=start_z + defocus_distances[1])
        analysis_positions = [motor.user_readback.get() for motor in defocus_offset_motors]
        # Opposite of search offsets
        analysis_offsets = [analysis - start for (start, analysis) in zip(start_positions, analysis_positions)]

    # analysis_x = nano_stage.topx.user_readback.get()  # hard-coded from mv_along_axis
    # analysis_y = nano_stage.y.user_readback.get() # hard-coded from mv_along_axis
    
    print(f'{analysis_positions=}')
    position_values = [None,] * len(search_axes)
    # print(f'{search_axes=}')
    # print(f'{search_args=}')
    if search_axes is not None:
        # offset_x = analysis_x - search_x
        # offset_y = analysis_y - search_y
        
        print(f'{analysis_offsets=}')
        for i in range(len(search_axes)):
            if search_axes[i] is not None:
                vals = np.linspace(*search_args[3 * i : 3 * i + 2],
                                   int(search_args[3 * i + 2]))

                if search_axes[i] == 'x' and search_motors[i] in defocus_offset_motors:
                    position_values[i] = vals + analysis_offsets[0]
                    print('Applying analysis correction in x')
                elif search_axes[i] == 'y' and search_motors[i] in defocus_offset_motors:
                    position_values[i] = vals + analysis_offsets[1]
                    print('Applying analysis correction in y')
                else:
                    position_values[i] = vals

    # Iterate through and adjust rois
    analysis_args_list = []
    move_positions_list = []
    valid_rois = []
    fixed_rois = []
    for roi_index, roi in enumerate(rois):
        output = _generate_analysis_args(roi,
                                         # data,
                                         analysis_args,
                                         search_motors,
                                         search_axes,
                                         analysis_motors,
                                         analysis_axes,
                                         position_values,
                                         fix_edges,
                                         feature_type,
                                         point_method)
        analysis_args_list.append(output[0])
        move_positions_list.append(output[1])
        valid_rois.append(output[2])
        fixed_rois.append(output[3])

    # Trim rois
    if (max_num_rois is not None
        and sum(valid_rois) > max_num_rois):
        ostr = (f'Number of ROIs {sum(valid_rois)} is greater than '
                + f'requested maximum {max_num_rois}. Trimming ROIs to'
                + ' only the greatest intensity.')
        print(ostr)

        # sorted_ints = sorted(rois_int[valid_rois])
        sorted_ints = sorted([val for (val, cond)
                              in zip(roi_ints, valid_rois) if cond],
                             reverse=True)

        for roi_ind in range(len(rois)):
            if not valid_rois[roi_ind]:
                continue
            if roi_ints[roi_ind] not in sorted_ints[:max_num_rois]:
                valid_rois[roi_ind] = False

    # Plot found regions and areas for analysis
    if plot_analysis_rois:
        _plot_analysis_args(search_bs_run.start['scan_id'],
                            data,
                            rois,
                            analysis_args_list,
                            valid_rois,
                            fixed_rois,
                            position_values,
                            move_positions_list,
                            search_motors,
                            search_axes,
                            analysis_motors,
                            analysis_axes,
                            feature_type)
    
    if wait_time > 0:
        print(f'Waiting {wait_time} sec before starting ROI analysis...')
        bps.sleep(wait_time)

    # Additional analysis preparation
    if analysis_prep_function is not None:
        print('Preparing for analysis...')
        yield from analysis_prep_function()
    
    print(f'Position values of 0 range {np.min(position_values[0])}-{np.max(position_values[0])}')
    print(f'Position values of 1 range {np.min(position_values[1])}-{np.max(position_values[1])}')
    print(analysis_args_list)
    print(move_positions_list)
    
    # return
    
    # Go through found and valid rois
    analysis_uids = []
    if len(rois) > 0:
        print(f'Starting ROI analysis for {len(rois)} ROIs!')
        for roi_index, roi in enumerate(rois):
            if not valid_rois[roi_index]:
                note_str = (f'ROI {roi_index} is either outside motor range or exceeds maximum ROI number.'
                            + '\nSkipping this ROI.')
                print(note_str)
                continue
            elif fixed_rois[roi_index]:
                warn_str = (f'WARNING: ROI {roi_index} range was '
                            + 'tuncated to fit within motor limits.')
                print(warn_str)

            # Setup move call
            move_positions = move_positions_list[roi_index]
            if any([pos is not None for pos in move_positions]):
                mv_str = f'Moving to new position for ROI {roi_index}:'
                move_args = []
                backlash_args = []

                for p_ind, pos in enumerate(move_positions):
                    # Check if move is actually called
                    if pos is None:
                        continue
                    
                    # Add to move call
                    s_motor = search_motors[p_ind]
                    mv_str += f'\n\t{s_motor.name} = {pos}'
                    move_args += [s_motor, pos]

                    # Determine backlash
                    if s_motor in motors_with_backlash:
                        backlash_args += [s_motor, pos - move_backlash]
                
                # Actually move!
                print(mv_str)
                if len(backlash_args) > 0:
                    yield from mov(*backlash_args)
                yield from mov(*move_args)

            # Actually search!
            print(f'Starting analysis of ROI {roi_index}!')
            uid = yield from analysis_function(*analysis_args_list[roi_index],
                                               **analysis_kwargs)
            analysis_uids.append(uid)
            
            if wait_time > 0 and roi_index != len(rois) - 1:
                print(f'Waiting {wait_time} sec before proceeding...')
                bps.sleep(wait_time)
    else:
        print('No ROIs found in search area!')
    
    return search_uid, tuple(analysis_uids)


def _get_motor_axis(motor):

    if motor is None:
        axis = None
    if not hasattr(motor, 'name'):
        raise AttributeError(f'Motor does not have a name.')
    if 'x' in motor.name.split('_')[-1]:
        axis = 'x'
    elif 'y' in motor.name.split('_')[-1]:
        axis = 'y'
    elif 'z' in motor.name.split('_')[-1]:
        axis = 'z'
    elif 'th' in motor.name.split('_')[-1]:
        axis = 'theta'
    elif 'energy' in motor.name.split('_')[-1]:
        axis = 'energy'
    else:
        raise ValueError(f'Unknown motor {motor.name}.')

    return axis


def _get_processed_data(bs_run,
                        data_key='xs_fluor',
                        data_slice=None,
                        normalize=False,
                        data_processing_function=None,
                        integrating_function=np.sum,
                        ):
    
    print((f"Retrieving {data_key} data from scan "
           + f"{bs_run.start['scan_id']}."))

    # Determine data path
    # TODO: Is there a better way of determining this pathway?
    if bs_run.start['scan']['type'] in ['XRF_FLY']:
        ds = bs_run['stream0']['data']
    elif bs_run.start['scan']['type'] in ['ENERGY_RC', 'ANGLE_RC', 'XAS_STEP']:
        ds = bs_run['primary']['data']
    else:
        err_str = f"Scan type of {bs_run.start['scan']['type']} is not currently supported."
        raise RuntimeError(err_str)

    # Applies to all xs_fluor, hence not adjusted when generating from roi
    if data_key == 'xs_fluor' and isinstance(data_slice, slice):
        data_slice = (slice(None), data_slice)
    
    # Data dimensionality
    # TODO: Is there a better way of handling this???
    if ('fluor' in data_key
        or 'image' in data_key):
        data_dims = 2
    else:
        data_dims = 0

    if data_slice is None:
        data = ds[data_key][:]
    else:
        try:
            # Python 3.11 implmentation
            data = ds[data_key][..., *data_slice] 
        except TypeError:
            # Previous versions
            data = ds[data_key][..., (data_slice)]
        except Exception as e:
            print('Error slicing data.')
            raise e

    if normalize:
        for sclr_key in ['i0', 'im', 'sclr_i0', 'sclr_im']:
            if sclr_key in ds:
                data /= ds[sclr_key][:]
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
        start_data_ndim = data.ndim
        data = np.asarray(data_processing_function(data))
        proc_data_ndim = data.ndim
        # Track if data dimensionality has been reduced
        data_dims -= (start_data_ndim - proc_data_ndim)

    axis = tuple(range(data_dims - data.ndim, 0, 1))
    data = integrating_function(data, axis=axis) # e.g., numpy.sum or numpy.max

    return data


def _get_rois(data,
              roi_mask,
              feature_type='points',
              point_method='multiple'):

    if feature_type == 'points':
        
        if point_method == 'multiple':
            labels = label(roi_mask.squeeze())
            rois = peak_local_max(data.squeeze(),
                                  labels=labels,
                                  min_distance=2) # Some padding
            rois_int = [data[tuple(roi)] for roi in rois]                      
        else:
            labels = label(roi_mask)
            if point_method == 'max':
                rois = peak_local_max(data.squeeze(),
                                      labels=label(roi_mask.squeeze()),
                                      min_distance=2, # Some padding
                                      num_peaks_per_label=1)
            elif point_method == 'center':
                coords = [r.coords for r in regionprops(labels)]
                rois = [np.mean([c.min(axis=0), c.max(axis=0)], axis=0) for c in coords]
            elif point_method == 'com':
                rois = center_of_mass(data, labels=labels, index=range(1, np.max(labels))) 

            rois_int = [np.sum(data[labels == num])
                        for num in np.unique(labels) if num != 0]      
        
        # Expand along zero dimensions
        if data.shape != data.squeeze().shape:
            flat_dims = np.array([d == 1 for d in data.shape])
            exp_rois = []
            for roi in rois:
                new_roi = np.zeros(data.ndim, dtype=int)
                new_roi[~flat_dims] = roi
                exp_rois.append(new_roi)
            rois = np.asarray(exp_rois)
        
    elif feature_type == 'regions':
        if roi_mask.ndim > 1:
            labels = label(roi_mask)
            rois = regionprops(labels, intensity_image=data)
            rois_int = [np.sum(data[labels == num])
                    for num in np.unique(labels) if num != 0]
        else:
            labels = label(roi_mask.reshape(-1, 1))
            rois = regionprops(labels, intensity_image=data)
            for roi in rois:
                # Hack to get back to 1D
                roi.slice = tuple([roi.slice[0]])
            rois_int = [np.sum(data.reshape(-1, 1)[labels == num])
                    for num in np.unique(labels) if num != 0]
    
    return rois, rois_int
    

def _get_motors_from_tiled(bs_run):
    
    # Get motors
    if bs_run.start['scan']['type'] != 'XRF_FLY':
        err_str = ("Only XRF_Fly scans are currently implemented, not "
                   + f"{bs_run.start['scan']['type']}")
        raise NotImplementedError(err_str)

    # Get motors
    motor_names = [getattr(nano_stage, cpt).name
                    for cpt in nano_stage.component_names]
    first_motor = getattr(nano_stage,
            nano_stage.component_names[
                motor_names.index(
                    bs_run.start['scan']['fast_axis']['motor_name'])])
    second_motor = getattr(nano_stage,
            nano_stage.component_names[
                motor_names.index(
                    bs_run.start['scan']['slow_axis']['motor_name'])])

    return [first_motor, second_motor]


def _generate_analysis_args(roi,
                            analysis_args,
                            search_motors,
                            search_axes,
                            analysis_motors,
                            analysis_axes,
                            position_values,
                            fix_edges,
                            feature_type,
                            point_method,
                            ):

    # Get ROIs in real values
    if feature_type == 'points':
        roi_values = [positions[roi_i] for roi_i, positions
                      in zip(roi[::-1], position_values)]
    elif feature_type == 'regions':
        roi_values = [positions[slice_i] for slice_i, positions
                      in zip(roi.slice[::-1], position_values)]
    roi_steps = [np.mean(np.diff(vals)) if len(vals) > 1 else 0 for vals in position_values]
    # print(roi_steps)

    # Setup useful values
    new_analysis_args = []
    move_positions = []
    VALID_ROI_RANGE = True
    FIXED_ROI_RANGE = False

    # Iterate though search motors to determine moves
    for s_ind in range(len(search_motors)):
        # Check if search motor is called for analysis
        if search_motors[s_ind] not in analysis_motors:
            # print('Search motor is not in analysis motors. Must move.')
            # Record locations to move motor
            if feature_type == 'points':
                val = roi_values[s_ind]
                move_positions.append(np.round(val, 3))
            elif feature_type == 'regions':
                # TODO: Add different methods of finding val (center, max, COM)
                
                if point_method == 'max':
                    val = roi_values[s_ind][np.argmax(np.max(roi.intensity_image, axis=s_ind))]
                elif point_method == 'com':
                    val = np.sum(np.sum(roi.intensity_image, axis=s_ind)
                                        * roi_values[s_ind]
                                 / np.sum(roi.intensity_image))
                elif point_method == 'center':
                    val = np.mean(roi_values[s_ind])
                
                move_positions.append(np.round(val, 3))

        else: # Search motor is used for analysis
            # Update arguments will be determined later
            move_positions.append(None)
    
    # Iterate through analysis motors
    if analysis_motors is not None:
        for a_ind in range(len(analysis_motors)):
            a_motor = analysis_motors[a_ind]
            # Ignore blank inputs
            if a_motor is None:
                continue

            # Parse agrument inputs
            start, end, num = analysis_args[3 * a_ind : 3 * (a_ind + 1)]

            # Check if axis is supposed to move?
            if num == 1:
                # print('Only one step, should be moving on!')
                new_analysis_args += [start, end, int(num)]
                continue
            
            # Was the analysis axis also used in search?
            if analysis_axes[a_ind] in search_axes:
                # Arguments must be altered.
                # Determine corresponding search axis
                # s_ind = np.nonzero(np.array(search_axes) == analysis_axes[a_ind])[0][0]
                s_ind = [s == analysis_axes[a_ind] for s in search_axes].index(True)

                # Get some values
                ext = end - start
                cen = start + (ext / 2) # Should always be zero???
                step = ext / (num - 1) # TODO: how to handle zero-division
                roi_start = np.min(roi_values[s_ind]) - (roi_steps[s_ind] / 2)
                roi_end = np.max(roi_values[s_ind]) + (roi_steps[s_ind] / 2)
                # print(f'ROI start at {roi_start}')
                # print(f'ROI end at {roi_end}')

                # Did they use the same motors?
                if a_motor in search_motors:
                    # print(f'Analysis motor {a_motor} [{a_ind}] in search motors!')
                    # Directly alter motor arguments
                    new_start = roi_start - (ext / 2)
                    new_end = roi_end + (ext / 2)
                    # print(f'ROI start at {roi_start}')
                    # print(f'ROI end at {roi_end}')

                else:
                    # print(f'Analysis motor {a_motor} [{a_ind}] not in search motors!')
                    # Expand regions around roi
                    roi_ext = roi_end - roi_start
                    # print(roi_ext)
                    new_ext = ext + roi_ext
                    new_start = cen - (new_ext / 2)
                    new_end = cen + (new_ext / 2)

                # Check if new scan arguments are within range
                safety_factor = 0.025 # Fraction of full scan range
                if a_motor.low_limit != a_motor.high_limit: # Equal limits means no limits
                    safe_range = safety_factor * (a_motor.high_limit - a_motor.low_limit)

                    if (new_end < a_motor.low_limit + safe_range
                        or new_start > a_motor.high_limit - safe_range):
                        # Cannot be fixed and should never happen
                        VALID_ROI_RANGE = False
                    elif new_start < a_motor.low_limit + safe_range:
                        if fix_edges:
                            new_start = a_motor.low_limit + safe_range
                            FIXED_ROI_RANGE = True
                        else:
                            VALID_ROI_RANGE = False
                    elif new_end > a_motor.high_limit - safe_range:
                        if fix_edges:
                            new_end = a_motor.high_limit - safe_range
                            FIXED_ROI_RANGE= True
                        else:
                            VALID_ROI_RANGE = False

                # Interpolate scan grid around center
                # This may contract scan area within new_start and new_end
                # print(new_start, new_end)
                new_ext = new_end - new_start
                # print(f'Unrounded center at {new_start + (new_ext / 2)}')
                new_cen = np.round(new_start + (new_ext / 2), 3)
                # print(f'Rounded center at {new_cen}')
                new_int = new_ext // step
                new_new_start = new_cen - (new_int / 2 * step)
                new_new_end = new_cen + (new_int / 2 * step)
                new_num = int(new_int) + 1
                # print(new_new_start, new_new_end, new_num)

                # Record new values
                new_analysis_args += [new_new_start, new_new_end, new_num]

            else:
                # Arguments can be used as is.
                new_analysis_args += [start, end, int(num)]

        # Finish filling up args
        other_args = analysis_args[len(new_analysis_args):] 
        new_analysis_args += other_args    
    
    else:
        # Verbatim analysis arguments
        new_analysis_args = analysis_args

    return (new_analysis_args,
            move_positions,
            VALID_ROI_RANGE,
            FIXED_ROI_RANGE)


def _plot_analysis_args(scan_id,
                        data,
                        rois,
                        analysis_args_list,
                        valid_rois,
                        fixed_rois,
                        position_values,
                        move_positions_list,
                        search_motors,
                        search_axes,
                        analysis_motors,
                        analysis_axes,
                        feature_type):

    # Setup useful values
    # Andy will love all this list comprehension
    pos_steps = [np.mean(np.diff(vals)) if len(vals) > 1 else 0 for vals in position_values]
    colors = ['red' if not valid else 'yellow' if fixed else 'lime' for valid, fixed in zip(valid_rois, fixed_rois)]
    fig, ax = plt.subplots()
    ax.set_title(f'scan{scan_id}: Found ROIs')

    # Convert to 1D if necessary
    data_ind = 0
    data_ndim = data.ndim
    if data.squeeze().ndim != data.ndim:
        flat_dims = [d == 1 for d in data.shape]
        flat_ind = flat_dims.index(True)
        data_ind = flat_dims.index(False)
        data = data.squeeze(axis=flat_ind)
        pos_ind = [step != 0 for step in pos_steps].index(True)
    plot_dims = data.ndim

    # 2D search
    if plot_dims == 2:
        # Inverts y-axis
        extent = [np.min(position_values[0]) - (pos_steps[0] / 2),
                  np.max(position_values[0]) + (pos_steps[0] / 2),
                  np.max(position_values[1]) + (pos_steps[1] / 2),
                  np.min(position_values[1]) - (pos_steps[1] / 2)]
        im = ax.imshow(data, extent=extent)
        fig.colorbar(im, ax=ax)
        ax.set_aspect('equal')
        ax.set_xlabel(f'{search_motors[0].name} [{search_motors[0].motor_egu.get()}]')
        ax.set_ylabel(f'{search_motors[1].name} [{search_motors[1].motor_egu.get()}]')

        # Add found ROIs
        if feature_type == 'points':
            xplot = position_values[rois[:, 1]]
            yplot = position_values[rois[:, 0]]
            
            ax.scatter(xplot,
                       yplot,
                       c=colors,
                       marker='+',
                       s=100,
                       label='ROIs')
        else:
            rect_list = []
            for ind, roi in enumerate(rois):
                xplot = position_values[0][roi.slice[1]]
                yplot = position_values[1][roi.slice[0]]

                rect = patches.Rectangle(
                            (np.min(xplot) - (pos_steps[0] / 2),
                             np.min(yplot) - (pos_steps[1] / 2)),
                             np.max(xplot) - np.min(xplot) + pos_steps[0],
                             np.max(yplot) - np.min(yplot) + pos_steps[1],
                            linewidth=1.5,
                            linestyle='--',
                            edgecolor=colors[ind],
                            facecolor='none')
                rect_list.append(rect)
            pc = PatchCollection(rect_list,
                                match_original=True,
                                label='ROIs')
            ax.add_collection(pc)
        
        # Add new mapped ROIs if matching axes
        if all([s_axis in analysis_axes for s_axis in search_axes]):
            # Check if tranposed
            if analysis_axes[0] == search_axes[0]:
                order = slice(None, None, 1)
            else:
                order = slice(None, None, -1)
            
            rect_list = []
            for ind, args in enumerate(analysis_args_list):
                sorted_args = [args[:3], args[3:6]][order]
                xstart, xend, xnum = sorted_args[0]
                xstep = (xend - xstart) / (xnum - 1)
                ystart, yend, ynum = sorted_args[1]
                ystep = (yend - ystart) / (ynum - 1)

                if move_positions_list[ind][0] is not None:
                    xmove = move_positions_list[ind][0]
                else:
                    xmove = 0
                if move_positions_list[ind][1] is not None:
                    ymove = move_positions_list[ind][1]
                else:
                    ymove = 0
                
                rect = patches.Rectangle(
                            (xstart - (xstep / 2) + xmove,
                             ystart - (ystep / 2) + ymove),
                            xend - xstart + xstep,
                            yend - ystart + ystep,
                            linewidth=2,
                            linestyle='-',
                            edgecolor=colors[ind],
                            facecolor='none')
                rect_list.append(rect)
            pc = PatchCollection(rect_list,
                                match_original=True,
                                label='Analysis')
            ax.add_collection(pc)
        
        # Regions, but without matching analysis axes (i.e., point analysis)
        elif feature_type == 'regions':
            xpos = np.asarray(move_positions_list)[:, 0]
            ypos = np.asarray(move_positions_list)[:, 1]
            ax.scatter(xpos,
                       ypos,
                       c=colors,
                       marker='+',
                       s=100,
                       label='Analysis')
    
    # 1D search
    if plot_dims == 1:
        ax.plot(position_values[pos_ind],
                data,
                '.-',
                c='k')
        ax.set_xlabel(f'{search_motors[pos_ind].name} [{search_motors[pos_ind].motor_egu.get()}]')

        if feature_type == 'points':
            xplot = position_values[pos_ind][rois[:, data_ind]]
            
            ax.scatter(xplot,
                       data.squeeze()[rois[:, data_ind]],
                       c=colors,
                       marker='+',
                       s=100,
                       label='ROIs')

        else:
            for ind, roi in enumerate(rois):
                xplot = position_values[pos_ind][roi.slice[data_ind]]
                ax.axvline(xplot[0], linestyle='--', c=colors[ind])
                ax.axvline(xplot[-1], linestyle='--', c=colors[ind])

        # Add new mapped ROIs if matching axes
        if all([s_axis in analysis_axes for s_axis in search_axes]):

            for ind, args in enumerate(analysis_args_list):
                # xstart, xend, xnum = args[:3]
                xstart, xend, xnum = args[3 * pos_ind : 3 * (pos_ind + 1)]

                ax.axvline(xstart, c=colors[ind])
                ax.axvline(xend, c=colors[ind])
        
        # Regions, but without matching analysis axes (i.e., point analysis)
        elif feature_type == 'regions':
            xpos = np.asarray(move_positions_list)[:, pos_ind]
            ax.scatter(xpos,
                       data.squeeze()[xpos],
                       c=colors,
                       marker='+',
                       s=100,
                       label='Analysis')
    
    # Finally finished!
    # fig.show()


# Convenience Wrappers

def coarse_xrf_search_and_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=coarse_scan_and_fly,
                search_motors=[nano_stage.topx, nano_stage.y],
                data_key='xs_fluor',
                analysis_motors=[nano_stage.topx, nano_stage.y],
                analysis_function=coarse_scan_and_fly
                )

def coarse_xrf_search_and_nano_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=coarse_scan_and_fly,
                search_motors=[nano_stage.topx, nano_stage.y],
                data_key='xs_fluor',
                analysis_motors=[nano_stage.sx, nano_stage.sy],
                analysis_function=nano_scan_and_fly
                )

def coarse_xrf_search_and_xanes_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=coarse_scan_and_fly,
                search_motors=[nano_stage.topx, nano_stage.y],
                data_key='xs_fluor',
                analysis_motors=None,
                analysis_function=xanes_plan
                )

def nano_xrf_search_and_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=nano_scan_and_fly,
                search_motors=[nano_stage.sx, nano_stage.sy],
                data_key='xs_fluor',
                analysis_motors=[nano_stage.sx, nano_stage.sy],
                analysis_function=nano_scan_and_fly
                )

def nano_xrf_search_and_xanes_analyze(**kwargs):
    yield from search_and_analyze_base(
                **kwargs,
                search_function=nano_scan_and_fly,
                search_motors=[nano_stage.sx, nano_stage.sy],
                data_key='xs_fluor',
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
#                                        search_motors=[nano_stage.topx, nano_stage.y],
#                                        data_cutoff=200,
#                                        data_key='xs_fluor',
#                                        search_defocus_distance=1200,
#                                        feature_type='regions',
#                                        move_for_analysis=True,
#                                        analysis_args=[-2.5, 2.5, 21, -2.5, 2.5, 21, 0.1],
#                                        analysis_function=sub_search,
#                                        analysis_motors=[nano_stage.sx, nano_stage.sy],
#                                        wait_time=1)