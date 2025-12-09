import numpy as np
import scipy as sp
import scipy.interpolate
import datetime as dt
import pandas as pd
from copy import deepcopy
import time
from MCEM_files import IceWallMeltFunctionGhost, Cfunctions, ObjectBoundary



def MCEM(b0, max_water_d, beta, water_t, ice_albedo, latitude, longitude, channel_orientation, start_date,
         runtime_hours, data_filename, save_hrs=200, ice_col_spacing=0.01, rk_loop_tstep=20,
         boundary_spacing=0.01, gfilter_ls=0.003, velocity_shift=0, diurnal=False):
    '''
    b0 - inital boundary points. Array shape (n, 2). n number of points, 2 x,y coords.
    Q - water flux m^3/s
    V - average water velocity m/s
    beta - channel slope, dimensionless
    water_t - water temperature, Kelvin
    ice_albedo
    lat and long, input in RADIANS
    channel_orientation - RADIANS
    start_date - start date as a datetime object. Must be on the hour for PROMICE data.
    runtime_hours
    save_hrs - integer, hourly save rate of data
    ice_col_spacing - space between modelled temperature points, m
    rk_loop_tstep - number of seconds taken at each step to evolve heat equation in rk loop, max 3600 for an hour
    data_filename - path to data
    gfilter_sigma - gaussian filter sigma value. Found by Doing length scale/avg distance between points. To keep
    consistent avg distance between points is boundary spacing. Current length scale is 0.003
    '''

    # start time of function run for metrics
    timing_start = time.process_time()

    '''
    Variable storage and meta data creation
    '''
    # define dictionary to store the boundary points each of which are object with attributes
    dict_sol = dict()
    # define dictionary containing the meta data
    meta_data = {
        'Number of boundary points': len(b0),
        # 'Water flux m^3/s': Q,
        # 'Water Velocity m/s': V,
        'Input File': data_filename,
        'Channel Orientation radians': channel_orientation,
        'Channel Slope': beta,
        'Start Date': start_date,
        'End Date': start_date + dt.timedelta(hours=runtime_hours),
        'Save Hours multiple': save_hrs,
        'RK loop timestep s': rk_loop_tstep,
        'Ice column temp spacing m': ice_col_spacing,
        'Ice albedo': ice_albedo
    }

    '''
    Creating a new starting array so that each point in an object with an associated ice temperature 
    profile
    '''
    # define the number of points in boundary array
    lenb = len(b0)
    # define blank list
    b = []
    # fill list with point classes with initial positions defined in b0
    # 3m standard column depth
    # column initial temp just for set up will be changed
    for i in range(len(b0)):
        b.append(ObjectBoundary.ObjectPoint(b0[i], albedo=ice_albedo, columnInitialTemp=271.25, columnDepth=3,
                             columnPointSpacing=ice_col_spacing))

    '''
    Define start time 
    '''
    # define start date, datetime variable
    t = start_date

    '''
    Load in data from the file
    data should be clean before hand with no NaN
    '''
    # import data as a pandas df
    data = pd.read_csv(data_filename)
    data['time'] = pd.to_datetime(data['time'], format='mixed')

    '''
    get initial ice temperatures
    Interpolate temperature profiles at depth to get a new initial column temp profile
    '''

    # get temperature at surface to m below surface
    t_surface = data[(data['time'] == t)].iloc[0]['t_surf'] + 270.15
    ice_t_1 = data[(data['time'] == t)].iloc[0]['t_i_1'] + 273.15
    ice_t_2 = data[(data['time'] == t)].iloc[0]['t_i_2'] + 273.15
    ice_t_3 = data[(data['time'] == t)].iloc[0]['t_i_3'] + 273.15
    # get equally spaced x points
    new_x = np.linspace(3, 0, b[0].noTempPoints)
    # get y values
    yvals = [ice_t_3, ice_t_2, ice_t_1, t_surface]

    # to change for warmer or colder start
    # for i in range(len(yvals)):
    #     yvals[i] = yvals[i] + Kelvin increment
    xvals = [3, 2, 1, 0]

    # interpolate to get new y vals
    new_y = sp.interpolate.interp1d(xvals, yvals, kind='linear')(new_x)
    new_y = np.flip(new_y)

    # set point object with new temp profile
    for i in range(len(b)):
        b[i].columnTempProfile = np.copy(new_y)

    '''
    define flat point to check melt rate against
    '''
    # define flat point to check overall melt
    flat_point = ObjectBoundary.ObjectPoint(b0[0], albedo=ice_albedo, columnInitialTemp=271.25, columnDepth=3,
                             columnPointSpacing=ice_col_spacing)
    # set flat melt to same temperature column
    flat_point.columnTempProfile = np.copy(new_y)
    # define counter for amount of melt flat point experiences
    flat_melt = 0

    '''
    To ensure that the distance between points is less than or equal to 1cm
    '''
    b = Cfunctions.ensure_max_distance(b, threshold=boundary_spacing)

    '''
    Flipping for symmetry
    '''
    # define to say if the boundary has been flipped
    # avoids geometrical or numerical asymmetry
    flipped = False
    if np.pi <= channel_orientation < 2 * np.pi:
        channel_orientation = channel_orientation - np.pi
        flipped = True

    '''
    Define wet points and hydrological parameters at t = 0
    '''
    # fill with water if there is some
    if max_water_d > 0:
        # use list comprehension to get position of all boundary points
        b_pos = [x.position for x in b]
        # define y values
        b_pos_y = [x.position[1] for x in b]
        # get relative water depth
        rel_water_d = max_water_d + min(b_pos_y)
        # dirunal variability
        if diurnal == True:
            rel_water_d = Cfunctions.water_level(max_water_d, t.hour, 0.2) + min(b_pos_y)


        # loop around all point objects and make the points wet
        for j in range(len(b)):
            if (b[j].position[1] <= rel_water_d):
                b[j].isWet = True

        # get list of wp points
        wp = [x for x in b if x.isWet == True]
        # get position of wp
        wp_pos = [x.position for x in wp]
        # get area of wp
        A = Cfunctions.bucket_area(wp_pos)
        # get length of wp
        P = Cfunctions.perimeter(wp_pos)
        # use Chezy formula to get water velocity, using coeff (39.363) from field data
        V = 39.363 * (A / P * beta) ** 0.5
        print('V', V)
        # get water flux
        Q = V * A
        print('Q', Q)

        # get area of the channel (assuming trapezoid sections can be used to find area)
        b_area = Cfunctions.polyarea(b_pos)
        A = Cfunctions.bucket_area(wp_pos)

        # add error check if stream is overflowing
        if A > b_area:
            print('Water cross-sectional area greater than channel area')
            # create a dictionary of the two dictionaries
            output_dict = {'meta data': meta_data,
                           'data': dict_sol}
            # define end time
            timing_end = time.process_time()
            time2run = '{} s'.format(timing_end - timing_start)

            output_dict['meta data']['runtime'] = time2run

            return output_dict


    '''
    Decide which points are shaded
    '''
    # first need the position of the boundary points
    b_pos = np.array([x.position for x in b])
    gps_lat = data[(data['time'] == t)].iloc[0]['gps_lat']
    gps_lon = data[(data['time'] == t)].iloc[0]['gps_lon']
    Cfunctions.is_shaded(b, time=t, latitude=np.deg2rad(gps_lat), longitude=np.deg2rad(gps_lon),
                         channel_orientation=
                         channel_orientation)

    '''
    Saving solution at t = 0
    '''
    if flipped == True:
        current_pos = [x.position for x in b]
        original_centre_x = (b0[0, 0] + b0[-1, 0]) / 2

        # reverse dictionary so all other attributes match
        flipped_b = deepcopy(b)
        flipped_b.reverse()
        current_pos = [x.position for x in flipped_b]

        # flip across the centre
        for i in range(len(current_pos)):
            flipped_b[i].position = np.array((2 * original_centre_x - current_pos[i][0], current_pos[i][1]))
        # only save the solution every save_hrs hours

        dict_sol[0] = deepcopy(flipped_b)
    else:
        dict_sol[0] = deepcopy(b)

    '''
    MAIN LOOP
    Open loop which runs the processes over the hours stated
    '''
    # start from 1 as already saved a copy of original boundary at 0, +1 to account for python counting
    for h in range(1, runtime_hours + 1):
        print('hour', h)
        print('number of b points', len(b))

        '''
        Creating variables from the imported data
        The headings are based off PROMICE data
        '''
        air_pressure = data[(data['time'] == t)].iloc[0]['p_u']
        # convert to kelvin
        t_air = data[(data['time'] == t)].iloc[0]['t_u'] + 273.15
        t_surface = data[(data['time'] == t)].iloc[0]['t_surf'] + 273.15
        rel_hum = data[(data['time'] == t)].iloc[0]['rh_u']
        wind_v = data[(data['time'] == t)].iloc[0]['wspd_u']
        sw_in = data[(data['time'] == t)].iloc[0]['dsr_cor']
        sw_out = data[(data['time'] == t)].iloc[0]['usr_cor']
        lw_in = data[(data['time'] == t)].iloc[0]['dlr']
        lw_out = data[(data['time'] == t)].iloc[0]['ulr']
        # this is bulk latent heat
        latent = data[(data['time'] == t)].iloc[0]['dlhf_u']
        sensible = data[(data['time'] == t)].iloc[0]['dshf_u']
        gps_lat = data[(data['time'] == t)].iloc[0]['gps_lat']
        gps_lon = data[(data['time'] == t)].iloc[0]['gps_lon']

        # change sw_in so that it is actual shortwave, not that observed on a flat surface
        # not sure if this is needed
        sw_in = Cfunctions.flatSurface2actual_sw(sw_in, t, np.deg2rad(gps_lat), np.deg2rad(gps_lon))

        '''
        Unshading all points from previous time
        '''
        for i in range(len(b)):
            # b[i].position = np.array([new_x[i], new_y[i]])
            # shade all points
            b[i].isShaded = False

        '''
        Decide which points are shaded
        '''
        # first need the position of the boundary points
        b_pos = np.array([x.position for x in b])

        Cfunctions.is_shaded(b, time=t, latitude=np.deg2rad(gps_lat), longitude=np.deg2rad(gps_lon),
                             channel_orientation=
                             channel_orientation)

        '''
        Get water level and flux at a given time
        '''

        b_pos_y = [x.position[1] for x in b]
        # get relative water depth
        rel_water_d = max_water_d + min(b_pos_y)
        # diurnal switch
        if diurnal == True:
            rel_water_d = Cfunctions.water_level(max_water_d, t.hour, 0.2) + min(b_pos_y)
        # loop around all point objects and make the points wet, make others dry
        for j in range(len(b)):
            if (b[j].position[1] <= rel_water_d):
                b[j].isWet = True
            else:
                b[j].isWet = False

        wet_idx = [i for i in range(len(b)) if b[i].isWet == True]
        # to remove outliers (points in between which may not have been wetted)
        min_val = min(wet_idx)
        max_val = max(wet_idx)
        wet_idx = list(range(min_val, max_val + 1))
        for i in wet_idx:
            b[i].isWet = True

        # get list of wp points
        wp = [x for x in b if x.isWet == True]
        # get position of wp
        wp_pos = [x.position for x in wp]
        # get area of wp
        A = Cfunctions.polyarea(wp_pos)
        # get length of wp
        P = Cfunctions.perimeter(wp_pos)
        # use Chezy formula to get water velocity, using coeff from field data. This is mean velocity
        V = 39.363 * (A / P * beta) ** 0.5
        # print('V', V)
        # get water flux
        Q = V * A
        # print('Q', Q)

        # get area of the channel (assuming trapezoid sections can be used to find area)
        b_area = Cfunctions.polyarea(b_pos)

        # add error check if stream is overflowing
        if A > b_area:
            print('Water cross-sectional area greater than channel area')
            dict_sol[h] = deepcopy(b)
            # create a dictionary of the two dictionaries
            output_dict = {'meta data': meta_data,
                           'data': dict_sol}
            # define end time
            timing_end = time.process_time()
            time2run = '{} s'.format(timing_end - timing_start)

            output_dict['meta data']['runtime'] = time2run

            return output_dict

        '''
        Creating two separate parts, for if there is water in channel or not
        '''
        # if there is water
        if Q > 0:

            '''
            Gradient Calculations
            Since when moving points the gradient will change depending on point position all gradients should be worked
            out before they update. Can work them all out now, or update all positions at the end
            '''
            # work out interior point gradients
            # define blank lists to fill with gradients
            tang_gradients = []
            norm_gradients = []
            surface_tilts = []
            # find gradient of lhs point first
            # add gradient to lists

            i = 0
            tang_gradients.append(b[i].tangent_gradient(b[i].position, b[i + 2].position))
            norm_gradients.append(b[i].norm_gradient(b[i].position, b[i + 2].position))
            surface_tilts.append(b[i].surface_tilt(b[i].position, b[i + 2].position))

            for i in range(1, len(b) - 1):
                tang_gradients.append(b[i].tangent_gradient(b[i - 1].position, b[i + 1].position))
                n_grad = (b[i].norm_gradient(b[i - 1].position, b[i + 1].position))
                norm_gradients.append(n_grad)
                surface_tilts.append(b[i].surface_tilt(b[i - 1].position, b[i + 1].position))
            # now rhs boundary
            i = -1
            tang_gradients.append(b[i].tangent_gradient(b[i - 2].position, b[i].position))
            norm_gradients.append(b[i].norm_gradient(b[i - 2].position, b[i].position))
            surface_tilts.append(b[i].surface_tilt(b[i - 2].position, b[i].position))

            # get the surface azimuth of each point
            # get position of all points
            b_pos = np.asarray([x.position for x in b])
            surface_azimuths = Cfunctions.surface_azimuths(b_pos, channel_direction=channel_orientation)

            '''
            #fix to make sure norm gradients are roughly correct. 
            if to the left of centre point which will be deepest point norm gradients should be positive
            if to the right they should be negative
            '''
            # # find index of deepest point
            centre_point_idx = np.argmin(b_pos[:, 1])

            for i in range(len(b)):
                if i < centre_point_idx:
                    if norm_gradients[i] != 'not_defined':
                        norm_gradients[i] = float(norm_gradients[i])
                        if norm_gradients[i] < 0 and isinstance(norm_gradients[i], float):
                            norm_gradients[i] = -norm_gradients[i]
                if i > centre_point_idx:
                    if norm_gradients[i] != 'not_defined':
                        norm_gradients[i] = float(norm_gradients[i])
                        if norm_gradients[i] > 0 and isinstance(norm_gradients[i], float):
                            norm_gradients[i] = -norm_gradients[i]

            original_b_pos = np.asarray([x.position for x in b])

            '''
            Wetted Perimeter set up and defining shear
            '''
            # get WP index
            wet_idx = [idx for idx, x in enumerate(b) if x.isWet == True]
            # get WP points
            wp = [x for x in b if x.isWet == True]
            # extract the position and x,y coords
            wp_pos = np.asarray([x.position for x in wp])

            # get wall shear stress list
            wall_shear = Cfunctions.WTA_bed_stresses(wp_boundary=wp_pos, mean_v=V, slope=beta,
                                                     centre_shift=velocity_shift)

            # get day of the year
            day_no = t.timetuple().tm_yday
            # get local solar time
            lst = Cfunctions.local_solar_time(long=np.deg2rad(gps_lon), dn=day_no, lt=t.hour)

            '''
            Melt and move WP
            '''
            # define a shear index counter
            shear_idx = 0

            # list of local melt for each point
            mloc_list = []
            # loop round wp points
            for i in wet_idx:
                if b[i].isShaded == True:
                    sw = 0
                else:
                    sw = sw_in
                # get the melt rate and update temperature of ice column
                mmrnew = IceWallMeltFunctionGhost.IceWallMeltWP(point=b[i], surface_tilt=surface_tilts[i],
                                                                surface_azimuth=surface_azimuths[i],
                                                                sw_in=sw_in, lw_in=lw_in, lw_out=lw_out, lst=lst,
                                                                sens=sensible,
                                                                latent=latent, day_no=day_no,
                                                                latitude=np.deg2rad(gps_lat),
                                                                longitude=np.deg2rad(gps_lon),
                                                                air_temp=t_air,
                                                                beta=beta, water_t=water_t,
                                                                water_v=V, rel_water_d=rel_water_d,
                                                                water_flux=Q, P=Cfunctions.perimeter(wp_pos),
                                                                wall_force=
                                                                wall_shear[shear_idx], delta_t=rk_loop_tstep)
                # increase shear idx
                shear_idx += 1
                mloc = mmrnew
                mloc_list.append(mloc)
                # move the point
                Cfunctions.pointsPositionUpdater(point=b[i], norm_gradient=norm_gradients[i], mloc=mloc)


            '''
            Dry Perimeter moving and temp profile update
            '''
            # get DP index
            dry_idx = [idx for idx, x in enumerate(b) if x.isWet == False]
            # loop around all dry points
            for i in dry_idx:
                if b[i].isShaded == True:
                    sw = 0
                else:
                    sw = sw_in
                melt_amount = IceWallMeltFunctionGhost.IceWallMeltDPGhost(point=b[i], surface_tilt=surface_tilts[i],
                                                                          surface_azimuth=surface_azimuths[i],
                                                                          sw_in=sw, lw_in=lw_in, lw_out=lw_out,
                                                                          lst=lst, sens=sensible,
                                                                          latent=latent, day_no=day_no,
                                                                          latitude=np.deg2rad(gps_lat),
                                                                          longitude=np.deg2rad(gps_lon),
                                                                          delta_t=rk_loop_tstep)

                if melt_amount > 0:  # and b[i].isUndercut == False:
                    # move dp point
                    Cfunctions.pointsPositionUpdater(point=b[i], norm_gradient=norm_gradients[i], mloc=melt_amount)

                    # to stop melting below water line
                    # get WP points
                    wp = [x for x in b if x.isWet == True]
                    # extract the position and x,y coords
                    wp_pos = np.asarray([x.position for x in wp])
                    if b[i].position[1] < max(wp_pos[:, 1]):
                        b[i].position[1] = max(wp_pos[:, 1])

            # doing single flat point
            # find melt
            # orientation irrelevant (surface azimtuh) since surface tilt = 0
            flat_mr = IceWallMeltFunctionGhost.IceWallMeltDPGhost(point=flat_point, surface_tilt=0,
                                                                  surface_azimuth=channel_orientation,
                                                                  sw_in=sw, lw_in=lw_in, lw_out=lw_out,
                                                                  lst=lst, sens=sensible,
                                                                  latent=latent, day_no=day_no,
                                                                  latitude=np.deg2rad(gps_lat),
                                                                  longitude=np.deg2rad(gps_lon),
                                                                  delta_t=rk_loop_tstep)
            # move point and track melt
            flat_melt += flat_mr
            Cfunctions.pointsPositionUpdater(point=flat_point, norm_gradient='not_defined', mloc=flat_mr)
            # print('total flat melt', flat_melt)
            # print('flat point y value', flat_point.position[1])
            # if flat point is lower than any of the points remove them
            # create a list of values to be removes
            remove_list = []
            # append indicies to this list
            for i in range(len(b)):
                if b[i].position[1] > flat_point.position[1]:
                    # print('point removal trigger')
                    remove_list.append(i)

            # flip list so that index doesn't dynamically change
            reverse_list = sorted(remove_list, reverse=True)
            for i in reverse_list:
                if i < len(b):
                    b.pop(i)

            # if enough points removed then have to return the solution, that is if the channel is less than 4 points
            if len(b) < 4:
                if flipped == True:
                    current_pos = [x.position for x in b]
                    original_centre_x = (b0[0, 0] + b0[-1, 0]) / 2

                    # reverse dictionary so all other attributes match
                    flipped_b = deepcopy(b)
                    flipped_b.reverse()
                    current_pos = [x.position for x in flipped_b]

                    # flip across the centre
                    for i in range(len(current_pos)):
                        flipped_b[i].position = np.array((2 * original_centre_x - current_pos[i][0], current_pos[i][1]))
                    # save solution at end time
                    dict_sol[h] = deepcopy(flipped_b)
                else:
                    dict_sol[h] = deepcopy(b)

                # create a dictionary of the two dictionaries
                output_dict = {'meta data': meta_data,
                               'data': dict_sol}
                # define end time
                timing_end = time.process_time()
                time2run = '{} s'.format(timing_end - timing_start)

                output_dict['meta data']['runtime'] = time2run

                return output_dict

            '''
            Gaussian filtering
            '''
            # get position of points
            b_pos = np.asarray([x.position for x in b])
            # option is chosen so that the threshold, which is constant, is in control of gfilter. Keeping
            # constant rate of filtering/smoothing throughout the runs.
            sigma = gfilter_ls / boundary_spacing
            b_new = Cfunctions.smooth_boundary(b_pos, sigma)
            for i in range(len(b)):
                b[i].position = b_new[i]

            '''
            Redistribution of points by interpolation
            '''

            # use list comprehension to get position of all boundary points
            b_pos = np.asarray([x.position for x in b])

            new_line = Cfunctions.interpcurve(len(b), b_pos[:, 0], b_pos[:, 1])
            for i in range(len(b)):
                b[i].position = new_line[i]

            '''
            Ensuring resolution is 1cm or better
            '''
            # ensuring points are never less than 1cm apart
            b = Cfunctions.ensure_max_distance(b, threshold=boundary_spacing)

            b_pos = np.asarray([x.position for x in b])

            '''
            Redoing the wet and dry points so that the moved boundary has correct wetted/dry points
            '''
            for i in range(len(b)):
                # dry all points
                b[i].isWet = False

            # define y values
            b_pos_y = [x.position[1] for x in b]
            # get relative water depth
            rel_water_d = max_water_d + min(b_pos_y)

            if diurnal == True:
                rel_water_d = Cfunctions.water_level(max_water_d, t.hour, 0.2) + min(b_pos_y)

            # loop around all point objects and make the points wet
            for j in range(len(b)):
                if (b[j].position[1] <= rel_water_d):
                    b[j].isWet = True

            '''
            add boundary to dictionary
            '''

            if flipped == True:
                current_pos = [x.position for x in b]
                original_centre_x = (b0[0, 0] + b0[-1, 0]) / 2

                # reverse dictionary so all other attributes match
                flipped_b = deepcopy(b)
                flipped_b.reverse()
                current_pos = [x.position for x in flipped_b]

                # flip across the centre
                for i in range(len(current_pos)):
                    flipped_b[i].position = np.array((2 * original_centre_x - current_pos[i][0], current_pos[i][1]))
                # only save the solution every save_hrs hours
                if h % save_hrs == 0:
                    dict_sol[h] = deepcopy(flipped_b)
            else:
                if h % save_hrs == 0:
                    dict_sol[h] = deepcopy(b)

            '''
            advance time forwards by an hour
            '''
            t = t + dt.timedelta(hours=1)

        # if there isn't water
        else:

            for j in range(len(b)):
                b[j].isWet = False


            '''
            Gradient Calculations
            Since when moving points the gradient will change depending on point position all gradients should be worked
            out before they update. Can work them all out now, or update all positions at the end
            '''
            # work out interior point gradients
            # define blank lists to fill with gradients
            tang_gradients = []
            norm_gradients = []
            surface_tilts = []
            # find gradient of lhs point first
            # add gradient to lists

            i = 0
            tang_gradients.append(b[i].tangent_gradient(b[i].position, b[i + 2].position))
            norm_gradients.append(b[i].norm_gradient(b[i].position, b[i + 2].position))
            surface_tilts.append(b[i].surface_tilt(b[i].position, b[i + 2].position))

            for i in range(1, len(b) - 1):
                tang_gradients.append(b[i].tangent_gradient(b[i - 1].position, b[i + 1].position))
                n_grad = (b[i].norm_gradient(b[i - 1].position, b[i + 1].position))
                norm_gradients.append(n_grad)
                surface_tilts.append(b[i].surface_tilt(b[i - 1].position, b[i + 1].position))
            # now rhs boundary
            i = -1
            tang_gradients.append(b[i].tangent_gradient(b[i - 2].position, b[i].position))
            norm_gradients.append(b[i].norm_gradient(b[i - 2].position, b[i].position))
            surface_tilts.append(b[i].surface_tilt(b[i - 2].position, b[i].position))

            # get the surface azimuth of each point
            # get position of all points
            b_pos = np.asarray([x.position for x in b])
            surface_azimuths = Cfunctions.surface_azimuths(b_pos, channel_direction=channel_orientation)

            '''
            #fix to make sure norm gradients are roughly correct. 
            if to the left of centre point which will be deepest point norm gradients should be positive
            if to the right they should be negative
            '''
            # # find index of deepest point
            centre_point_idx = np.argmin(b_pos[:, 1])

            for i in range(len(b)):
                if i < centre_point_idx:
                    if norm_gradients[i] != 'not_defined':
                        norm_gradients[i] = float(norm_gradients[i])
                        if norm_gradients[i] < 0 and isinstance(norm_gradients[i], float):
                            norm_gradients[i] = -norm_gradients[i]
                if i > centre_point_idx:
                    if norm_gradients[i] != 'not_defined':
                        norm_gradients[i] = float(norm_gradients[i])
                        if norm_gradients[i] > 0 and isinstance(norm_gradients[i], float):
                            norm_gradients[i] = -norm_gradients[i]

            original_b_pos = np.asarray([x.position for x in b])




            # get day of the year
            day_no = t.timetuple().tm_yday
            # get local solar time
            lst = Cfunctions.local_solar_time(long=np.deg2rad(gps_lon), dn=day_no, lt=t.hour)



            '''
            Dry Perimeter moving and temp profile update
            '''
            # get DP index
            dry_idx = [idx for idx, x in enumerate(b) if x.isWet == False]
            # loop around all dry points
            for i in dry_idx:
                if b[i].isShaded == True:
                    sw = 0
                else:
                    sw = sw_in
                melt_amount = IceWallMeltFunctionGhost.IceWallMeltDPGhost(point=b[i], surface_tilt=surface_tilts[i],
                                                                          surface_azimuth=surface_azimuths[i],
                                                                          sw_in=sw, lw_in=lw_in, lw_out=lw_out,
                                                                          lst=lst, sens=sensible,
                                                                          latent=latent, day_no=day_no,
                                                                          latitude=np.deg2rad(gps_lat),
                                                                          longitude=np.deg2rad(gps_lon),
                                                                          delta_t=rk_loop_tstep)

                if melt_amount > 0:  # and b[i].isUndercut == False:
                    # move dp point
                    Cfunctions.pointsPositionUpdater(point=b[i], norm_gradient=norm_gradients[i], mloc=melt_amount)

                    # to stop melting below water line
                    # get WP points
                    wp = [x for x in b if x.isWet == True]
                    # extract the position and x,y coords
                    wp_pos = np.asarray([x.position for x in wp])
                    if b[i].position[1] < max(wp_pos[:, 1]):
                        b[i].position[1] = max(wp_pos[:, 1])

            # doing single flat point
            # find melt
            # orientation irrelevant (surface azimtuh) since surface tilt = 0
            flat_mr = IceWallMeltFunctionGhost.IceWallMeltDPGhost(point=flat_point, surface_tilt=0,
                                                                  surface_azimuth=channel_orientation,
                                                                  sw_in=sw, lw_in=lw_in, lw_out=lw_out,
                                                                  lst=lst, sens=sensible,
                                                                  latent=latent, day_no=day_no,
                                                                  latitude=np.deg2rad(gps_lat),
                                                                  longitude=np.deg2rad(gps_lon),
                                                                  delta_t=rk_loop_tstep)
            # move point and track melt
            flat_melt += flat_mr
            Cfunctions.pointsPositionUpdater(point=flat_point, norm_gradient='not_defined', mloc=flat_mr)
            # print('total flat melt', flat_melt)
            # print('flat point y value', flat_point.position[1])
            # if flat point is lower than any of the points remove them
            # create a list of values to be removes
            remove_list = []
            # append indicies to this list
            for i in range(len(b)):
                if b[i].position[1] > flat_point.position[1]:
                    # print('point removal trigger')
                    remove_list.append(i)

            # flip list so that index doesn't dynamically change
            reverse_list = sorted(remove_list, reverse=True)
            for i in reverse_list:
                if i < len(b):
                    b.pop(i)

            # if enough points removed then have to return the solution, that is if the channel is less than 4 points
            if len(b) < 4:
                if flipped == True:
                    current_pos = [x.position for x in b]
                    original_centre_x = (b0[0, 0] + b0[-1, 0]) / 2

                    # reverse dictionary so all other attributes match
                    flipped_b = deepcopy(b)
                    flipped_b.reverse()
                    current_pos = [x.position for x in flipped_b]

                    # flip across the centre
                    for i in range(len(current_pos)):
                        flipped_b[i].position = np.array((2 * original_centre_x - current_pos[i][0], current_pos[i][1]))
                    # save solution at end time
                    dict_sol[h] = deepcopy(flipped_b)
                else:
                    dict_sol[h] = deepcopy(b)

                # create a dictionary of the two dictionaries
                output_dict = {'meta data': meta_data,
                               'data': dict_sol}
                # define end time
                timing_end = time.process_time()
                time2run = '{} s'.format(timing_end - timing_start)

                output_dict['meta data']['runtime'] = time2run

                return output_dict

            '''
            Gaussian filtering
            '''
            # get position of points
            b_pos = np.asarray([x.position for x in b])
            # option is chosen so that the threshold, which is constant, is in control of gfilter. Keeping
            # constant rate of filtering/smoothing throughout the runs.
            sigma = gfilter_ls / boundary_spacing
            b_new = Cfunctions.smooth_boundary(b_pos, sigma)
            for i in range(len(b)):
                b[i].position = b_new[i]

            '''
            Redistribution of points by interpolation
            '''

            # use list comprehension to get position of all boundary points
            b_pos = np.asarray([x.position for x in b])

            new_line = Cfunctions.interpcurve(len(b), b_pos[:, 0], b_pos[:, 1])
            for i in range(len(b)):
                b[i].position = new_line[i]

            '''
            Ensuring resolution is 1cm or better
            '''
            # ensuring points are never less than 1cm apart
            b = Cfunctions.ensure_max_distance(b, threshold=boundary_spacing)

            b_pos = np.asarray([x.position for x in b])

            '''
            Redoing the wet and dry points so that the moved boundary has correct wetted/dry points
            '''
            for i in range(len(b)):
                # dry all points
                b[i].isWet = False

            # get relative water depth
            rel_water_d = max_water_d + min(y1)
            if diurnal == True:
                rel_water_d = Cfunctions.water_level(max_water_d, t.hour, 0.2) + min(b_pos_y)

            # loop around all point objects and make the points wet
            for j in range(len(b)):
                if (b[j].position[1] <= rel_water_d):
                    b[j].isWet = True

            '''
            add boundary to dictionary
            '''

            if flipped == True:
                current_pos = [x.position for x in b]
                original_centre_x = (b0[0, 0] + b0[-1, 0]) / 2

                # reverse dictionary so all other attributes match
                flipped_b = deepcopy(b)
                flipped_b.reverse()
                current_pos = [x.position for x in flipped_b]

                # flip across the centre
                for i in range(len(current_pos)):
                    flipped_b[i].position = np.array((2 * original_centre_x - current_pos[i][0], current_pos[i][1]))
                # only save the solution every save_hrs hours
                if h % save_hrs == 0:
                    dict_sol[h] = deepcopy(flipped_b)
            else:
                if h % save_hrs == 0:
                    dict_sol[h] = deepcopy(b)

            '''
            advance time forwards by an hour
            '''
            t = t + dt.timedelta(hours=1)

    # create a dictionary of the two dictionaries
    output_dict = {'meta data': meta_data,
                   'data': dict_sol}
    # define end time
    timing_end = time.process_time()
    time2run = '{} s'.format(timing_end - timing_start)

    output_dict['meta data']['runtime'] = time2run

    return output_dict
