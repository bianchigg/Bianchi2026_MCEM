import numpy as np
from MCEM_files import Cfunctions
from numba import jit


@jit(nopython=True)
def rk_loop_ghost(temp_profile, n_temp_points, T_prev, delta_t, c_i, rho_i, k_i, dx, E):
    # this rk loop will have two options, depending on whether ice is at melt temperature or not
    if temp_profile[-1] < 273.15:
        # first rk step
        # print('RK Loop')
        T1 = np.copy(temp_profile)
        # define ghost point
        GP1 = T_prev[-2] + (2 * dx) / k_i * E[-1]
        # updating interior points
        for j in range(1, n_temp_points - 1):
            T1[j] = T_prev[j] + delta_t / (c_i * rho_i * 2) * (
                    k_i / dx ** 2 * (T_prev[j - 1] - 2 * T_prev[j] + T_prev[j + 1]) + E[j])
        # update end point using GP
        T1[-1] = T_prev[-1] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T_prev[-2] - 2 * T_prev[-1] + GP1) + E[-1])
        if T1[-1] >= 273.15:
            T1[-1] = 273.15

        # second rk step
        # define ghost point
        GP2 = T1[-2] + (2 * dx) / k_i * E[-1]
        T2 = np.copy(temp_profile)
        for j in range(1, n_temp_points - 1):
            T2[j] = T_prev[j] + delta_t / (c_i * rho_i * 2) * (
                    k_i / dx ** 2 * (T1[j - 1] - 2 * T1[j] + T1[j + 1]) + E[j])
        # update end point using GP
        T2[-1] = T_prev[-1] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T1[-2] - 2 * T1[-1] + GP2) + E[-1])
        if T2[-1] >= 273.15:
            T2[-1] = 273.15

        # third rk step
        # define ghost point
        GP3 = T2[-2] + (2 * dx) / k_i * E[-1]
        T3 = np.copy(temp_profile)
        for j in range(1, n_temp_points - 1):
            T3[j] = T_prev[j] + delta_t / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T2[j - 1] - 2 * T2[j] + T2[j + 1]) + E[j])
        # update end point using GP
        T3[-1] = T_prev[-1] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T2[-2] - 2 * T2[-1] + GP3) + E[-1])
        if T3[-1] >= 273.15:
            T3[-1] = 273.15

        # final rk step
        # define final GP
        GP4 = T3[-2] + (2 * dx) / k_i * E[-1]
        for j in range(1, n_temp_points - 1):
            K1 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T_prev[j - 1] - 2 * T_prev[j] + T_prev[j + 1]) + E[j])
            K2 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T1[j - 1] - 2 * T1[j] + T1[j + 1]) + E[j])
            K3 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T2[j - 1] - 2 * T2[j] + T2[j + 1]) + E[j])
            K4 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T3[j - 1] - 2 * T3[j] + T3[j + 1]) + E[j])
            # change of bc also here
            temp_profile[0] = temp_profile[1]
            temp_profile[j] = T_prev[j] + delta_t / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
            if temp_profile[j] >= 273.15:
                temp_profile[j] = 273.15

        # update end point with final rk step
        K1 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T_prev[-2] - 2 * T_prev[-1] + GP1) + E[-1])
        K2 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T1[-2] - 2 * T1[-1] + GP2) + E[-1])
        K3 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T2[-2] - 2 * T2[-1] + GP3) + E[-1])
        K4 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T3[-2] - 2 * T3[-1] + GP4) + E[-1])
        temp_profile[-1] = T_prev[-1] + delta_t / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        # to catch if above melt temp
        if temp_profile[-1] >= 273.15:
            temp_profile[-1] = 273.15
        # no change if still below 273.15
        length_difference = 0
        return temp_profile, length_difference

    else:
        # do rk loops with updated melt point GP
        # set end point to melt temp
        temp_profile[-1] = 273.15
        # latent heat of phase transition J/kg
        gamma = 335000
        # define sw_in
        inc_rad = E[-1]
        # first rk step
        T1 = np.copy(temp_profile)
        # estimate length difference first
        # temp diff between ice surface and point below
        t_diff = T_prev[-1] - T_prev[-2]

        # finding the total distance a point changes position
        rk1_ld = delta_t / (rho_i * gamma) * (- inc_rad + k_i * (t_diff) / dx)
        # define ghost point
        GP1 = T_prev[-2] + (2 * dx) / k_i * (E[-1] + rho_i * k_i * (rk1_ld / delta_t))
        # updating interior points
        for j in range(1, n_temp_points - 1):
            T1[j] = T_prev[j] + delta_t / (c_i * rho_i * 2) * (
                    k_i / dx ** 2 * (T_prev[j - 1] - 2 * T_prev[j] + T_prev[j + 1]) + E[j])
        # update end point using GP
        T1[-1] = T_prev[-1] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T_prev[-2] - 2 * T_prev[-1] + GP1) + E[-1])
        if T1[-1] >= 273.15:
            T1[-1] = 273.15

        # second rk step
        T2 = np.copy(temp_profile)
        # temp diff between ice surface and point below
        t_diff = T1[-1] - T1[-2]
        # finding the total distance a point changes position
        rk2_ld = delta_t / (rho_i * gamma) * (- inc_rad + k_i * (t_diff) / dx)

        GP2 = T1[-2] + (2 * dx) / k_i * (E[-1] + rho_i * k_i * (rk2_ld / delta_t))

        for j in range(1, n_temp_points - 1):
            T2[j] = T_prev[j] + delta_t / (c_i * rho_i * 2) * (
                    k_i / dx ** 2 * (T1[j - 1] - 2 * T1[j] + T1[j + 1]) + E[j])
        # update end point using GP
        T2[-1] = T_prev[-1] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T1[-2] - 2 * T1[-1] + GP2) + E[-1])
        if T2[-1] >= 273.15:
            T2[-1] = 273.15

        # third rk step
        T3 = np.copy(temp_profile)
        # temp diff between ice surface and point below
        t_diff = T2[-1] - T2[-2]
        # finding the total distance a point changes position
        rk3_ld = delta_t / (rho_i * gamma) * (- inc_rad + k_i * (t_diff) / dx)
        # define ghost point
        GP3 = T2[-2] + (2 * dx) / k_i * (E[-1] + rho_i * k_i * (rk3_ld / delta_t))

        for j in range(1, n_temp_points - 1):
            T3[j] = T_prev[j] + delta_t / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T2[j - 1] - 2 * T2[j] + T2[j + 1]) + E[j])
        # update end point using GP
        T3[-1] = T_prev[-1] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T2[-2] - 2 * T2[-1] + GP3) + E[-1])
        if T3[-1] >= 273.15:
            T3[-1] = 273.15

        # final rk step
        # temp diff between ice surface and point below
        t_diff = T3[-1] - T3[-2]
        # finding the total distance a point changes position
        rk4_ld = delta_t / (rho_i * gamma) * (- inc_rad + k_i * (t_diff) / dx)
        # define final GP
        GP4 = T3[-2] + (2 * dx) / k_i * (E[-1] + rho_i * k_i * (rk3_ld / delta_t))

        for j in range(1, n_temp_points - 1):
            K1 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T_prev[j - 1] - 2 * T_prev[j] + T_prev[j + 1]) + E[j])
            K2 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T1[j - 1] - 2 * T1[j] + T1[j + 1]) + E[j])
            K3 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T2[j - 1] - 2 * T2[j] + T2[j + 1]) + E[j])
            K4 = 1 / (c_i * rho_i) * (
                    k_i / dx ** 2 * (T3[j - 1] - 2 * T3[j] + T3[j + 1]) + E[j])
            # change of bc also here
            temp_profile[0] = temp_profile[1]
            temp_profile[j] = T_prev[j] + delta_t / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
            if temp_profile[j] >= 273.15:
                temp_profile[j] = 273.15

        # update end point with final rk step
        K1 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T_prev[-2] - 2 * T_prev[-1] + GP1) + E[-1])
        K2 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T1[-2] - 2 * T1[-1] + GP2) + E[-1])
        K3 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T2[-2] - 2 * T2[-1] + GP3) + E[-1])
        K4 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T3[-2] - 2 * T3[-1] + GP4) + E[-1])
        temp_profile[-1] = T_prev[-1] + delta_t / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        # to catch if above melt temp
        if temp_profile[-1] >= 273.15:
            temp_profile[-1] = 273.15

        # use final temp profile to work out the length difference in ice
        # temp diff between ice surface and point below
        t_diff = temp_profile[-1] - temp_profile[-2]
        # finding the total distance a point changes position
        length_difference = delta_t / (rho_i * gamma) * (- inc_rad + k_i * (t_diff) / dx)

        return temp_profile, length_difference


def IceWallMeltDPGhost(point, surface_tilt, surface_azimuth, sw_in, lw_in, lw_out, lst, sens, latent, day_no,
                       latitude, \
                       longitude, delta_t):
    '''
    point - this is an object
    A function which updates the temperature profile of ice column. Returns 0 if not melt occurs, else it returns the
    amount which has melted.
    '''
    # get actual sw
    sw_in = sw_in * np.cos(Cfunctions.incidence_angle(lat=latitude, hour_angle=Cfunctions.hour_angle(lst),
                                                      declination=Cfunctions.declination(day_no),
                                                      surface_azimuth=surface_azimuth,
                                                      observer_tilt=surface_tilt))
    # catch for if incidence angle is negative
    if sw_in < 0:
        sw_in = 0

    # define incoming radiaiton
    # lw out will depend on surface temp of ice
    # emissivity of ice
    ems = 0.98
    sb_const = 5.67 * 10 ** (-8)
    lw_out = ems * sb_const * np.copy(point.columnTempProfile[-1]) ** 4

    # assigning each object the relevant energies at surface
    point.sw = (1 - point.albedo) * sw_in
    point.lw = lw_in - lw_out
    point.sens = sens
    point.latent = latent
    point.aoi = Cfunctions.incidence_angle(lat=latitude, hour_angle=Cfunctions.hour_angle(lst),
                                           declination=Cfunctions.declination(day_no),
                                           surface_azimuth=surface_azimuth,
                                           observer_tilt=surface_tilt)

    point.surfaceEnergy = sw_in + lw_in - lw_out + sens + latent
    inc_rad = (1 - point.albedo) * sw_in + lw_in - lw_out + sens + latent
    # only 0.45 of solar radiation is penetrating
    pen_rad = ((1 - point.albedo) * sw_in * 0.45)

    # define blank E array
    E = np.zeros(point.noTempPoints)
    idx = np.arange(0, point.noTempPoints)
    # get the distance between the profile points
    dx = point.columnPointSpacing
    # populate E with energy at ice depths
    # absorption coeff
    abs_coeff = 0.069
    # attenuation coefficient
    att_coeff = 1.754
    E[idx] = pen_rad * att_coeff * np.exp(-(point.noTempPoints - idx - 1) * dx * att_coeff)
    E[-1] = inc_rad
    # thermal conductivity of ice w/(mK)
    k_i = 2.22
    # ice density kg/m3
    rho_i = 917
    # specific heat of ice J/(kg K)
    c_i = 2090
    # latent heat of phase transition J/kg
    gamma = 335000

    # define delta t based on second steps to take within an hour, delta_t in seconds
    delta_t = delta_t
    # 3600 seconds in an hour
    no_steps = int(3600 / delta_t)
    # define blank melt amount
    melt_tot = 0
    '''
    for loop here to loop round seconds to get an hours worth of update steps
    '''
    for i in range(no_steps):
        '''
        RK Method
        '''
        # define temperature profile at previous time
        T_prev = np.copy(point.columnTempProfile)

        temp_profile = point.columnTempProfile
        n_temp_points = point.noTempPoints
        new_temp_profile, length_difference = rk_loop_ghost(temp_profile, n_temp_points, T_prev, delta_t, c_i, rho_i,
                                                            k_i, dx, E)
        point.columnTempProfile = new_temp_profile

        if length_difference < 0:
            melt_tot += length_difference

    return abs(melt_tot)


@jit(nopython=True)
def rk_loop(temp_profile, n_temp_points, T_prev, delta_t, c_i, rho_i, k_i, dx, E):
    # first rk step
    # print('RK Loop')
    T1 = np.copy(temp_profile)
    # updating interior points
    for j in range(1, n_temp_points - 1):
        T1[j] = T_prev[j] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T_prev[j - 1] - 2 * T_prev[j] + T_prev[j + 1]) + E[j])
        if T1[j] >= 273.15:
            T1[j] = 273.15

    # second rk step
    T2 = np.copy(temp_profile)
    for j in range(1, n_temp_points - 1):
        T2[j] = T_prev[j] + delta_t / (c_i * rho_i * 2) * (
                k_i / dx ** 2 * (T1[j - 1] - 2 * T1[j] + T1[j + 1]) + E[j])
        if T2[j] >= 273.15:
            T2[j] = 273.15
    # third rk step
    T3 = np.copy(temp_profile)
    for j in range(1, n_temp_points - 1):
        T3[j] = T_prev[j] + delta_t / (c_i * rho_i) * (
                k_i / dx ** 2 * (T2[j - 1] - 2 * T2[j] + T2[j + 1]) + E[j])
        if T3[j] >= 273.15:
            T3[j] = 273.15

    # final rk step
    for j in range(1, n_temp_points - 1):
        # change of bc here
        T_prev[0] = T_prev[1]
        K1 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T_prev[j - 1] - 2 * T_prev[j] + T_prev[j + 1]) + E[j])
        K2 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T1[j - 1] - 2 * T1[j] + T1[j + 1]) + E[j])
        K3 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T2[j - 1] - 2 * T2[j] + T2[j + 1]) + E[j])
        K4 = 1 / (c_i * rho_i) * (
                k_i / dx ** 2 * (T3[j - 1] - 2 * T3[j] + T3[j + 1]) + E[j])
        # change of bc also here
        temp_profile[0] = temp_profile[1]
        temp_profile[j] = T_prev[j] + delta_t / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        if temp_profile[j] >= 273.15:
            temp_profile[j] = 273.15
    return temp_profile


def IceWallMeltWP(point, surface_tilt, surface_azimuth, sw_in, lw_in, lw_out, lst, sens, latent, day_no, latitude,
                  longitude, air_temp, beta, water_t, water_v, rel_water_d, water_flux, P, wall_force, delta_t):
    # get actual sw
    sw_in = sw_in * np.cos(Cfunctions.incidence_angle(lat=latitude, hour_angle=Cfunctions.hour_angle(lst),
                                                      declination=Cfunctions.declination(day_no),
                                                      surface_azimuth=surface_azimuth,
                                                      observer_tilt=surface_tilt))
    # catch for if incidence angle is negative
    if sw_in < 0:
        sw_in = 0
    # then get sw after it has been attenuated through water, assume water travel depth is depth of point in the water
    # water albedo - NOTE this changes with angle of incidence but keep the same for now
    water_albedo = 0.03
    water_d = rel_water_d - point.position[1]
    # 0.45 since only 45% of sw penetrates i.e is PAR
    inc_rad = sw_in * np.exp(-0.1343 * water_d) * (1 - point.albedo) * (1 - water_albedo)
    pen_rad = 0.45 * inc_rad
    # define blank E array
    E = np.zeros(point.noTempPoints)
    idx = np.arange(0, point.noTempPoints)
    # get the distance between the profile points
    dx = point.columnPointSpacing
    # populate E with energy at ice depths -  will say only sw radiation penetrates for now
    '''
    Need to think here about what inc radiation will penetrate the ice
    '''
    # attenuation coefficient
    att_coeff = 1.754
    # E should be max at max idx as that is surface
    E[idx] = pen_rad * att_coeff * np.exp(-(point.noTempPoints - idx - 1) * dx * att_coeff)
    E[-1] = pen_rad

    # thermal conductivity of ice w/(mK)
    k_i = 2.22
    # ice density kg/m3
    rho_i = 917
    # specific heat of ice J/(kg K)
    c_i = 2090
    # latent heat of phase transition kJ/kg
    gamma = 335000

    # assume surface point is at melt temp always
    # trying without this assumption
    # point.columnTempProfile[-1] = 273.15
    # define temperature profile at previous time
    T_prev = np.copy(point.columnTempProfile)  # JE - do you need to update T_prev in the loop over i?
    # define delta t based on second steps to take within an hour, delta_t in seconds
    delta_t = delta_t
    # 3600 seconds in an hour
    no_steps = int(3600 / delta_t)
    # define blank melt amount
    melt_tot = 0

    '''
    for loop here to loop round seconds to get an hours worth of update steps
    '''
    for i in range(no_steps):
        '''
        RK Method
        '''
        temp_profile = point.columnTempProfile
        n_temp_points = point.noTempPoints
        new_temp_profile = rk_loop(temp_profile, n_temp_points, T_prev, delta_t, c_i, rho_i, k_i, dx, E)
        point.columnTempProfile = new_temp_profile

        '''
        Melt rate calculations
        '''
        # work out how much melt has happened at this time step
        water_t = water_t
        melt_t = 273.15
        # flux from water into ice
        rho_w = 1000
        cw = 4184
        # Prandtl number for water at 0 degrees, madruga 2023 13.48
        Pr = 13.48

        Qw = wall_force * cw * (water_t - melt_t) * Pr ** (-2 / 3) / water_v
        Qw = wall_force * cw * (water_t - melt_t) / (water_v + 8.85 * np.sqrt(wall_force / rho_w) * (Pr - 1))

        # htc = wall_force * cw / (water_v + 8.85 * np.sqrt(wall_force / rho_w) * (Pr - 1))
        # print('heat flux from shear', Qw)
        # flux from ice into water, check thermal conductivity
        # choosing temperature of ice in far field i.e at bottom of considered temperature profile

        Qi = k_i * (273.15 - point.columnTempProfile[-2]) / dx
        # print('temp just below surface', point.columnTempProfile[-2])
        # print('Qi close', Qi)

        # Qi = k_i * (273.15 - point.columnTempProfile[0]) / point.columnDepth
        # # print('Qi far field wp', Qi)

        # solar flux at ice surface
        Qs = pen_rad
        # atmospheric fluxes at air water interface
        rho_air = 1.29
        ca = 1005
        # normalised heat transfer coefficient air water - CHECK source but roughly in range
        Cah = 5E-3

        '''
        Neglecting for now as unsure of parametrisation, currently becomes much too cold in winter months
        this could be due to the fact that it's likely there is no water in winter months
        will need to rewrite code so that this can be accounted for
        # does Qa really make it to the ice?
        Qa = rho_air * ca * Cah * (air_temp - water_t)
        '''
        # Qa = rho_air * ca * Cah * (air_temp - water_t)
        # =
        # assigning each object the relevant energies at surface
        point.sw = Qs
        # note no lw out included in this
        # point.lw = Qa
        point.shear = Qw
        point.aoi = Cfunctions.incidence_angle(lat=latitude, hour_angle=Cfunctions.hour_angle(lst),
                                               declination=Cfunctions.declination(day_no),
                                               surface_azimuth=surface_azimuth,
                                               observer_tilt=surface_tilt)
        point.surfaceEnergy = Qw - Qi + Qs

        # latent heat of fusion
        L = 3.35 * 10 ** 5

        temp_test = point.columnTempProfile
        # melt increment
        inc = delta_t / (rho_i * L) * (Qw - Qi + Qs)
        # print(inc)
        if inc < 0:
            inc = 0
        # have to assume melt
        melt_tot += inc

        # updating end point
        # need to add check for melt temp
        point.columnTempProfile[-1] = point.columnTempProfile[-2] + dx / k_i * (rho_i * L * inc + Qw + Qs)
        if point.columnTempProfile[-1] >= 273.15:
            point.columnTempProfile[-1] = 273.15
    # print('shear', Qw)
    # print('ice', Qi)
    # print('point just below surface', point.columnTempProfile[-2])
    # print('solar', Qs)
    return melt_tot
