import numpy as np
import math
# import matplotlib.pyplot as plt
from MCEM_files import ObjectBoundary
from scipy.ndimage import zoom
from numba import jit, prange
from scipy.ndimage import gaussian_filter1d
# from shapely.geometry import Polygon, LineString
from scipy.interpolate import interp1d
from itertools import combinations


def u_boundary(width, depth, res):

    """
    Generate a symmetric U-shaped channel boundary represented as a parabola.

    Parameters
    ----------
    width : float
        Total channel top width (units: m).
    depth : float
        Channel depth from top to lowest point (units: m).
    res : int
        Number of points used for the boundary resolution.

    Returns
    -------
    b0 : (N, 2) np.ndarray
        Array of (x, z) coordinates describing the U-shaped boundary.
        x units: m
        z units: m (vertical elevation)
    """

    # parabola equation is z = p x^2

    # define width points
    x = np.linspace(-width / 2, width / 2, res)
    # define the parabola
    max_x = x[0] ** 2
    p = depth / max_x
    z = p * x ** 2

    # translate into correct frame, we choose a nominal ice depth of 500m and centre it at x = 50
    z = z + 500 - max(z)
    x = x + 50
    # join arrays
    b0 = np.vstack((x, z))
    b0 = np.transpose(b0)
    return b0


def fit_half_parabola(x1, z1, xv, zv):
    """
    Fit a parabola of the form:  z = a * (x - xv)^2 + zv

    Parameters
    ----------
    x1, z1 : float
        Known point on the parabola.
    xv, zv : float
        Vertex point of the parabola.

    Returns
    -------
    parabola : Callable
        Function parabola(x) returning z values.
    """

    # Use vertex form: z = a*(x - xv)^2 + zv
    # Plug in point (x1, z1) to solve for 'a'
    a = (z1 - zv) / (x1 - xv) ** 2
    return lambda x: a * (x - xv) ** 2 + zv


def two_sided_u(x_left, x_right, x_center, z_bottom, z_top, res=100):
    """
    Construct an asymmetric U-shaped boundary from two different half-parabolas.

    Parameters
    ----------
    x_left : float
        Leftmost x-coordinate of the boundary.
    x_right : float
        Rightmost x-coordinate of the boundary.
    x_center : float
        x-coordinate of the vertex (lowest point).
    z_bottom : float
        Vertical elevation of the vertex (units: m).
    z_top : float
        Vertical elevation at channel banks (units: m).
    res : int
        Total number of points to generate.

    Returns
    -------
    boundary : (N,2) np.ndarray
        Combined x–z coordinate pairs for the U shape.
    """

    # Left parabola from x_left to x_center
    x_left_vals = np.linspace(x_left, x_center, res // 2)
    left_parabola = fit_half_parabola(x_left, z_top, x_center, z_bottom)
    z_left_vals = left_parabola(x_left_vals)

    # Right parabola from x_center to x_right
    x_right_vals = np.linspace(x_center, x_right, res // 2)
    right_parabola = fit_half_parabola(x_right, z_top, x_center, z_bottom)
    z_right_vals = right_parabola(x_right_vals)

    # Combine both halves
    x_full = np.concatenate([x_left_vals, x_right_vals])
    z_full = np.concatenate([z_left_vals, z_right_vals])

    return np.column_stack((x_full, z_full))

def perimeter(boundary):
    """
    Compute the perimeter length of a polyline boundary.

    Parameters
    ----------
    boundary : (N,2) array_like
        Sequence of (x, y) coordinate pairs.

    Returns
    -------
    perim : float
        Total perimeter length (units: m).
    """

    # set perimeter to be 0
    perim = 0
    # loop around all elements in boundary array apart from the final
    for i in range(len(boundary) - 1):
        # math.dist works out eucliden distance between 2 points
        perim += math.dist(boundary[i], boundary[i + 1])

    return perim


def area_finder(boundary):
    """
    Approximate cross-sectional area using trapezoidal slices
    between mirrored boundary points.

    Parameters
    ----------
    boundary : (N,2) array_like
        Channel boundary coordinates (x, y).

    Returns
    -------
    trap_area : float
        Estimated area (units: m²).
    """

    b = boundary
    # initialise trapezoid area
    trap_area = 0

    # left index of centre
    lind = int(len(b) / 2 - 1)

    # loop around half the number of points in array as connecting trapezes.
    for i in range(lind, -1, -1):
        # calculate area of trapezium
        # can generalise: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y
        # -coordinates/30408825#30408825
        w1 = math.dist(b[i], b[len(b) - i - 1])
        w2 = math.dist(b[i + 1], b[len(b) - i - 1 - 1])
        h = b[i][1] - b[i + 1][1]

        # add trapezoid area to current area count
        trap_area += (w1 + w2) * h * 0.5

    return trap_area


def water_level(max_lvl, hour, variation):
    """
    Compute diurnal water level using a cosine function.

    Parameters
    ----------
    max_lvl : float
        Maximum water level (m).
    hour : float
        Hour of the day (0–24).
    variation : float
        Total peak-to-peak water-level variation (m).

    Returns
    -------
    wl : float
        Water level (m).
    """

    # 0.05 for 0.1cm variation
    wl = variation / 2 * np.cos((16 - hour) * np.pi / 12) + (max_lvl - variation / 2)
    return wl


## below is better than trape
def polyarea(b):
    """
    Compute polygon area using the shoelace formula.
    taken from https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    Parameters
    ----------
    b : (N,2) array_like
        Polygon vertices (x, y).

    Returns
    -------
    area : float
        Polygon area (units: m²).
    """

    b = np.array(b)
    x, y = b.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))



def polyarea_lowest_y(b, n=4):
    """
    Compute the polygon area using the lowest n points (by y-value).

    Parameters
    ----------
    b : (N,2) array_like
        Boundary coordinates.
    n : int
        Number of lowest-elevation points to include.

    Returns
    -------
    area : float
        Area of polygon formed by lowest n points (m²).
    req_idx_sorted : np.ndarray
        Indices of the selected points.
    """

    # convert to an array
    b = np.array(b)
    # get the y values
    by = b[:, 1]
    # sort the indicies by height
    sorted_indicies = by.argsort()
    # get required indicies
    req_idx = sorted_indicies[:n]
    # re-sort into ascending order
    req_idx_sorted = np.sort(req_idx)
    # calculate area with lowest n points
    area = polyarea(b[req_idx_sorted])

    return area, req_idx_sorted


def bucket_area(curve_points):
    """
    Compute area between a curve and its maximum height by reflecting
    the curve horizontally and forming a closed polygon.

    Parameters
    ----------
    curve_points : (N,2) np.ndarray
        Curve coordinates (x, y).

    Returns
    -------
    area : float
        Shaded or filled area (m²).
    """

    curve_points = np.array(curve_points)
    max_y = np.max(curve_points[:, 1])

    # Mirror curve along top
    reflected = curve_points[::-1].copy()
    reflected[:, 1] = max_y

    # Form the full polygon
    polygon = np.vstack([curve_points, reflected])

    return polyarea(polygon)


def area_properties(Area, boundary):
    """
    Determine properties of a wetted cross-section for a known water area.

    Parameters
    ----------
    Area : float
        Known water cross-sectional area (m²).
    boundary : (N,2) np.ndarray
        Channel boundary coordinates (x, y).

    Returns
    -------
    Hmax : float
        Water surface elevation relative to boundary (m).
    width : float
        Water-surface width (m).
    P_points : (K,2) np.ndarray
        Wetted-perimeter coordinate points.
    """


    A = np.copy(Area)
    b = np.copy(boundary)

    # initialise trapezoid area
    trap_area = 0
    old_trap_area = 0

    # left index of centre
    lind = int(len(b) / 2 - 1)
    # right index of centre
    rind = int(len(b) / 2)

    # central points may not be at the middle indicies
    # instead find the points with the lowest y value
    min_ind = np.where(b[:, 1] == b[:, 1].min())[0]

    # if there is more than one minimum value select the middle one
    if len(min_ind) > 1:
        # could be many points with lowest value, choose median two
        min_ind = int(np.median(min_ind))
        # get lind
        lind = int(min_ind)
        rind = lind + 1
    else:
        # check here to see if median value is left or right of central
        # might make a difference when calculating area
        # find middle x value of boundary
        middle_x = np.median(b[:, 0])
        # if median lowest point is left of this do nothing
        if b[min_ind, 0] < middle_x:
            pass
        # if to the right, move left by one index
        elif b[min_ind, 0] > middle_x:
            min_ind = min_ind - 1
        lind = int(min_ind)
        rind = lind + 1

    # define blank area estimate
    area_estimate = 0
    # number of boundary points to start area estimate
    n = 4
    # loop around half the number of points in array as connecting trapezes.
    for i in range(lind, -1, -1):
        if area_estimate < Area:
            # estimate area using polyarea formula
            area_estimate, wet_idx = polyarea_lowest_y(b, n)
            n += 2
            # index of top points of the fill
            stop_ind = i + 1

    Hmax = b[stop_ind, 1]
    width = b[wet_idx[0], 0] - b[wet_idx[-1], 0]

    # get the points of the wetted perimeter
    P_points = b[wet_idx, :]

    return Hmax, width, P_points


def local_solar_time(long, dn, lt):
    """
    Compute Local Solar Time (LST) including equation-of-time correction.

    Parameters
    ----------
    long : float
        Longitude (radians).
    dn : int
        Day number of the year (1–365).
    lt : float
        Local clock time (hours).

    Returns
    -------
    lst : float
        Local Solar Time (hours).
    """

    # input in radians where applicable
    # Time zone estimate: one time zone per 15 degrees (π/12 radians)
    if long < 0:
        tz = math.floor(long / (np.pi / 12))
    elif long > 0:
        tz = math.ceil(long / (np.pi / 12))
    else:
        tz = 0

    # Local Standard Time Meridian in radians
    lstm = (np.pi / 12) * tz

    # Equation of Time (EoT) — use radians consistently
    b = 2 * np.pi * (dn - 81) / 364  # radians version of 360/365 * (dn - 81)
    eot_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)  # minutes

    # Time Correction Factor (TC) in minutes (radians to degrees: multiply by 180/π)
    tc_min = 4 * ((long - lstm) * 180 / np.pi) + eot_min

    # Convert correction to hours and add to local time
    lst = lt + tc_min / 60

    return lst


def hour_angle(lst):
    """
    Compute solar hour angle from Local Solar Time.

    Parameters
    ----------
    lst : float
        Local Solar Time (hours).

    Returns
    -------
    omega : float
        Hour angle (radians).
    """

    omega = 15 * (lst - 12)
    return np.deg2rad(omega)


def declination(dn):
    """
    Compute solar declination angle using the Spencer (1971) approximation.

    Parameters
    ----------
    dn : int
        Day number of the year (1–365).

    Returns
    -------
    delta : float
        Solar declination (radians).
    """

    # equation in radians (spenser 1971)
    gamma = 2 * np.pi * (dn - 1) / 365
    delta = 0.006918 - 0.339912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma) + (
            0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    return delta

def azimuth(lat, declination, solar_alt):
    # issue whn solar altitude angle is 90 (pi/2)
    # in this case azimuth is irrelevant
    if solar_alt == np.pi / 2:
        return np.pi
    cos_zeta = (np.sin(solar_alt) * np.sin(lat) - np.sin(declination)) / (np.cos(solar_alt) * np.cos(lat))
    # issue arises when values get to edges of horizon
    # only at latitude 0 so going to pass on solving for now
    azimuth = np.arccos(cos_zeta)
    # azimuth limit is from 0 to pi

    if np.isnan(azimuth) == True:
        # if nan set = 180 = pi, not real world correct but produces results at this point solar alt is below horizon
        # so is irrelevant
        azimuth = np.pi  # this is wrong but ok
    return azimuth


def solar_alt(lat, hour_angle, declination):
    """
    Solar altitude angle above the horizon.

    Parameters
    ----------
    lat : float
        Latitude (radians).
    hour_angle : float
        Solar hour angle (radians).
    declination : float
        Solar declination (radians).

    Returns
    -------
    alt_val : float
        Solar altitude angle (radians).
    """

    sin_alpha = (np.cos(lat) * np.cos(declination)) * np.cos(hour_angle) + np.sin(lat) * np.sin(declination)
    alt_val = np.arcsin(sin_alpha)

    return alt_val


def shade_alt(channel_boundary, channel_orientation, solar_time, day_no, latitude):
    """
    Compute the height of shading cast inside a U-shaped channel geometry.

    Parameters
    ----------
    channel_boundary : (N,2) np.ndarray
        Channel boundary coordinate pairs (x, y).
    channel_orientation : float
        Channel orientation angle (radians from north)
    solar_time : float
        Local solar time (hours).
    day_no : int
        Day of year (1–365).
    latitude : float
        Latitude (radians).

    Returns
    -------
    H_fin : float
        Height of shade inside channel (m).
        Returns 1 if full shade, 0 if no shade.
    side : str
        'l' or 'r' indicating which wall casts the shade.
        'b' for both or special cases.
    """


    # issue when solar altitude angle is 90 (pi/2)
    # in this case azimuth is irrelevant
    if solar_alt == np.pi / 2:
        return np.pi
    cos_zeta = (np.sin(solar_alt) * np.sin(lat) - np.sin(declination)) / (np.cos(solar_alt) * np.cos(lat))
    # issue arises when values get to edges of horizon
    # only at latitude 0 so going to pass on solving for now
    azimuth = np.arccos(cos_zeta)
    # azimuth limit is from 0 to pi

    if np.isnan(azimuth) == True:
        # if nan set = 180 = pi, not real world correct but produces results at this point solar alt is below horizon
        # so is irrelevant
        azimuth = np.pi  # this is wrong but ok
    return azimuth


def shade_alt(channel_boundary, channel_orientation, solar_time, day_no, latitude):
    """
    Compute the shading height within a U-shaped channel based on solar geometry.

    Parameters
    ----------
    channel_boundary : ndarray (N, 2)
        XY coordinates describing the channel boundary, likely forming a "U" shape.
    channel_orientation : float
        Orientation of the channel in radians (0–2π).
    solar_time : float
        Local solar time in hours (0–24).
    day_no : int
        Day of the year (1–365 or 366).
    latitude : float
        Latitude of the site in radians.

    Returns
    -------
    H_fin : float
        Final shading height inside the channel. Returns 0 when fully illuminated.
    side : {'l', 'r', 'b'}
        'l' = shade cast on left wall
        'r' = shade cast on right wall
        'b' = sun below horizon or channel fully lit

    Notes
    -----
    This function models solar shading by constructing internal “V” geometry within
    the U-shaped boundary and performing multiple trigonometric projections to
    determine shadow height.
    """


    '''
    Define inscribed V on the U channel
    '''
    # define V inside U boundary
    # define the middle xpoint in the boundary
    midpoint_x = (channel_boundary[0, 0] + channel_boundary[-1, 0]) / 2
    # define the mid point
    midpoint = np.array((midpoint_x, min(channel_boundary[:, 1])))
    # create a linspace of all the x point needed for each side
    lhs_x = np.linspace(midpoint_x, channel_boundary[0, 0], 100)
    rhs_x = np.linspace(midpoint_x, channel_boundary[-1, 0], 100)
    lhs_y = np.linspace(min(channel_boundary[:, 1]), channel_boundary[0, 1], 100)
    rhs_y = np.linspace(min(channel_boundary[:, 1]), channel_boundary[-1, 1], 100)
    # define array
    lhs = np.column_stack((lhs_x, lhs_y))
    rhs = np.column_stack((rhs_x, rhs_y))

    # define direction vectors
    lhs_dir = array2direction_v(lhs)
    rhs_dir = array2direction_v(rhs)
    # find opening angle
    gamma = np.arccos(np.dot(lhs_dir, rhs_dir) / (np.linalg.norm(lhs_dir) * np.linalg.norm(rhs_dir)))
    '''
    defining needed solar angles
    '''
    # declination
    dec = declination(day_no)
    # hour angle
    ha = hour_angle(solar_time)
    # sun altitude
    beta = solar_alt(latitude, ha, dec)
    # if sun is below horizon return all shade
    if beta < 0:
        return 1, 'b'
    # azimuth
    az = azimuth(latitude, dec, beta)
    # condition to make azimuth go to 2 pi
    if 12 < solar_time < 24:
        az = 2 * np.pi - az

    '''
    deciding if shade is going to be cast on lhs or rhs of channel
    '''
    if 0 < az < channel_orientation:
        side = 'r'
    elif np.pi + channel_orientation < az <= 2 * np.pi:
        side = 'r'
    # cases to catch if sun is aligned with the channel, i.e parallel
    elif az == channel_orientation:
        return 0, 'b'
    elif az == channel_orientation + np.pi:
        return 0, 'b'
    else:
        side = 'l'

    '''
    working out flat angle, alpha, based on side which shade is being cast
    '''
    if side == 'l':
        # define v points
        T = rhs
        T_len = math.dist((T[0, 0], T[0, 1]), (T[-1, 0], T[-1, 1]))
        # define array connecting top of V side and extending to the right
        # keep y same as horizontal vector
        Fx = np.linspace(T[-1, 0], T[-1, 0] + 1, 100)
        Fy = np.ones(100) * T[-1, 1]
        F = np.column_stack((Fx, Fy))
        # get direction vectors for both
        T_dir = array2direction_v(T)
        V_dir = array2direction_v(F)
        # get alpha
        alpha = np.pi - np.arccos(np.dot(V_dir, T_dir) / (np.linalg.norm(V_dir) * np.linalg.norm(T_dir)))

    elif side == 'r':
        # define v points
        T = lhs
        T_len = math.dist((T[0, 0], T[0, 1]), (T[-1, 0], T[-1, 1]))
        # define array connecting top of V side and extending to the right
        # keep y same as horizontal vector
        Fx = np.linspace(T[-1, 0], T[-1, 0] - 1, 100)
        Fy = np.ones(100) * T[-1, 1]
        F = np.column_stack((Fx, Fy))
        # get direction vectors for both
        T_dir = array2direction_v(T)
        V_dir = array2direction_v(F)
        # get alpha
        alpha = np.pi - np.arccos(np.dot(V_dir, T_dir) / (np.linalg.norm(V_dir) * np.linalg.norm(T_dir)))

    '''
    defining inside V angles and work out length of shade on V, M
    '''
    # condition to catch if sun always shining in crevasse
    if alpha + beta > np.pi:
        return 0, 'b'

    mu = np.pi - alpha - beta
    theta = np.pi - gamma - mu
    M = T_len * np.sin(mu) / np.sin(theta)

    '''
    working out the height of shade cast on the V
    '''
    eta = alpha - gamma
    H = np.sin(eta) * M

    '''
    working on the smaller triangle to find the height of shade cast on 
    '''
    delta = np.pi / 2 - eta
    rho = np.pi - delta - theta

    # finding hypotenuse of smaller triangle
    R = np.sin(rho) * H / np.sin(theta)

    # finding height
    H_fin = np.sin(eta) * R / np.sin(np.pi / 2)

    return H_fin, side





def slope(x1, y1, x2, y2):
    """
    Compute the slope of a line defined by two points.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of the first point.
    x2, y2 : float
        Coordinates of the second point.

    Returns
    -------
    float
        Slope of the line (dy/dx).
    """
    return (y2 - y1) / (x2 - x1)



def angle(s1, s2):
    """
    Compute the angle (in degrees) between two slopes.

    Parameters
    ----------
    s1 : float
        First slope.
    s2 : float
        Second slope.

    Returns
    -------
    float
        Included angle between the two slopes, in degrees.
    """
    return math.degrees(math.atan((s2 - s1) / (1 + (s2 * s1))))




def array2direction_v(v):
    """
    Convert an array of XY coordinates into a unit direction vector
    from the last point to the first point.

    Parameters
    ----------
    v : ndarray (N, 2)
        Array of coordinates.

    Returns
    -------
    list of float
        Unit direction vector [dx, dy].
    """
    v_dist = [v[0, 0] - v[-1, 0], v[0, 1] - v[-1, 1]]
    v_norm = math.sqrt(v_dist[0] ** 2 + v_dist[1] ** 2)
    v_dir = [v_dist[0] / v_norm, v_dist[1] / v_norm]
    return v_dir


def flatSurface2actual_sw(inc_sw, time, latitude, longitude):
    """
    Convert incoming shortwave radiation measured on a flat surface to
    actual incident shortwave radiation on a horizontal plane.

    Parameters
    ----------
    inc_sw : float
        Incoming shortwave radiation on a flat detector.
    time : datetime
        Timestamp from which the day-of-year and hour are extracted.
    latitude : float
        Latitude in radians.
    longitude : float
        Longitude in degrees or radians, depending on local_solar_time().

    Returns
    -------
    float
        Actual incident shortwave radiation corrected for solar incidence angle.
        Returns 0 when the sun is below the local horizon.
    """
    dn = time.timetuple().tm_yday
    lt = time.timetuple().tm_hour
    lst = local_solar_time(longitude, dn, lt)
    ha = hour_angle(lst)
    dec = declination(dn)
    inc_ang = incidence_angle(latitude, ha, dec, 0, 0)

    E = inc_sw * np.cos(inc_ang)
    if E < 0:
        E = 0
    return E


def incidence_angle(lat, hour_angle, declination, surface_azimuth, observer_tilt):
    """
    Compute the angle of solar incidence on a tilted surface.

    Parameters
    ----------
    lat : float
        Latitude in radians.
    hour_angle : float
        Solar hour angle in radians.
    declination : float
        Solar declination in radians.
    surface_azimuth : float
        Surface azimuth angle in radians.
    observer_tilt : float
        Tilt of the observing surface in radians.

    Returns
    -------
    float
        Solar incidence angle (radians) between the sunlight vector and surface normal.
    """

    surface_azimuth = (surface_azimuth + np.pi) % (2 * np.pi)
    # ensures surface azimuth stays in 0 - 2pi range
    cos_theta = (
            np.sin(lat) * np.sin(declination) * np.cos(observer_tilt)
            - np.cos(lat) * np.sin(declination) * np.cos(surface_azimuth) * np.sin(observer_tilt)
            + np.cos(lat) * np.cos(declination) * np.cos(hour_angle) * np.cos(observer_tilt)
            + np.sin(lat) * np.cos(declination) * np.cos(hour_angle) * np.sin(observer_tilt) * np.cos(surface_azimuth)
            + np.cos(declination) * np.sin(hour_angle) * np.sin(observer_tilt) * np.sin(surface_azimuth)
    )

    # Clip to valid domain for arccos
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # If sun is behind the surface, return 90 degrees
    inc_angle = np.arccos(cos_theta)

    return inc_angle



def pointsPositionUpdater(point, norm_gradient, mloc):
    """
    Move a point a given distance `mloc` along a surface normal defined by `norm_gradient`.

    Parameters
    ----------
    point : object
        Object containing `.position` as a mutable array-like [x, y].
    norm_gradient : float or 'not_defined'
        Gradient of the surface normal. If 'not_defined', movement is vertical.
    mloc : float
        Distance to move the point.

    Notes
    -----
    This function solves for the next coordinates using a quadratic system and
    chooses appropriate roots based on gradient sign.
    """
    ...


    # to catch if tangent gradient is 0 then normal has infinite gradient
    if norm_gradient == 'not_defined':
        # print('norm grad not defined trigger')
        point.position[1] = point.position[1] - mloc

    else:

        old_b_pos = np.copy(point.position)
        b_pos = point.position
        # define coefficients to find x2
        x2_coeff = [1 + norm_gradient ** 2, -2 * norm_gradient ** 2 * b_pos[0] - 2 * b_pos[0],
                    b_pos[0] ** 2 - mloc ** 2 + norm_gradient ** 2 * b_pos[0] ** 2]

        x2s = np.real(np.roots(x2_coeff))
        # roots correct, sometimes include small (negligible) complex part, choose to discard, solving by
        # hand with quadratic formula has no complex but is likely slower
        # this will give a two values value, how to choose which one?
        if norm_gradient < 0:
            y2 = b_pos[1] + norm_gradient * (b_pos[0] - x2s[1])
            # update position
            point.position[0] = x2s[0]
            point.position[1] = y2
        elif norm_gradient > 0:
            y2 = b_pos[1] + norm_gradient * (b_pos[0] - x2s[0])
            # update position
            point.position[0] = x2s[1]
            point.position[1] = y2

        # to catch if point increases in height
        # if old_b_pos[1] < point.position[1]:
        #     point.position[1] = old_b_pos[1]

        if mloc > 0.02:
            print('diff between old and new points', math.dist(old_b_pos, point.position))





def interpcurve(N, pX, pY):
    """
    Redistribute points along a curve so that they are equally spaced by arc length.

    Parameters
    ----------
    N : int
        Number of new interpolated points.
    pX : ndarray
        X-coordinates of the original curve.
    pY : ndarray
        Y-coordinates of the original curve.

    Returns
    -------
    ndarray (N, 2)
        New curve with evenly spaced points.

    Notes
    -----
    Adapted from:
    https://stackoverflow.com/questions/18244305/how-to-redistribute-points-evenly-over-a-curve
    """
    ...

    # equally spaced in arclength
    N = np.transpose(np.linspace(0, 1, N))

    # how many points will be uniformly interpolated?
    nt = N.size

    # number of points on the curve
    n = pX.size
    pxy = np.array((pX, pY)).T
    p1 = pxy[0, :]
    pend = pxy[-1, :]
    last_segment = np.linalg.norm(np.subtract(p1, pend))

    epsilon = 10 * np.finfo(float).eps

    # #IF the two end points are not close enough lets close the curve
    # if last_segment > epsilon*np.linalg.norm(np.amax(abs(pxy),axis=0)):
    #     pxy=np.vstack((pxy,p1))
    #     nt = nt + 1
    # else:
    #     print('Contour already closed')

    pt = np.zeros((nt, 2))

    # Compute the chordal arclength of each segment.
    chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength
    cumarc = np.append(0, np.cumsum(chordlen))

    tbins = np.digitize(N, cumarc)  # bin index in which each N is in

    # catch any problems at the ends
    tbins[np.where(np.bitwise_or(tbins <= 0, N <= 0))] = 1
    tbins[np.where(np.bitwise_or(tbins >= n, N >= 1))] = n - 1

    s = np.divide((N - cumarc[tbins - 1]), chordlen[tbins - 1])
    pt = pxy[tbins - 1, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

    return pt


def interpcurve_new(N, xpoints, ypoints):
    # Step 1: Compute cumulative distance along the path
    points = np.stack((xpoints, ypoints), axis=-1)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    distances = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Step 2: Interpolate x and y as functions of distance
    fx = interp1d(distances, xpoints, kind='cubic', bounds_error=False, fill_value="extrapolate")
    fy = interp1d(distances, ypoints, kind='cubic', bounds_error=False, fill_value="extrapolate")

    # Step 3: Create equally spaced distances
    num_points = N  # how many points you want
    equal_distances = np.linspace(0, distances[-1], num_points)

    # Step 4: Sample interpolated points
    x_uniform = fx(equal_distances)
    y_uniform = fy(equal_distances)
    uniform_points = np.stack((x_uniform, y_uniform), axis=-1)

    # Force start and end to exactly match input endpoints (optional but precise)
    x_uniform[0], y_uniform[0] = xpoints[0], ypoints[0]
    x_uniform[-1], y_uniform[-1] = xpoints[-1], ypoints[-1]

    return uniform_points


def interpcurve_uniform(N, xpoints, ypoints):
    # Stack into (N, 2) array
    points = np.stack((xpoints, ypoints), axis=1)

    # Compute arc-length parameter s
    d = np.hypot(*np.diff(points.T))
    s = np.insert(np.cumsum(d), 0, 0)

    # Create interpolation functions for x and y
    fx = interp1d(s, points[:, 0], kind='linear')
    fy = interp1d(s, points[:, 1], kind='linear')

    # Generate evenly spaced arc-length positions
    s_uniform = np.linspace(0, s[-1], N)

    # Interpolated coordinates
    x_new = fx(s_uniform)
    y_new = fy(s_uniform)

    # Force start and end to exactly match input endpoints (optional but precise)
    x_new[0], y_new[0] = xpoints[0], ypoints[0]
    x_new[-1], y_new[-1] = xpoints[-1], ypoints[-1]

    # Sort by x if needed to ensure monotonic order
    sorted_idx = np.argsort(x_new)
    x_sorted = x_new[sorted_idx]
    y_sorted = y_new[sorted_idx]

    return np.stack((x_sorted, y_sorted), axis=1)


def interpcurve_symmetric(N, xpoints, ypoints):
    points = np.stack((xpoints, ypoints), axis=-1)

    # Step 1: Find the base (lowest y)
    min_y = np.min(ypoints)
    base_indices = np.where(ypoints == min_y)[0]
    if len(base_indices) == 1:
        base_idx = base_indices[0]
    else:
        base_idx = base_indices[np.argmin(np.abs(xpoints[base_indices] - np.mean(xpoints[base_indices])))]

    # Step 2: Split into halves
    left_half = points[:base_idx + 1][::-1]  # includes base point at end
    right_half = points[base_idx + 1:]  # excludes base point

    # Step 3: Interpolate helper
    def interp_half(half, n):
        if len(half) < 2:  # can't interpolate
            return half, 0
        d = np.hypot(*np.diff(half.T))
        s = np.insert(np.cumsum(d), 0, 0)
        fx = interp1d(s, half[:, 0], kind='linear')
        fy = interp1d(s, half[:, 1], kind='linear')
        s_uniform = np.linspace(0, s[-1], n)
        return np.stack((fx(s_uniform), fy(s_uniform)), axis=-1), s[-1]

    # Step 4: Compute lengths for proportional sampling
    _, len_left = interp_half(left_half, 10)
    _, len_right = interp_half(right_half, 10)
    total_len = len_left + len_right
    n_left = int(np.floor(N * len_left / total_len))
    n_right = N - n_left

    # Step 5: Interpolate each half
    interp_left, _ = interp_half(left_half, n_left)
    interp_right, _ = interp_half(right_half, n_right)

    # Step 6: Combine and sort by x to ensure monotonic increasing x
    combined = np.vstack((interp_left, interp_right))
    combined_sorted = combined[np.argsort(combined[:, 0])]

    return combined_sorted


def interpcurve_uniform(N, xpoints, ypoints):
    # Stack into (N, 2) array
    points = np.stack((xpoints, ypoints), axis=1)

    # Compute arc-length parameter s
    d = np.hypot(*np.diff(points.T))
    s = np.insert(np.cumsum(d), 0, 0)

    # Create interpolation functions for x and y
    fx = interp1d(s, points[:, 0], kind='linear')
    fy = interp1d(s, points[:, 1], kind='linear')

    # Generate evenly spaced arc-length positions
    s_uniform = np.linspace(0, s[-1], N)

    # Interpolated coordinates
    x_new = fx(s_uniform)
    y_new = fy(s_uniform)

    # Sort by x if needed to ensure monotonic order
    sorted_idx = np.argsort(x_new)
    x_sorted = x_new[sorted_idx]
    y_sorted = y_new[sorted_idx]

    return np.stack((x_sorted, y_sorted), axis=1)


def interpcurve_middle_out(N, xpoints, ypoints):
    points = np.stack((xpoints, ypoints), axis=1)
    d = np.hypot(*np.diff(points.T))
    s = np.insert(np.cumsum(d), 0, 0)

    fx = interp1d(s, points[:, 0], kind='linear')
    fy = interp1d(s, points[:, 1], kind='linear')

    s_total = s[-1]
    half_N = N // 2

    # Symmetric sampling around the center of the arc length
    if N % 2 == 0:
        half_spacing = np.linspace(0, s_total / 2, half_N, endpoint=False)
        s_uniform = np.concatenate((
            s_total / 2 - half_spacing[::-1],
            s_total / 2 + half_spacing
        ))
    else:
        half_spacing = np.linspace(0, s_total / 2, half_N, endpoint=False)
        s_uniform = np.concatenate((
            s_total / 2 - half_spacing[::-1],
            [s_total / 2],
            s_total / 2 + half_spacing
        ))

    # Interpolate
    x_new = fx(s_uniform)
    y_new = fy(s_uniform)

    # Fix endpoints explicitly
    x_new[0] = xpoints[0]
    y_new[0] = ypoints[0]
    x_new[-1] = xpoints[-1]
    y_new[-1] = ypoints[-1]

    # Optional: sort to enforce monotonic x-direction
    sorted_idx = np.argsort(x_new)
    x_final = x_new[sorted_idx]
    y_final = y_new[sorted_idx]

    return np.stack((x_final, y_final), axis=1)


def WTA_bed_stresses(wp_boundary, mean_v, slope, centre_shift=0.0):
    """
    Compute bed shear stress along a channel wetted perimeter using the
    law of the wall to estimate velocity gradients.

    Parameters
    ----------
    wp_boundary : ndarray, shape (N, 2)
        Ordered (x, y) coordinates describing the wetted perimeter polygon
        from left bank → bed → right bank.
    mean_v : float
        Mean flow velocity (m/s).
    slope : float
        Channel bed slope (dimensionless).
    centre_shift : float, optional
        Horizontal shift of the velocity maximum location, between -1 and 1.
        -1 = left side, 0 = centre, 1 = right side.

    Returns
    -------
    bed_sstress : list of float
        Bed shear stress (Pa) evaluated at each point in wp_boundary.

    Notes
    -----
    - Computes hydraulic radius from polygon area / perimeter.
    - Uses Chezy-derived roughness length ``l0`` to apply the log law.
    - Determines the velocity core location via channel geometry and
      optional centre shifting.
    - Computes velocity gradients by ray-distance to velocity core.
    - Applies a symmetric correction to avoid numerical asymmetry.
    """

    # Ensure centre_shift is within bounds
    centre_shift = max(min(centre_shift, 1.0), -1.0)

    # Define surface centre of the channel
    x_min, x_max = wp_boundary[0, 0], wp_boundary[-1, 0]
    channel_width = x_max - x_min
    v_max_x = x_min + (1 + centre_shift) / 2 * channel_width
    v_max_y = np.max(wp_boundary[:, 1])
    v_max_pos = np.array([v_max_x, v_max_y])

    # Try to use bed centre if there's a clear minimum point in y (bed level)
    v_min_y = np.min(wp_boundary[:, 1])
    index = np.where(wp_boundary[:, 1] == v_min_y)

    if len(index[0]) > 1:
        central_index = len(index[0]) // 2
        x1 = wp_boundary[index[0][central_index - 1], 0]
        x2 = wp_boundary[index[0][central_index], 0]
        v_max_pos = np.array([(x1 + x2) / 2, v_max_y])
    else:
        v_max_pos = np.array([wp_boundary[index[0][0], 0], v_max_y])

    # Get surface y-level (maximum y)
    surface_y = np.max(wp_boundary[:, 1])
    min_y = np.min(wp_boundary[:, 1])
    low_points = wp_boundary[wp_boundary[:, 1] == min_y]
    bottom_center_x = np.mean(low_points[:, 0])

    # Final v_max_pos: vertically above channel bottom at surface
    v_max_pos = np.array([bottom_center_x, surface_y])
    v_max_pos[0] = x_min + (1 + centre_shift) / 2 * channel_width

    # Cross-sectional area and wetted perimeter
    A = polyarea(wp_boundary)
    P = perimeter(wp_boundary)
    R = A / P
    Pl = P / (len(wp_boundary) - 1)

    # Estimate roughness height using original formula (Chezy)
    l0 = 0.37 * A / P * np.exp(-39.36 / (2.5 * np.sqrt(9.81)))

    # Use Law of the Wall to compute max velocity from l0
    g = 9.81
    kappa = 0.41
    h = surface_y - min_y  # flow depth
    v_shear = math.sqrt(g * R * slope)

    # Guard against very small l0 or h/l0 < 1
    if h <= l0:
        raise ValueError("Flow depth must be greater than roughness length l0 for log law to apply.")

    v_max = (v_shear / kappa) * math.log(h / l0)

    # Compute ray lengths from each point to velocity max core
    ray_lengths = [math.dist(pt, v_max_pos) for pt in wp_boundary]

    # Use v_max and l0 to compute velocity gradient at each bed point
    bed_v_grad = [v_max / l0 * 1 / np.log(ray_len / l0) for ray_len in ray_lengths]

    # Trapezoidal integration to compute denominator
    denominator = sum(i * i * Pl for i in bed_v_grad[1:-1]) \
                  + bed_v_grad[0] ** 2 * Pl / 2 + bed_v_grad[-1] ** 2 * Pl / 2

    # numerical issues arise, only for no fluxes straight down case, when slight assymetry from du_dr happens
    # --- compare symmetric pairs and correct large mismatches ---
    threshold = 1e-9  # choose your own numerical tolerance
    to_compare = np.array(bed_v_grad, dtype=float)  # ensure it's an array
    n = len(to_compare)
    half = n // 2  # number of pairs

    for i in range(half):
        left = to_compare[i]
        right = to_compare[-(i + 1)]
        diff = abs(left - right)

        if diff < threshold:
            mean_val = 0.5 * (left + right)
            to_compare[i] = mean_val
            to_compare[-(i + 1)] = mean_val

    # update du_dr with corrected values
    bed_v_grad = to_compare

    # Shear stress at each point
    bed_sstress = [
        (1000 * g * slope * A * grad ** 2) / denominator
        for grad in bed_v_grad
    ]

    return bed_sstress








def seg_intersect(a1, a2, b1, b2):
    """
    Determine whether two 2D line segments intersect.

    Parameters
    ----------
    a1, a2 : array-like of float, shape (2,)
        Endpoints of the first segment.
    b1, b2 : array-like of float, shape (2,)
        Endpoints of the second segment.

    Returns
    -------
    bool
        True if the two segments intersect (including collinear overlap),
        False otherwise.

    Notes
    -----
    Uses orientation tests and on-segment checks for all general and special
    geometric cases.
    """

    def orient(p, q, r):
        # Cross product (q - p) x (r - p)
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        # check if q lies on segment pr
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    # General case
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # Special cases (collinear & overlapping)
    if o1 == 0 and on_segment(a1, b1, a2): return True
    if o2 == 0 and on_segment(a1, b2, a2): return True
    if o3 == 0 and on_segment(b1, a1, b2): return True
    if o4 == 0 and on_segment(b1, a2, b2): return True

    return False


def is_shaded(object_boundary, time, latitude, longitude, channel_orientation):
    """
    Determine shading of each point on a channel boundary polygon
    based on solar position.

    Parameters
    ----------
    object_boundary : list
        List of objects with a `.position` attribute (2-element array)
        and a `.isShaded` boolean attribute to be updated.
    time : datetime-like
        Timestamp used to compute solar angles.
    latitude : float
        Latitude in radians.
    longitude : float
        Longitude in radians.
    channel_orientation : float
        Channel azimuth orientation in radians (0 = north).

    Returns
    -------
    None
        The function updates `object_boundary[i].isShaded` in place.

    Notes
    -----
    - Computes declination, hour angle, solar altitude and azimuth.
    - Determines whether the sun is to left or right of channel axis.
    - A “sun point” outside the domain is created and rays are cast
      from each boundary point.
    - If the ray to the sun intersects the channel opening line, the
      point receives sunlight; otherwise it is shaded.
    - Endpoints inherit shading status from their neighboring points.
    """

    # get day of the year
    day_no = time.timetuple().tm_yday
    # get local solar time
    solar_time = local_solar_time(long=longitude, dn=day_no, lt=time.hour)

    '''
    defining needed solar angles
    '''
    # declination
    dec = declination(day_no)
    # hour angle
    ha = hour_angle(solar_time)
    # sun altitude
    beta = solar_alt(latitude, ha, dec)
    # if sun is below horizon return all shade
    if beta < 0:
        # print('below horizon trigger')
        for i in range(len(object_boundary)):
            object_boundary[i].isShaded = True
        return
    # azimuth
    az = azimuth(latitude, dec, beta)
    # condition to make azimuth go to 2 pi
    if 12 < solar_time < 24:
        az = 2 * np.pi - az

    '''
    deciding if shade is going to be cast on lhs or rhs of channel
    '''
    # print(np.rad2deg(az))
    if 0 < az < channel_orientation:
        side = 'r'
    elif np.pi + channel_orientation < az <= 2 * np.pi:
        side = 'r'
    # cases to catch if sun is aligned with the channel, i.e parallel
    elif az == channel_orientation:
        for i in range(len(object_boundary)):
            object_boundary[i].isShaded = True
        return
    elif az == channel_orientation + np.pi:
        for i in range(len(object_boundary)):
            object_boundary[i].isShaded = True
    else:
        side = 'l'

    '''
    Get top of channel line 
    '''
    # use list comprehension to get position of all boundary points
    b_pos = [x.position for x in object_boundary]

    # define channel surface points
    top_l = b_pos[0]
    top_r = b_pos[-1]

    '''
    Define point which represents sun
    '''
    # side is which side the shade will be cast on
    if side == 'r':
        # get slope of line from sun to top point
        slope = np.tan(beta)

        sun_x = top_l[0] - 1
        sun_y = -slope * sun_x + top_l[1] + top_l[0] * slope

    if side == 'l':
        # get slope of line from sun to top point
        slope = np.tan(beta)

        sun_x = top_r[0] + 1
        sun_y = slope * sun_x + top_r[1] - top_l[0] * slope

    sun = [sun_x, sun_y]

    '''
    run through all positions and see if they intersect the channel opening
    '''

    for i in range(1, len(b_pos) - 1):
        # define 4 points, first two surface, next two ray connecting sun to point on channel
        a1 = np.array(top_l)
        a2 = np.array(top_r)
        b1 = np.array(b_pos[i])
        b2 = np.array(sun)
        # plt.plot([a1[0], a2[0]], [a1[1], a2[1]])
        # plt.plot([b1[0], b2[0]], [b1[1], b2[1]])
        # check if they intersect
        # if they intersect then point cna see sun so no shade
        # true false logic looks incorrect but works?
        if seg_intersect(a1, a2, b1, b2) == True:
            object_boundary[i].isShaded = False
        elif seg_intersect(a1, a2, b1, b2) == False:
            object_boundary[i].isShaded = True

    # to make sure top points are shaded correctly. If point below it is shaded then it too must be shaded
    if object_boundary[1].isShaded == True:
        object_boundary[0].isShaded = True
    else:
        object_boundary[0].isShaded = False

    if object_boundary[-2].isShaded == True:
        object_boundary[-1].isShaded = True
    else:
        object_boundary[-1].isShaded = False

    return


def surface_azimuths(points, channel_direction):
    """
    Automatically assigns LHS and RHS azimuths for a U-shaped boundary
    based on channel direction and point layout.

    Parameters:
    - points: (N, 2) array of boundary (x, y) coordinates
    - channel_direction: direction the U is pointing in radians (0 = North)

    Returns:
    - azimuths: array of surface azimuths (radians)
    - lhs_indices: indices of LHS points
    - rhs_indices: indices of RHS points
    """
    # Get the axis perpendicular to channel direction (i.e., left-right)
    nx = np.cos(channel_direction + np.pi / 2)
    ny = np.sin(channel_direction + np.pi / 2)

    # Perpendicular axis to channel direction (left-right vector)
    nx = -np.sin(channel_direction)
    ny = np.cos(channel_direction)

    # Project all points onto the normal axis
    projections = points[:, 0] * nx + points[:, 1] * ny

    # Use the median projection to divide LHS and RHS
    median_proj = np.median(projections)
    lhs_indices = np.where(projections < median_proj)[0]
    rhs_indices = np.where(projections > median_proj)[0]

    # Assign azimuths
    azimuths = np.zeros(len(points))
    azimuth_lhs = (np.pi / 2 + channel_direction) % (2 * np.pi)
    azimuth_rhs = (3 * np.pi / 2 + channel_direction) % (2 * np.pi)
    # azimuths[lhs_indices] = azimuth_lhs
    # azimuths[rhs_indices] = azimuth_rhs
    azimuths[0:len(lhs_indices)] = azimuth_lhs
    azimuths[len(lhs_indices):] = azimuth_rhs

    return azimuths


def is_undercut(object_boundary):
    '''
    Get positions
    '''
    # use list comprehension to get position of all boundary points
    b_pos = [x.position for x in object_boundary]

    '''
    run through all positions and see if they intersect the channel in a vertical direction
    '''

    for i in range(len(b_pos)):

        # define vertical point much higher than the point to be considered
        vert = [b_pos[i][0], b_pos[i][1] + 100000]
        # could just set to helfway and centre, this isn't exactly halfwway but close
        # vert = [(b_pos[0][0] + b_pos[-1][0]) / 2, b_pos[0][1]]
        # define 4 points, first two the boundary point and the much higher vertical point
        a1 = np.array(b_pos[i])
        a2 = np.array(vert)
        # then loop through all line segments making up the boundary, lenb - 1
        for j in range(len(b_pos) - 1):
            b1 = np.array(b_pos[j])
            b2 = np.array(b_pos[j + 1])
            # print('a1,a2,b1,b2', a1, a2, b1, b2)

            if seg_intersect(a1, a2, b1, b2) == True:
                object_boundary[i].isUndercut = True
                # break out of the loop at this point
                break

            elif seg_intersect(a1, a2, b1, b2) == False:
                object_boundary[i].isUndercut = False

    return



@jit(nopython=True)
def intersect_loop(b_pos, a1, a2):
    # --- find ymax and anchor roof points ---
    ymax = -1e12
    for k in range(len(b_pos)):
        if b_pos[k][1] > ymax:
            ymax = b_pos[k][1]

    xmin_top = -1e12
    xmax_top = 1e12
    for k in range(len(b_pos)):
        if b_pos[k][1] == ymax:
            if b_pos[k][0] < xmin_top:
                xmin_top = b_pos[k][0]
            if b_pos[k][0] > xmax_top:
                xmax_top = b_pos[k][0]

    left_roof = (xmin_top, ymax)
    right_roof = (xmax_top, ymax)

    # --- build extended boundary with roof points ---
    n = len(b_pos)
    b_ext = np.empty((n + 2, 2))
    b_ext[0] = left_roof
    for i in range(n):
        b_ext[i + 1] = b_pos[i]
    b_ext[-1] = right_roof

    # --- segment intersection test over extended boundary ---
    for j in range(len(b_ext) - 1):
        b1 = b_ext[j]
        b2 = b_ext[j + 1]

        det_abc = (a1[0] - b1[0]) * (a2[1] - b1[1]) - (a1[1] - b1[1]) * (a2[0] - b1[0])
        det_abd = (a1[0] - b2[0]) * (a2[1] - b2[1]) - (a1[1] - b2[1]) * (a2[0] - b2[0])
        det_cda = (b1[0] - a1[0]) * (b2[1] - a1[1]) - (b1[1] - a1[1]) * (b2[0] - a1[0])
        det_cdb = (b1[0] - a2[0]) * (b2[1] - a2[1]) - (b1[1] - a2[1]) * (b2[0] - a2[0])

        if det_abc * det_abd < 0 and det_cda * det_cdb < 0:
            return True  # intersection or undercut found

    return False


@jit(nopython=True, parallel=True)
def is_undercut_numba(b_pos):
    '''
    Get positions
    '''

    '''
    run through all positions and see if they intersect the channel in a vertical direction
    '''
    isUndercut = np.zeros(len(b_pos))
    for i in prange(len(b_pos)):
        # define vertical point much higher than the point to be considered
        vert = np.array([b_pos[i][0], b_pos[i][1] + 100000])
        # define 4 points, first two the boundary point and the much higher vertical point
        a1 = b_pos[i]
        a2 = vert
        isUndercut[i] = intersect_loop(b_pos, a1, a2)

    return isUndercut



# Function to calculate Euclidean distance between two points
def calculate_distance(p1, p2):
    """
    Compute the Euclidean distance between two 2-D points.

    Parameters
    ----------
    p1 : array-like of float
        First point as (x, y).
    p2 : array-like of float
        Second point as (x, y).

    Returns
    -------
    float
        Euclidean distance between p1 and p2.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)



def ensure_max_distance(dictionary, threshold=0.05):
    """
    Ensure that spacing between consecutive boundary points does not exceed a maximum distance.
    If gaps exceed `threshold`, new intermediate ObjectPoint instances are inserted.

    Parameters
    ----------
    dictionary : list of ObjectBoundary.ObjectPoint
        Ordered list of boundary points containing .position, .albedo, .columnTempProfile,
        .columnInitialTemp, .columnDepth, and .columnPointSpacing attributes.
    threshold : float, optional
        Maximum allowed spacing between adjacent points (default = 0.05).

    Returns
    -------
    list of ObjectBoundary.ObjectPoint
        New list containing the original points plus any inserted intermediate points.
        Ensures:
            - No segment exceeds `threshold`
            - If points were added, total number of points is even (middle point removed if necessary)

    Notes
    -----
    - New points inherit albedo and thermal properties from surrounding points.
    - Inserted points are positioned evenly along the segment.
    - When inserting multiple new points, the function redistributes them symmetrically
      from the midpoint outward.
    - If any points were added, the function ensures an **even total number** of points
      because other parts of the model rely on symmetry about the channel centreline.
    """
    coords = [x.position for x in dictionary]
    new_dictionary = [dictionary[0]]  # Start with the first point

    # add term to see if points added
    added_trigger = False
    for i in range(1, len(coords)):
        prev_point = coords[i - 1]
        curr_point = coords[i]

        dist = calculate_distance(prev_point, curr_point)

        if dist > threshold:
            added_trigger = True
            # Calculate number of segments needed
            num_segments = math.ceil(dist / threshold)
            num_insert = num_segments - 1

            # Insert points from the center outward
            mid_index = num_insert // 2
            inserts = []

            for j in range(1, num_segments):
                t = j / num_segments
                x = prev_point[0] + (curr_point[0] - prev_point[0]) * t
                y = prev_point[1] + (curr_point[1] - prev_point[1]) * t
                new_obj = ObjectBoundary.ObjectPoint([x, y],
                                                     albedo=dictionary[i - 1].albedo,
                                                     columnInitialTemp=271.25,
                                                     columnDepth=3,
                                                     columnPointSpacing=dictionary[0].columnPointSpacing)

                if i < len(coords) - 1:
                    new_obj.columnTempProfile = (dictionary[i - 1].columnTempProfile +
                                                 dictionary[i + 1].columnTempProfile) / 2
                inserts.append(new_obj)

            # Reorder inserts from the middle outward (symmetrical)
            reordered = []
            left = mid_index - 1
            right = mid_index
            while left >= 0 or right < len(inserts):
                if right < len(inserts):
                    reordered.append(inserts[right])
                    right += 1
                if left >= 0:
                    reordered.insert(0, inserts[left])
                    left -= 1

            new_dictionary.extend(reordered)

        # Add the end point
        new_dictionary.append(dictionary[i])

    # --- Ensure even number of points ---, only trigger if points are added
    if added_trigger:
        n_points = len(new_dictionary)
        if n_points % 2 != 0:
            centre_index = n_points // 2
            del new_dictionary[centre_index]

    return new_dictionary



def smooth_boundary(boundary, sigma=2):
    """
    Smooths a 2D boundary using Gaussian filtering.

    Parameters:
    - boundary: np.ndarray of shape (N, 2), the boundary points.
    - sigma: float, the standard deviation for Gaussian kernel.

    Returns:
    - smoothed_boundary: np.ndarray of shape (N, 2), the smoothed boundary.
    """
    boundary = np.asarray(boundary)
    x_smooth = gaussian_filter1d(boundary[:, 0], sigma=sigma, mode='reflect')
    y_smooth = gaussian_filter1d(boundary[:, 1], sigma=sigma, mode='reflect')

    return np.stack((x_smooth, y_smooth), axis=-1)


def distances_between_adjacent_points(points):
    """
    Calculate the Euclidean distances between adjacent 2D points.

    Parameters:
    points (array-like): A list or NumPy array of 2D points, e.g., [[x1, y1], [x2, y2], ..., [xn, yn]]

    Returns:
    list: A list of distances between each pair of adjacent points.
    """
    points = np.array(points)
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return distances.tolist()


def T_inf(V, wall_shear):
    rho_w = 1000
    cw = 4184
    return 273.15 + (V**2 + V*np.sqrt(wall_shear/rho_w) * 110.45)/cw
