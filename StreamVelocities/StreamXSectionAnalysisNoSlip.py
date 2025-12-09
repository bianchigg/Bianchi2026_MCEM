import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MCEM_files import Cfunctions
import scipy
import matplotlib as mpl
import math

'''
import data and clean
'''
# read in data from Greenland data folder
green_data = pd.read_excel(
    r'Stream_measurements_noslip.xlsx',
    header=0, sheet_name=0)
# data read in is positions with an associated velocity. If the velocity is zero, then it is a channel boundary with
# a no slip condition.

# convert time to datetime format
green_data['time'] = pd.to_datetime(green_data['time'])
# get a list of unique dates
uni_dates = green_data['time'].unique()

'''
Plotting the data
'''
# set up subplot to plot the data
# define axis to subplot data
fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8, 12), layout='compressed')
# get indices
ax_ind = np.indices(np.shape(ax))
# Transpose the indices to get them as a list of (row, column) pairs
# ax_ind_list = list(zip(ax_ind[0].flatten(), ax_ind[1].flatten()))


# Loop around all unique dates and get data for each
for i in range(len(uni_dates)):

    # get data at specific time
    green_t = green_data[green_data['time'] == uni_dates[i]]
    green_t.reset_index()
    '''
    data cleaning
    change y positions so they are correct, currently measured velocities are at 0.135 and 0.06m. This needs to be
    changed so that it is these values above the bottom boundary
    '''
    # find boundary y. All points on boundary have 0 velocity due to no slip condition
    boundary = green_t[green_t['v (m/s)'] == 0]

    # loop around all points in df
    for j in range(len(green_t)):
        # only alter the points not on the boundary
        if green_t['v (m/s)'].iloc[j] > 0:
            # find boundary point with corresponding x value
            max_d_col = boundary[boundary['vx_pos (m)'] == green_t['vx_pos (m)'].iloc[j]]
            max_d = max_d_col['vy_pos (m)']
            # get the relative depth
            rel_d = - max_d + np.copy(green_t['vy_pos (m)'].iloc[j])
            # change the df so it is the relative depth
            green_t['vy_pos (m)'].iloc[j] = np.copy(rel_d)
        # then change boundary points so they are negative as well
        if green_t['v (m/s)'].iloc[j] == 0:
            green_t['vy_pos (m)'].iloc[j] = - np.copy(green_t['vy_pos (m)'].iloc[j])

    # plot the velocities
    for j in range(len(green_t)):
        # only plot if velocity is above 0
        if green_t['v (m/s)'].iloc[j] > 0:
            pass
            # ax[ax_ind_list[i][0], ax_ind_list[i][1]].text(green_t['vx_pos (m)'].iloc[j],
            #                                               green_t['vy_pos (m)'].iloc[j],
            #                                               str(green_t['v (m/s)'].iloc[j]))

    # creating a grid to use for imshow
    # original irregular points
    xo = green_t['vx_pos (m)']
    yo = green_t['vy_pos (m)']
    zo = green_t['v (m/s)']

    # Size of regular grid
    ny, nx = 1000, 1000
    # Generate a regular grid to interpolate the data.
    xi = np.linspace(min(xo), max(xo), nx)
    yi = np.linspace(min(yo), max(yo), ny)
    xi, yi = np.meshgrid(xi, yi)
    # interpolate the irregular data onto regular grid of size ny*nx. Normal interpolation is linear, can also cubic
    # or nearest
    zi = scipy.interpolate.griddata((xo, yo), zo, (xi, yi), method='cubic')

    '''
    some zi become negative, and some are above the maximum observed velocity
    add condition to bound between 0 and max observed velocity
    '''
    for (p, q), value in np.ndenumerate(zi):
        # if nan keep nan
        if math.isnan(value):
            pass
        # if value is less than 0
        elif value < 0:
            # set to 0
            zi[p, q] = 0
        # if value is greater than max observed
        elif value > 20 / 17 * max(zo):
            zi[i, j] = 20 / 17 * max(zo)

    '''
    Finding centroid of the channels
    '''
    # need to find average of x coords of each channel and average y coords of each
    # these averages only need to be done where zi is defined i.e not nan
    # set counters up for x and y vals
    no_x_vals = 0
    x_tot = 0
    no_y_vals = 0
    y_tot = 0
    for (p, q), value in np.ndenumerate(zi):
        if math.isnan(value):
            pass
        else:
            # add x values to total
            x_tot += xi[p, q]
            # increase x counter
            no_x_vals += 1

            y_tot += yi[p, q]
            no_y_vals += 1

    centroid = [x_tot / no_x_vals, y_tot / no_y_vals]

    '''
    colour bar creating
    '''
    # create custom colormap for discrete colorbar,
    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    # define colourmap
    cmap = plt.cm.plasma
    # extract colours
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # make first entry grey
    # cmaplist[0] = (0.5, 0.5, 0.5, 1.0)
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0, 1, 21)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    '''
    plotting
    '''

    # # plot the data and add colourbar
    # img = ax[ax_ind_list[i][0], ax_ind_list[i][1]].pcolormesh(xi, yi, zi, cmap = cmap, norm = norm)
    # # add colorbar on teh axis of the plot
    # plt.colorbar(img, ax=ax[ax_ind_list[i][0], ax_ind_list[i][1]])
    # # plot the centroid
    # ax[ax_ind_list[i][0], ax_ind_list[i][1]].scatter(centroid[0], centroid[1], marker='*', s=120, c = 'k')
    # # set the heading to the time
    # ax[ax_ind_list[i][0], ax_ind_list[i][1]].set_title('{}'.format(green_t['time'].iloc[0]))

    # plot the data and add colourbar
    img = ax[i].pcolormesh(xi, yi, zi, cmap=cmap, norm=norm)
    # plot the centroid
    ax[i].scatter(centroid[0], centroid[1], marker='*', s=120, c='k')
    # set the heading to the time
    ax[i].set_title('{}'.format(green_t['time'].iloc[0]))

    yticks = ax[i].get_yticks()
    ax[i].set_yticklabels([f"{abs(y):.2f}" for y in yticks])
ax[i].set_xlabel('Distance from right bank (m)')
# add colour bar for all plots
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

cbar = fig.colorbar(img, ax=ax.ravel().tolist())
cbar.set_label('Velocity (m/s)')
fig.supylabel('Depth (m)')
# fig.tight_layout()
# save option
# plt.savefig(r'filename.png', dpi=300)
plt.show()

'''
Plotting all on individual figures
'''
time_list = []
gm_coeff_list = []
chezy_coeff_list = []
dw_coeff_list = []
water_d_list = []
area_list = []
wp_list = []
Q6_list = []
sixth_d_v_list = []
max_d_list = []
for i in range(len(uni_dates)):
    time_list.append(uni_dates[i])
    # get data at specific time
    green_t = green_data[green_data['time'] == uni_dates[i]]
    green_t.reset_index()
    '''
    data cleaning
    change y positions so they are correct, currently measured velocities are at 0.135 and 0.06m. This needs to be
    changed so that it is these values above the bottom boundary
    '''
    # find boundary y. All points on boundary have 0 velocity due to no slip condition
    boundary = green_t[green_t['v (m/s)'] == 0]




    '''
    get area and wp for each 
    '''
    x_pos = boundary['vx_pos (m)']
    y_pos = boundary['vy_pos (m)']
    # join arrays
    b = np.vstack((x_pos, y_pos))
    b = np.transpose(b)
    print(i)
    area = Cfunctions.polyarea(b)
    area_list.append(area)
    print('area = ', area)
    wp = Cfunctions.perimeter(b)
    wp_list.append(wp)
    print('wp = ', wp)
    meanv = np.nanmean(zi)
    print('meanv = ', meanv)
    Qv = area * meanv
    print('Qv = ', Qv)
    # trying to find max velocity at 0.6 of the depth. Find 0.6 of depth, get all values at that point, get max
    # search the meshgrid at for max value, i.e max depth. use min as values are negative but y_pos from boundary is
    # positive
    sixth_d = 0.6 * min(-y_pos)
    # find the value which is closest to this in yi
    closest = yi.flat[np.abs(yi - sixth_d).argmin()]
    # find the points in yi which are at this depth. just get single y value, thats what [0][0] does, choose y's then
    # select first one. [1][0] would select x
    sixth_d_idx = np.where(yi == closest)[0][0]
    # find max v at that index in z array
    sixth_d_v = np.nanmax(zi[sixth_d_idx])
    sixth_d_v_list.append(sixth_d_v)
    print('610v = ', sixth_d_v)
    Q6 = area * sixth_d_v
    Q6_list.append(Q6)
    print('Q6 = ', Q6)
    # derived from QGIS analysis
    chan_slope = 3.05 * 10 ** (-3)
    chan_slope = 0.135

    # gm coeff with both water depth (should be mean wd) and hydraulic radius
    gm_coeff = (area / wp) ** (2 / 3) * chan_slope ** (1 / 2) / sixth_d_v
    gm_coeff_list.append(gm_coeff)
    print('gm')
    print(gm_coeff)


    # chezy
    chezy_coeff = sixth_d_v / ((area / wp * chan_slope) ** (0.5))
    chezy_coeff_list.append(chezy_coeff)
    print('chezy')
    print(chezy_coeff)


    # dw
    dw_coeff = 8 * 9.81 * chan_slope * area / wp / sixth_d_v ** 2
    dw_coeff_list.append(chezy_coeff)
    print('dw')
    print(dw_coeff)

    water_d_list.append(-min(green_t['vy_pos (m)']))




# creat dictionary with all the list
data_dict = {'time': time_list,
             'wp': wp_list,
             'area': area_list,
             'sixth_d_v': sixth_d_v_list,
             'Q6': Q6_list,
             'gm_coeff': gm_coeff_list,
             'chezy_coeff': chezy_coeff_list,
             'dw_coeff': dw_coeff_list,
             'water_d': water_d_list}
df = pd.DataFrame(data_dict)
print(df)
# saving routines
# df.to_csv('StreamXSectionVals.csv', index=False)
# df.to_excel('StreamXSectionVals.xlsx', index=False)
# df.to_latex('StreamXSectionVals.tex', index=False, label='tab: StreamXSectionData', columns=['time', 'wp',
#                                                                                              'area',
#                                                                                              'sixth_d_v', 'Q6',
#                                                                                              'gm_coeff',
#                                                                                              'chezy_coeff',
#                                                                                              'dw_coeff'])
# mean coefficients used in paper
gm_mean = np.mean(gm_coeff_list)
chezy_mean = np.mean(chezy_coeff_list)
dw_mean = np.mean(dw_coeff_list)
