import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import numpy as np
import sys
# append location of MCEM_files
sys.path.append(r"C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to upload\MCEM_files")
from MCEM_files import Cfunctions, MCEM_1_1, IceWallMeltFunctionGhost, ObjectBoundary
import datetime as dt
import pandas as pd

'''
Load data being used for reference, change path accordingly
'''
path = r'C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to upload\WeatherData\KAN_L.csv'


'''
Run the model 
'''
# # decide how many hours to run for
# hours_to_run = 500
# # define initial resolution of the boundary
# number_b_points = 200
# # define the initial boundary
# initial_boundary = Cfunctions.two_sided_u(x_left=49.4,
#                                x_right=50.6,
#                                x_center=50,
#                                z_bottom=499.7,
#                                z_top=500,
#                                res=number_b_points
#                                )
# # define a start date for the model
#
# start_date = dt.datetime(2024, 5, 29, 0, 0, 0)
# # define a start time for benchmarking
# start = time.process_time()
#
#
# # run the MCEM
# b_moved = MCEM_1_1.MCEM(b0=initial_boundary, max_water_d=0.2, beta=0.028, water_t=273.25,
#                                 ice_albedo=0.27,
#                                 latitude=np.deg2rad(67), longitude=np.deg2rad(-50),
#                                 channel_orientation=np.pi / 2,
#                                 start_date=start_date,
#                                 runtime_hours=hours_to_run, data_filename=path
#                                 , save_hrs=10,
#                                 ice_col_spacing=0.01,
#                                 rk_loop_tstep=20, boundary_spacing=0.01, gfilter_ls=0.003, velocity_shift=0,
#                                 diurnal=False)
#
# end = time.time()
# print(time.process_time() - start, 's')
# print(b_moved['meta data']['runtime'])

# uncomment to save the data
# np.save('../tester.npy', b_moved)

# if don't want to run the model can load in from previous runs
data = np.load(r'C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to upload\PreRunData\RUN0.npy', allow_pickle=True)
# use to get data is usable format
b_moved = data.item()

'''
Plotting results, a dictionary with each hour having associated boundary
'''
# define figure

fig = plt.figure(figsize=(8, 5), dpi=400)
ax = fig.add_subplot(111)

# select which hours wanting to plot
h_wanted = list(b_moved['data'].keys())


# create colour cycle based on number of values wanting to plot
colour_cycle = cm.rainbow(np.linspace(0, 1, len(h_wanted)))
for i, c in zip((h_wanted), colour_cycle):
    # define various indicis of wanted variables
    shade_idx = [idx for idx, x in enumerate(b_moved['data'][i]) if x.isShaded == True]
    wet_idx = [idx for idx, x in enumerate(b_moved['data'][i]) if x.isWet == True]
    dry_idx = [idx for idx, x in enumerate(b_moved['data'][i]) if x.isWet == False]
    undercut_idx = [idx for idx, x in enumerate(b_moved['data'][i]) if x.isUndercut == True]
    end_temps = [x.columnTempProfile[-1] for x in b_moved['data'][i]]
    # get position from object dictionary
    newb = np.array([x.position for x in b_moved['data'][i]])
    x1, y1 = newb.T

    ax.scatter(x1, y1, label='hour {}'.format(i), color=c, marker='.')
    # plotting location of maximum surface energy
    # surface_energy = np.asarray([x.surfaceEnergy for x in b_moved['data'][i]])
    # max_surface_energy_idx = np.where(surface_energy == max(surface_energy))
    # ax.scatter(x1[max_surface_energy_idx], y1[max_surface_energy_idx], color='red', marker='*')


plt.ylabel('z (m)')
plt.xlabel('x (m)')
plt.grid()
plt.tight_layout()
plt.show()


'''
Extracting metrics such as incision, drift, WP width and DP width
'''
data = b_moved['data']
# extracting variables
# get position from object dictionary at time 0
b_pos0 = np.array([x.position for x in b_moved['data'][0]])
# define specific x and y
x0, y0 = b_pos0.T

# get position at end time
b_pos1 = np.array([x.position for x in b_moved['data'][max(h_wanted)]])

x1, y1 = b_pos1.T
'''
Total channel incision
'''
lowest_y0 = np.min(y0)
lowest_y1 = np.min(y1)
total_incision = lowest_y0 - lowest_y1

'''
Drift distance
'''
# find index with lowest y, which will be on wp
index0 = np.where(b_pos0[:, 1] == lowest_y0)
central_x0_max = float(np.max(b_pos0[index0, 0]))
central_x0_min = float(np.min(b_pos0[index0, 0]))
central_x0 = (central_x0_max + central_x0_min) / 2
print(central_x0)

index1 = np.where(b_pos1[:, 1] == lowest_y1)
central_x1 = float(np.min(b_pos1[index1, 0]))
total_drift = central_x1 - central_x0

'''
WP width
'''
wet_idx = [idx for idx, x in enumerate(data[max(h_wanted)]) if x.isWet == True]
wet_pos = b_pos1[wet_idx]
WP_width = wet_pos[-1, 0] - wet_pos[0, 0]

'''
DP width
'''
DP_width = b_pos1[-1, 0] - b_pos1[0, 0]

# print('RUN{}'.format(j))
print('Total incision', total_incision)
print('Total drift', total_drift)
print('WP width', WP_width)
print('DP width', DP_width)
print('Run hours', max(h_wanted))
