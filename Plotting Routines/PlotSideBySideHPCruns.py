import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
import sys
# Change path accordingly
path2MCEM = r'C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to upload\MCEM_files'
sys.path.append(path2MCEM)
import ObjectBoundary

path2data = r'C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to upload\PreRunData'
# load data, must allow pickle true
data = np.load(path2data + '\RUN0.npy', allow_pickle=True)
# use to get data is usable format not weird array thing
data = data.item()
if len(data) == 2:
    data = data['data']

datar = np.load(path2data+'\RUN_diurnal.npy', allow_pickle=True)
# use to get data is usable format not weird array thing
datar = datar.item()
if len(datar) == 2:
    datar = datar['data']

# get the dictionary keys corresponding to the hours
h_wanted = list(data.keys())
# create colour cycle based on number of values wanting to plot
colour_cycle = cm.brg(np.linspace(0, 1, len(h_wanted)))

# Time conversion: hours to days
h_days = [h / 24.0 for h in h_wanted]

# Color mapping
cmap = cm.winter
# cmap = cm.brg
norm = Normalize(vmin=min(h_days), vmax=max(h_days))
colour_cycle = cmap(norm(h_days))

# Set global font size
plt.rcParams.update({'font.size': 20})
# Plot with constrained layout to avoid warning tantrums
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 14), sharex=True, sharey=True, constrained_layout=True)

for i, c in zip((h_wanted), colour_cycle):
    # get position from object dictionary
    b_pos = np.array([x.position for x in data[i]])
    # define specific x and y
    x1, y1 = b_pos.T
    # get WP index
    wet_idx = [idx for idx, x in enumerate(data[i]) if x.isWet == True]
    # ax[0].scatter(x1[wet_idx], y1[wet_idx], color='blue')
    ax[0].scatter(x1, y1, label='hour {}'.format(i), color=c, marker='.')
    ax[0].fill_between(x1[wet_idx], y1[wet_idx], max(y1[wet_idx]), color='blue', alpha=.1)
    ax[0].set_title('Constant Water Level')
    ax[0].grid()
    ax[0].set_xlabel('x (m)')

    b_posr = np.array([x.position for x in datar[i]])
    xr, yr = b_posr.T
    undercut_idx = [idx for idx, x in enumerate(datar[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(datar[i]) if x.isWet == True]
    ax[1].fill_between(xr[wet_idx], yr[wet_idx], max(yr[wet_idx]), color='blue', alpha=.1)
    ax[1].scatter(xr, yr, label='hour {}'.format(i), color=c, marker='.')
    ax[1].set_title('Diurnal Water Level')
    ax[1].set_xlabel('x (m)')
    ax[1].grid()

# Create colorbar with flipped direction and day-based ticks
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm,
                    ax=ax,
                    location='right',
                    shrink=1,
                    label='Time (days)')

# Flip colorbar (low at bottom)
cbar.ax.invert_yaxis()

# Format ticks to 1 decimal place for days
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# fig.supxlabel("x (m)")
fig.supylabel("z (m)")
# custom_xlim = (46, 54)
# custom_ylim = (494, 500.05)
#
# # Setting the values for all axes.
# plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
# save as dexired
#plt.savefig(r"filename here", dpi=600)
plt.show()

'''
For runs without shortwave radiaition
'''
# load data, must allow pickle true
data = np.load(path2data + '\RUN_constant_lw.npy', allow_pickle=True)
# use to get data is usable format not weird array thing
data = data.item()
if len(data) == 2:
    data = data['data']

datar = np.load(path2data + '\RUN_diurnal_constant_lw.npy',
                allow_pickle=True)
# use to get data is usable format not weird array thing
datar = datar.item()
if len(datar) == 2:
    datar = datar['data']

h_wanted = list(data.keys())
# create colour cycle based on number of values wanting to plot
colour_cycle = cm.brg(np.linspace(0, 1, len(h_wanted)))

# Time conversion: hours to days
h_days = [h / 24.0 for h in h_wanted]

# Color mapping
cmap = cm.winter
# cmap = cm.brg
norm = Normalize(vmin=min(h_days), vmax=max(h_days))
colour_cycle = cmap(norm(h_days))

# Set global font size
plt.rcParams.update({'font.size': 20})
# Plot with constrained layout to avoid warning tantrums
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 14), sharex=True, sharey=True, constrained_layout=True)

for i, c in zip((h_wanted), colour_cycle):
    # get position from object dictionary
    b_pos = np.array([x.position for x in data[i]])
    # define specific x and y
    x1, y1 = b_pos.T


    wet_idx = [idx for idx, x in enumerate(data[i]) if x.isWet == True]
    # ax[0].scatter(x1[wet_idx], y1[wet_idx], color='blue')
    ax[0].scatter(x1, y1, label='hour {}'.format(i), color=c, marker='.')
    ax[0].fill_between(x1[wet_idx], y1[wet_idx], max(y1[wet_idx]), color='blue', alpha=.1)
    ax[0].set_title('Constant Water Level')
    ax[0].grid()
    ax[0].set_xlabel('x (m)')

    b_posr = np.array([x.position for x in datar[i]])
    xr, yr = b_posr.T
    undercut_idx = [idx for idx, x in enumerate(datar[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(datar[i]) if x.isWet == True]
    ax[1].fill_between(xr[wet_idx], yr[wet_idx], max(yr[wet_idx]), color='blue', alpha=.1)
    ax[1].scatter(xr, yr, label='hour {}'.format(i), color=c, marker='.')
    ax[1].set_title('Diurnal Water Level')
    ax[1].set_xlabel('x (m)')
    ax[1].grid()

# Create colorbar with flipped direction and day-based ticks
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm,
                    ax=ax,
                    location='right',
                    shrink=1,
                    label='Time (days)')

# Flip colorbar (low at bottom)
cbar.ax.invert_yaxis()

# Format ticks to 1 decimal place for days
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# fig.supxlabel("x (m)")
fig.supylabel("z (m)")
# custom_xlim = (46, 54)
# custom_ylim = (494, 500.05)
#
# # Setting the values for all axes.
# plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
# save as desired
# plt.savefig(r"file name here", dpi=600)
plt.show()

'''
Diurnal water temperature
'''

# load data, must allow pickle true
data = np.load(path2data + '\RUN_diurnal_water_T.npy',
               allow_pickle=True)
# use to get data is usable format not weird array thing
data = data.item()
if len(data) == 2:
    data = data['data']

datar = np.load(path2data + '\RUN_diurnal_water_T_and_level.npy',
                allow_pickle=True)
# use to get data is usable format not weird array thing
datar = datar.item()
if len(datar) == 2:
    datar = datar['data']

# seperate into data and meta data
# meta_data = data['meta data']
# data = data['data']
# print(j, meta_data)
h_wanted = list(data.keys())

# create colour cycle based on number of values wanting to plot
colour_cycle = cm.brg(np.linspace(0, 1, len(h_wanted)))

# Time conversion: hours to days
h_days = [h / 24.0 for h in h_wanted]

# Color mapping
cmap = cm.winter
# cmap = cm.brg
norm = Normalize(vmin=min(h_days), vmax=max(h_days))
colour_cycle = cmap(norm(h_days))

# Set global font size
plt.rcParams.update({'font.size': 20})
# Plot with constrained layout to avoid warning tantrums
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 14), sharex=True, sharey=True, constrained_layout=True)

for i, c in zip((h_wanted), colour_cycle):
    # get position from object dictionary
    b_pos = np.array([x.position for x in data[i]])
    # define specific x and y
    x1, y1 = b_pos.T
    undercut_idx = [idx for idx, x in enumerate(data[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(data[i]) if x.isWet == True]
    # ax[0].scatter(x1[wet_idx], y1[wet_idx], color='blue')
    ax[0].scatter(x1, y1, label='hour {}'.format(i), color=c, marker='.')
    ax[0].fill_between(x1[wet_idx], y1[wet_idx], max(y1[wet_idx]), color='blue', alpha=.1)
    ax[0].set_title('Constant Water Level')
    ax[0].grid()
    ax[0].set_xlabel('x (m)')

    b_posr = np.array([x.position for x in datar[i]])
    xr, yr = b_posr.T
    undercut_idx = [idx for idx, x in enumerate(datar[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(datar[i]) if x.isWet == True]
    ax[1].fill_between(xr[wet_idx], yr[wet_idx], max(yr[wet_idx]), color='blue', alpha=.1)
    ax[1].scatter(xr, yr, label='hour {}'.format(i), color=c, marker='.')
    ax[1].set_title('Diurnal Water Level')
    ax[1].set_xlabel('x (m)')
    ax[1].grid()

# Create colorbar with flipped direction and day-based ticks
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm,
                    ax=ax,
                    location='right',
                    shrink=1,
                    label='Time (days)')

# Flip colorbar (low at bottom)
cbar.ax.invert_yaxis()

# Format ticks to 1 decimal place for days
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# fig.supxlabel("x (m)")
fig.supylabel("z (m)")
# custom_xlim = (46, 54)
# custom_ylim = (494, 500.05)
#
# # Setting the values for all axes.
# plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
# save as desired
# plt.savefig(r" filename ", dpi=600)
plt.show()


'''
Plotting no sw vs sw
'''

# load data, must allow pickle true
data = np.load(path2data + '\RUN_constant_lw.npy', allow_pickle=True)
# use to get data is usable format not weird array thing
data = data.item()
if len(data) == 2:
    data = data['data']

datar = np.load(path2data + '\RUN0.npy', allow_pickle=True)
# use to get data is usable format not weird array thing
datar = datar.item()
if len(datar) == 2:
    datar = datar['data']


h_wanted = list(data.keys())

# create colour cycle based on number of values wanting to plot
colour_cycle = cm.brg(np.linspace(0, 1, len(h_wanted)))

# Time conversion: hours to days
h_days = [h / 24.0 for h in h_wanted]

# Color mapping
cmap = cm.winter
# cmap = cm.brg
norm = Normalize(vmin=min(h_days), vmax=max(h_days))
colour_cycle = cmap(norm(h_days))

# Set global font size
plt.rcParams.update({'font.size': 20})
# Plot with constrained layout to avoid warning tantrums
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 14), sharex=True, sharey=True, constrained_layout=True)

for i, c in zip((h_wanted), colour_cycle):
    # get position from object dictionary
    b_pos = np.array([x.position for x in data[i]])
    # define specific x and y
    x1, y1 = b_pos.T
    undercut_idx = [idx for idx, x in enumerate(data[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(data[i]) if x.isWet == True]
    # ax[0].scatter(x1[wet_idx], y1[wet_idx], color='blue')
    ax[0].scatter(x1, y1, label='hour {}'.format(i), color=c, marker='.')
    ax[0].fill_between(x1[wet_idx], y1[wet_idx], max(y1[wet_idx]), color='blue', alpha=.1)
    ax[0].set_title('Without Solar Radiation')
    ax[0].grid()
    ax[0].set_xlabel('x (m)')

    b_posr = np.array([x.position for x in datar[i]])
    xr, yr = b_posr.T
    undercut_idx = [idx for idx, x in enumerate(datar[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(datar[i]) if x.isWet == True]
    ax[1].fill_between(xr[wet_idx], yr[wet_idx], max(yr[wet_idx]), color='blue', alpha=.1)
    ax[1].scatter(xr, yr, label='hour {}'.format(i), color=c, marker='.')
    ax[1].set_title('With Solar Radiation')
    ax[1].set_xlabel('x (m)')
    ax[1].grid()

# Create colorbar with flipped direction and day-based ticks
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm,
                    ax=ax,
                    location='right',
                    shrink=1,
                    label='Time (days)')

# Flip colorbar (low at bottom)
cbar.ax.invert_yaxis()

# Format ticks to 1 decimal place for days
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# fig.supxlabel("x (m)")
fig.supylabel("z (m)")
custom_xlim = (48.5, 51.50)
# custom_ylim = (494, 500.05)
#
# # Setting the values for all axes.
plt.setp(ax, xlim=custom_xlim)#, ylim=custom_ylim)
# save as desired
plt.savefig(r"C:\Users\ggmb1\PycharmProjects\MCEM\images\sw_vs_nosw.png", dpi=600)
plt.show()


'''
Plotting no sw vs sw diurnal water level
'''

# load data, must allow pickle true
data = np.load(path2data + '\RUN_diurnal_constant_lw.npy',
               allow_pickle=True)
# use to get data is usable format not weird array thing
data = data.item()
if len(data) == 2:
    data = data['data']

datar = np.load(path2data + '\RUN_diurnal.npy', allow_pickle=True)
# use to get data is usable format not weird array thing
datar = datar.item()
if len(datar) == 2:
    datar = datar['data']


h_wanted = list(data.keys())

# create colour cycle based on number of values wanting to plot
colour_cycle = cm.brg(np.linspace(0, 1, len(h_wanted)))

# Time conversion: hours to days
h_days = [h / 24.0 for h in h_wanted]

# Color mapping
cmap = cm.winter
# cmap = cm.brg
norm = Normalize(vmin=min(h_days), vmax=max(h_days))
colour_cycle = cmap(norm(h_days))

# Set global font size
plt.rcParams.update({'font.size': 20})
# Plot with constrained layout to avoid warning tantrums
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 14), sharex=True, sharey=True, constrained_layout=True)

for i, c in zip((h_wanted), colour_cycle):
    # get position from object dictionary
    b_pos = np.array([x.position for x in data[i]])
    # define specific x and y
    x1, y1 = b_pos.T
    undercut_idx = [idx for idx, x in enumerate(data[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(data[i]) if x.isWet == True]
    # ax[0].scatter(x1[wet_idx], y1[wet_idx], color='blue')
    ax[0].scatter(x1, y1, label='hour {}'.format(i), color=c, marker='.')
    ax[0].fill_between(x1[wet_idx], y1[wet_idx], max(y1[wet_idx]), color='blue', alpha=.1)
    ax[0].set_title('Without Solar Radiation')
    ax[0].grid()
    ax[0].set_xlabel('x (m)')

    b_posr = np.array([x.position for x in datar[i]])
    xr, yr = b_posr.T
    undercut_idx = [idx for idx, x in enumerate(datar[i]) if x.isUndercut == True]
    wet_idx = [idx for idx, x in enumerate(datar[i]) if x.isWet == True]
    ax[1].fill_between(xr[wet_idx], yr[wet_idx], max(yr[wet_idx]), color='blue', alpha=.1)
    ax[1].scatter(xr, yr, label='hour {}'.format(i), color=c, marker='.')
    ax[1].set_title('With Solar Radiation')
    ax[1].set_xlabel('x (m)')
    ax[1].grid()

# Create colorbar with flipped direction and day-based ticks
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm,
                    ax=ax,
                    location='right',
                    shrink=1,
                    label='Time (days)')

# Flip colorbar (low at bottom)
cbar.ax.invert_yaxis()

# Format ticks to 1 decimal place for days
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# fig.supxlabel("x (m)")
fig.supylabel("z (m)")
custom_xlim = (48.50, 51.50)
# custom_ylim = (494, 500.05)
#
# # Setting the values for all axes.
plt.setp(ax, xlim=custom_xlim)#, ylim=custom_ylim)
# save option
#plt.savefig(r"filename", dpi=600)
plt.show()

