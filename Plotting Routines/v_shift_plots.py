import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
import matplotlib
import sys
sys.path.append(r"C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to upload\MCEM_files")
import ObjectBoundary
# change path here
path2data = r'C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to upload\PreRunData'
def find_centre(wp_boundary, centre_shift):
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

    return v_max_pos



'''
Extract relevant data for sensitivity analysis
looking for:
- total channel incision, diff between lowest point of channel at start and end
- channel drift from centre, x diff from lowest point
- WP width
- DP width
'''
results = []
runs = reversed([2, 3, 4, 5, 6, 7])
# loop through all the runs
for j in runs:
    # load data, must allow pickle true
    data = np.load(path2data + '\RUN_v_shift{}.npy'.format(j),
                   allow_pickle=
                   True)
    # use to get data is usable format not weird array thing
    data = data.item()
    if len(data) == 2:
        data = data['data']
    h_wanted = list(data.keys())
    # get position from object dictionary at time 0
    b_pos0 = np.array([x.position for x in data[0]])
    # define specific x and y
    x0, y0 = b_pos0.T

    # get position at end time
    b_pos1 = np.array([x.position for x in data[max(h_wanted)]])
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

    # Save results
    results.append({
        'Run': f'RUN{j}',
        'Total Incision': total_incision,
        'Total Drift': total_drift,
        'WP Width': WP_width,
        'DP Width': DP_width,
    })

    print('RUN{}'.format(j))
    print('Total incision', total_incision)
    print('Total drift', total_drift)
    print('WP width', WP_width)
    print('DP width', DP_width)
    print('Run hours', max(h_wanted))

# Convert to DataFrame
df = pd.DataFrame(results)

# Export to LaTeX
# latex_table = df.to_latex(index=False, float_format="%.3f")
# print(latex_table)

# === Figure setup ===
matplotlib.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 12))
ax_ind = np.indices(np.shape(ax))
ax_ind_list = list(zip(ax_ind[0].flatten(), ax_ind[1].flatten()))

# === Runs setup ===
# Exclude runs 1 and 8, remap runs 2–4 → 1–3, 5–7 → 4–6
original_runs = [2, 3, 4, 5, 6, 7]
run_names = [1, 2, 3, 4, 5, 6]
run_names = [6, 5, 4, 3, 2, 1]
# === Plotting ===
for j, run in enumerate(original_runs):
    data = np.load(
        path2data + '\RUN_v_shift{}.npy'.format(run),
        allow_pickle=True
    ).item()

    if len(data) == 2:
        data = data['data']

    h_wanted = list(data.keys())
    h_days = [h / 24.0 for h in h_wanted]
    cmap = cm.winter
    norm = Normalize(vmin=min(h_days), vmax=max(h_days))
    colour_cycle = cmap(norm(h_days))

    # Determine subplot position (skip [2,1] and [2,2] for colorbar)
    # New layout manually assigned for clarity
    subplot_positions = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]  # 6 plots
    row, col = subplot_positions[j]
    ax_here = ax[row, col]

    # --- Panel label A–F ---
    panel_labels = ['d', 'e', 'f','c', 'b', 'a']
    ax_here.text(
        0.02, 0.95,
        panel_labels[j],
        transform=ax_here.transAxes,
        fontsize=20,
        fontweight='bold',
        ha='left', va='top'
    )

    # --- Plot each hour ---
    lowest_y = np.inf
    lowest_x = None
    for i, c in zip(h_wanted, colour_cycle):
        b_pos = np.array([x.position for x in data[i]])
        x1, y1 = b_pos.T
        ax_here.scatter(x1, y1, label=f'hour {i}', color=c, marker='.')
        ax_here.set_ylim((470, 501))
        ax_here.set_xlim((33,68))
        ax_here.yaxis.set_major_locator(ticker.MultipleLocator(5))

        # Track the global lowest point for this subplot
        if np.min(y1) < lowest_y:
            lowest_y = np.min(y1)
            lowest_x = x1[np.argmin(y1)]

    # --- Add horizontal red line from lowest point to x=50 ---
    if lowest_y != np.inf:
        lowest_y = lowest_y - 1
        line_length = -(50 - lowest_x)
        # ax_here.hlines(lowest_y, xmin=lowest_x, xmax=50, color='red', linewidth=2)
        # Draw double-headed arrow
        ax_here.annotate(
            '', xy=(50, lowest_y), xytext=(lowest_x, lowest_y),
            arrowprops=dict(
                arrowstyle='<->',  # double-headed arrow
                color='red',
                linewidth=2
            )
        )

        # Place the text slightly above the line
        ax_here.text(
            (lowest_x + 50) / 2,  # midpoint in x
            lowest_y - 3,  # small vertical offset for clarity
            f"{line_length:.3f} m",  # formatted text
            color='red',
            fontsize=12,
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    ax_here.set_title(f'Run {run_names[j]}')
    ax_here.set_xlabel('(m)')
    ax_here.set_ylabel('(m)')
    ax_here.grid()

# === Move center plot content to top-right ===
# get position from object dictionary
b_pos = np.array([x.position for x in data[0]])
# define specific x and y
x0, y0 = b_pos.T
ax[0, 2].scatter(x0, y0, color='blue', marker='.')
# velocity shift lists
v_shift = [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75]
# v_shift = [-0.25, -0.5, -0.75, -0.99, 0.99, 0.75, 0.5, 0.25]
run_names = [1, 2, 3, 4, 5, 6]
for i, j in zip((v_shift), run_names):
    shifted = find_centre(b_pos, centre_shift=i)
    ax[0, 2].scatter(shifted[0], shifted[1], color='k', marker='*', label=i)
    ax[0, 2].text(shifted[0], shifted[1], f'{j}', color='k', fontsize=20, ha='center', va='bottom')
    ax[0, 2].grid()
    ax[0, 2].set_xlabel('(m)')
    ax[0, 2].set_ylabel('(m)')
    ax[0, 2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    ax[0,2].text(
            0.02, 0.95,
            'g',
            transform=ax[0,2].transAxes,
            fontsize=20,
            fontweight='bold',
            ha='left', va='top'
    )
custom_xlim = (49, 51)
custom_ylim = (499.65, 500.3)
# Top-right = [0,2], previously center = [1,1]
#ax[0, 2].set_title("Central Run Moved Here")
plt.setp(ax[0, 2], xlim=custom_xlim, ylim=custom_ylim)
# === Colorbar setup (bottom-right two plots) ===
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# Manually position colorbar in bottom-right area
cbar_ax = fig.add_axes([0.8, 0.08, 0.04, 0.55])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, label='Time (days)')
cbar.ax.invert_yaxis()
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Remove axes reserved for colorbar
ax[1, 2].axis('off')
ax[2, 2].axis('off')

# === Formatting ===
fig.tight_layout()  # leave room for colorbar rect=[0, 0, 0.75, 1]
# save file
#plt.savefig('filename', dpi=300)
plt.show()
