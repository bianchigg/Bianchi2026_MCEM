from MCEM_files import Cfunctions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

# import data as a pandas df
data = pd.read_csv(r'C:\Users\ggmb1\OneDrive - Cardiff University\PhD\WriteUps\Paper 1\Code to '
                 r'upload\WeatherData\KAN_L.csv')
data['time'] = pd.to_datetime(data['time'])

# get data for may, june, july, august, when melt is expected
# list of months
month_list = ['April', 'May', 'June', 'July', 'August']
month_number = [4, 5, 6, 7, 8]
# list of the days of the year which are the middle of each month
mid_month_day_no = [105, 135, 166, 196, 227]
# create dictionary for data, and sub dictionary for each month
month_data_dict = dict()
for i in month_list:
    month_data_dict[i] = {}

# move all data in appropriate month dict
for i, j in zip(month_list, month_number):
    month_data_dict[i]['all_data'] = data[data['time'].dt.month == j]

# create hourly averages for whole month, i.e 1am each day averaged
for i in month_list:
    # group by hour
    month_data_dict[i]['all_data']['hour'] = month_data_dict[i]['all_data'].time.dt.hour
    avg_df = month_data_dict[i]['all_data'].groupby(['hour'], as_index=False).mean()
    month_data_dict[i]['hour_avg'] = avg_df

# create a new column in df which will be multiplied by incidence angle
for i in month_list:
    month_data_dict[i]['hour_avg']['dsr_cor_N45'] = month_data_dict[i]['hour_avg']['dsr_cor']
    month_data_dict[i]['hour_avg']['dsr_cor_E45'] = month_data_dict[i]['hour_avg']['dsr_cor']
    month_data_dict[i]['hour_avg']['dsr_cor_S45'] = month_data_dict[i]['hour_avg']['dsr_cor']
    month_data_dict[i]['hour_avg']['dsr_cor_W45'] = month_data_dict[i]['hour_avg']['dsr_cor']

# loop around each entry and multiply by relevant incidence angle
for i, j in zip(month_list, month_number):
    # loop around hours in day, this is ok as index starts at 0
    for h in range(0, 24):
        month_data_dict[i]['hour_avg']['dsr_cor_N45'][h] = (month_data_dict[i]['hour_avg']['dsr_cor_N45'][h] *
                                                            np.cos(Cfunctions.incidence_angle(lat=np.deg2rad(67),
                                                                                              hour_angle=Cfunctions.hour_angle(
                                                                                                  h),
                                                                                              declination=Cfunctions.declination(
                                                                                                  mid_month_day_no[
                                                                                                      j - 4]),
                                                                                              surface_azimuth=0,
                                                                                              observer_tilt=np.pi / 4)))

        month_data_dict[i]['hour_avg']['dsr_cor_E45'][h] = (month_data_dict[i]['hour_avg']['dsr_cor_E45'][h] *
                                                            np.cos(Cfunctions.incidence_angle(lat=np.deg2rad(67),
                                                                                              hour_angle=Cfunctions.hour_angle(
                                                                                                  h),
                                                                                              declination=Cfunctions.declination(
                                                                                                  mid_month_day_no[
                                                                                                      j - 4]),
                                                                                              surface_azimuth=np.pi / 2,
                                                                                              observer_tilt=np.pi / 4)))

        month_data_dict[i]['hour_avg']['dsr_cor_S45'][h] = (month_data_dict[i]['hour_avg']['dsr_cor_S45'][h] *
                                                            np.cos(Cfunctions.incidence_angle(lat=np.deg2rad(67),
                                                                                              hour_angle=Cfunctions.hour_angle(
                                                                                                  h),
                                                                                              declination=Cfunctions.declination(
                                                                                                  mid_month_day_no[
                                                                                                      j - 4]),
                                                                                              surface_azimuth=np.pi,
                                                                                              observer_tilt=np.pi / 4)))

        month_data_dict[i]['hour_avg']['dsr_cor_W45'][h] = (month_data_dict[i]['hour_avg']['dsr_cor_W45'][h] *
                                                            np.cos(Cfunctions.incidence_angle(lat=np.deg2rad(67),
                                                                                              hour_angle=Cfunctions.hour_angle(
                                                                                                  h),
                                                                                              declination=Cfunctions.declination(
                                                                                                  mid_month_day_no[
                                                                                                      j - 4]),
                                                                                              surface_azimuth=-np.pi
                                                                                                              / 2,
                                                                                              observer_tilt=np.pi /
                                                                                                            4)))

        # below zero check
        if month_data_dict[i]['hour_avg']['dsr_cor_N45'][h] < 0:
            month_data_dict[i]['hour_avg']['dsr_cor_N45'][h] = 0
        # below zero check
        if month_data_dict[i]['hour_avg']['dsr_cor_E45'][h] < 0:
            month_data_dict[i]['hour_avg']['dsr_cor_E45'][h] = 0
        # below zero check
        if month_data_dict[i]['hour_avg']['dsr_cor_S45'][h] < 0:
            month_data_dict[i]['hour_avg']['dsr_cor_S45'][h] = 0
        # below zero check
        if month_data_dict[i]['hour_avg']['dsr_cor_W45'][h] < 0:
            month_data_dict[i]['hour_avg']['dsr_cor_W45'][h] = 0

# define figure and subplots
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(12, 16))

# plots the average solar radiation for whole day
for i in month_list:
    ax[0].plot(month_data_dict[i]['hour_avg']['hour'], month_data_dict[i]['hour_avg']['dsr_cor'])
    ax[1].plot(month_data_dict[i]['hour_avg']['hour'], month_data_dict[i]['hour_avg']['dsr_cor_N45'])
    ax[2].plot(month_data_dict[i]['hour_avg']['hour'], month_data_dict[i]['hour_avg']['dsr_cor_E45'])
    ax[3].plot(month_data_dict[i]['hour_avg']['hour'], month_data_dict[i]['hour_avg']['dsr_cor_S45'])
    ax[4].plot(month_data_dict[i]['hour_avg']['hour'], month_data_dict[i]['hour_avg']['dsr_cor_W45'])

ax[0].set_xlabel('Hour')
ax[0].set_xticks(ticks=np.linspace(0, 23, 24))
ax[0].set_title('Flat Surface')

ax[1].set_xticks(ticks=np.linspace(0, 23, 24))
ax[1].set_title('North Facing')
ax[2].set_xticks(ticks=np.linspace(0, 23, 24))
ax[2].set_title('East Facing')
ax[3].set_xticks(ticks=np.linspace(0, 23, 24))
ax[3].set_title('South Facing')
ax[4].set_xticks(ticks=np.linspace(0, 23, 24))
ax[4].set_xlabel('Hour of Day')
ax[4].set_title('West Facing')

fig.supylabel('Solar Radiation W/m$^2$')

for i in range(0, 5):
    ax[i].set_ylim(0, 700)

ax[4].legend(labels=month_list)
for axs in ax.flat:
    axs.label_outer()

size = 10
plt.rc('font', size=20)  # controls default text sizes
plt.rc('axes', titlesize=15)  # fontsize of the axes title
plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
plt.rc('legend', fontsize=15)  # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title

fig.tight_layout()
plt.savefig(r'C:\Users\ggmb1\PycharmProjects\MCEM\images\solar_orientation.png', dpi=300)
plt.show()
