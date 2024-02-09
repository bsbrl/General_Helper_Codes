# -*- coding: utf-8 -*-

"""
Title: Number of Active Channels Over Time Post-Surgery

Description:
    This Python script plots the number of active channels over time post-surgery for three different mice.
    The horizontal axis represents the number of days post-surgery, and the vertical axis represents
    the number of active channels. The data for each mouse is plotted as a separate line on the graph.

Usage:
    Run the script in a Python environment with matplotlib installed.
    The script does not require any command-line arguments or external input files.

Requirements:
    Python 3.x
    matplotlib library

Author: 
    Ibrahim Oladepo

Acknowledgments:
    ChatGPT was used in generating the code.

License:
    **

Example:
    Simply run the script to display the plot:
    $ python active_channels_plot.py

"""

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Scienceplots
import scienceplots
plt.style.use(['ieee'])

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

# Suppress annoying warnings!
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Data for Mouse 1423
mouse_1423_channels = [52, 53, 48, 44, 25, 28, 8, 32]
mouse_1423_days = [19, 26, 34, 40, 48, 54, 82, 92]

# Data for Mouse 1479
mouse_1479_channels = [59, 61, 60, 59, 59, 48, 37, 40]
mouse_1479_days = [8, 16, 22, 30, 36, 58, 64, 74]

# Data for Mouse 1481
mouse_1481_channels = [62, 62, 62, 58]
mouse_1481_days = [14, 21, 29, 35]

# Plotting the data
plt.figure(figsize=(7, 3), dpi=300)

plt.plot(mouse_1423_days, mouse_1423_channels, marker='o', label='Mouse 1423')
plt.plot(mouse_1479_days, mouse_1479_channels, marker='s', label='Mouse 1479')
plt.plot(mouse_1481_days, mouse_1481_channels, marker='^', label='Mouse 1481')

plt.xlabel('Number of Days Post Surgery')
plt.ylabel('Number of Active Channels')
plt.title('Active Channels Over Time Post-Surgery')
plt.legend()
# plt.grid(True)

plt.show()
