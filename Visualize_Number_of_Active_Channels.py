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
    MIT License

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
