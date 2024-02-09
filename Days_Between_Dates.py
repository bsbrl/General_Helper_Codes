# -*- coding: utf-8 -*-

"""
Title: Date Difference Calculator

Description:
    This Python script calculates the number of days between a series of dates
    and the first date listed. It is designed to help users quickly understand
    the time intervals between key dates in a project or event timeline. The
    script takes a list of dates in MM/DD/YYYY format and outputs the number
    of days each date is from the first date on the list.

Usage:
    Ensure you have Python installed on your system.
    The script can be run from the command line or integrated into larger Python projects.
    Modify the 'dates' list variable with your specific dates before running.

Requirements:
    Python 3.x
    No external libraries are required for the basic functionality.

Author: 
    Ibrahim Oladepo

Contributors:
    ChatGPT - OpenAI's language model was consulted for assistance with code logic
    and documentation.

License:
    **

Example:
    To run the script, ensure the 'dates' list contains your dates in the correct format.
    Run the script using a Python interpreter:
    $ python date_difference_calculator.py

    Output will be displayed in the terminal or command prompt.

Notes:
    This script is for educational and practical purposes. It can be modified
    or expanded based on user requirements. For more complex date manipulations,
    consider using libraries like pandas or numpy.
"""

from datetime import datetime

# List of dates in MM/DD/YYYY format
# dates = [
#     "9/7/2023", "9/26/2023", "10/3/2023", "10/11/2023",
#     "10/17/2023", "10/25/2023", "10/31/2023", "11/28/2023", "12/8/2023"
# ]

# dates = [
#     "9/25/2023", "10/3/2023", "10/11/2023", "10/17/2023",
#     "10/25/2023", "10/31/2023", "11/22/2023", "11/28/2023", "12/8/2023"
# ]

dates = ["9/12/2023", "9/26/2023", "10/3/2023", "10/11/2023", "10/17/2023"]

# Convert string dates to datetime objects
date_objects = [datetime.strptime(date, "%m/%d/%Y") for date in dates]

# Calculate the number of days from the first date
days_from_first_date = [(date - date_objects[0]).days for date in date_objects]

print(days_from_first_date)

