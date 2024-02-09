# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from datetime import datetime

# List of dates in MM/DD/YYYY format
dates = [
    "9/7/2023", "9/26/2023", "10/3/2023", "10/11/2023",
    "10/17/2023", "10/25/2023", "10/31/2023", "11/28/2023", "12/8/2023"
]

# Convert string dates to datetime objects
date_objects = [datetime.strptime(date, "%m/%d/%Y") for date in dates]

# Calculate the number of days from the first date
days_from_first_date = [(date - date_objects[0]).days for date in date_objects]

print(days_from_first_date)

