from datetime import datetime, timezone
from astropy.table import Table
from pathlib import Path
import json
import csv
import re

def parse_jpl_data(data_string):
    # Split the data string into lines
    lines = data_string.split('\n')

    # Prepare a list to hold the parsed data
    parsed_data = []

    # Iterate over each line
    for line in lines:
        # Split the line by comma
        values = line.split(',')

        # Check if the line has the expected number of values (at least 6)
        if len(values) < 6:
            continue

        # Parse the date/time value
        date_time_str = values[0].strip()
        try:
            # date_time = datetime.strptime(date_time_str, '%Y-%b-%d %H:%M')
            date_time = date_time_str
        except ValueError:
            # If the date/time format is incorrect, skip this line
            continue

        # Extract azimuth, elevation, and angular size
        try:
            azimuth = float(values[3].strip())
            elevation = float(values[4].strip())
            angular_size = float(values[5].strip())
        except ValueError:
            # If conversion to float fails, skip this line
            continue

        # Add the parsed data to the list
        parsed_data.append({
            'date_time': date_time,
            'azimuth': azimuth,
            'elevation': elevation,
            'angular_size': angular_size
        })

    # convert the list of dicts to an astropy table and write it out as a fits file
    data_table = Table(rows=parsed_data)
    return data_table


def parse_jpl_header(header_text):
    # Updated regex patterns for start and stop times, and step size
    patterns = {
        "target_body_name": r"Target body name: ([^(]+)",
        "center_body_name": r"Center body name: ([^(]+)",
        "start_time": r"Start time\s+:\s+A\.D\.\s+([^\s]+)\s+([^\s]+)\s+UT",
        "stop_time": r"Stop time\s+:\s+A\.D\.\s+([^\s]+)\s+([^\s]+)\s+UT",
        "step_size": r"Step-size\s+:\s+([^\\n]+)"
    }

    # Dictionary to store the parsed data
    parsed_data = {}

    # Iterate over the patterns and apply them to the header text
    for key, pattern in patterns.items():
        match = re.search(pattern, header_text)
        if match:
            # Concatenate date and time for start and stop times
            if key in ["start_time", "stop_time"]:
                parsed_data[key] = " ".join(match.groups())
            else:
                parsed_data[key] = match.group(1).strip()

    return parsed_data


def extract_and_parse_jpl_file(filename):
    # Read the entire file
    with open(filename, 'r') as file:
        file_contents = file.read()

    # Find the start and end of the header
    header_end_marker = "$$SOE"
    data_end_marker = "$$EOE"

    header_end_index = file_contents.find(header_end_marker)
    data_start_index = header_end_index + len(header_end_marker)
    data_end_index = file_contents.find(data_end_marker)

    # Extract the header and the data
    header = file_contents[:header_end_index].strip()
    data = file_contents[data_start_index:data_end_index].strip()

    # Parse the header
    header = parse_jpl_header(header)

    # Parse the data
    data = parse_jpl_data(data)

    # Write these out as a fits file (data) and a json file (header)
    fits_filename = filename.with_suffix('.fits')
    data.write(fits_filename, format='fits')
    json_filename = filename.with_suffix('.json')
    with open(json_filename, 'w') as file:
        json.dump(header, file, indent=4)

    return header, data


def readHorizonsData(filename):
    fits_filename = filename.with_suffix('.fits')
    json_filename = filename.with_suffix('.json')
    data = Table.read(fits_filename, format='fits')
    date_time = [datetime.strptime(t, '%Y-%b-%d %H:%M') for t in data['date_time']]
    data['date_time'] = [d.replace(tzinfo=timezone.utc) for d in date_time]

    with open(json_filename, 'r') as file:
        header = json.load(file)
    return header, data
