from astropy.table import Table
from pathlib import Path
import numpy as np
import traceback
import sqlite3



class Project():
    '''
    This is a class that organizes LMT Project Data in a way that is
    most suitable for the ToltecProjectViewer.
    '''

    def __init__(self, sqlname):
        '''
        Create a Project object.
        Inputs:
         - sqlname - full path filename of an LMT sqlite database for a project.
        '''

        # I don't get this, but dasha is calling Project with '' on the input.
        if len(sqlname) == 0:
            return

        # If the file doesn't exist, return None
        file = Path(sqlname)
        if (not file.exists()):
            self.tables = {}
            return 
        
        # cast the entire database to a dict of astropy tables
        tables = sqlite_to_astropy_tables(sqlname)

        # cull out the tables we don't want
        self.tables = cull_tables(tables)

        # extract some useful information right off the bat.
        self.receivers = self.determine_receivers()
        self.obsnums = self.get_obsnums()
        self.science_tables = self.get_filtered_tables('Dcs', 'Dcs_ObsGoal', 'Science')
        self.pointing_tables = self.get_filtered_tables('Dcs', 'Dcs_ObsGoal', 'Pointing')
        self.science_sources = self.get_sources(self.science_tables)
        self.pointing_sources = self.get_sources(self.pointing_tables)
        return


    def createScienceReportData(self):
        '''
        A method to generate a dictionary of data for the dasha project viewer.
        The dictionary must be suitable to be stored in a dcc.Store data.
        This will be a dict of dicts where the primary key is the obsnum.
        '''
        obsnums = self.science_tables['Dcs']['ObsNum']
        data = dict()
        for obsnum in obsnums:
            t = self.get_filtered_tables('Dcs', 'ObsNum', obsnum)
            obs = str(obsnum)
            data[obs] = dict()
            data[obs]['valid'] = t['Scans']['Valid'][0]
            data[obs]['date'] = t['Scans']['Date'][0]
            data[obs]['time'] = t['Scans']['Time'][0]
            data[obs]['obsnum'] = obsnum
            data[obs]['source name'] = t['Source']['Source_SourceName'][0]
            data[obs]['instrument'] = self.determine_receivers(t)
            data[obs]['mapping mode'] = self.get_mapping_mode(t)
            data[obs]['tau'] = t['Radiometer']['Radiometer_Tau'][0]
            data[obs]['int time'] = t['Dcs']['Dcs_IntegrationTime'][0]
        return data


    def createPointingReportData(self):
        '''
        A method to generate a dictionary of data for the dasha project viewer.
        The dictionary must be suitable to be stored in a dcc.Store data.
        This will be a dict of dicts where the primary key is the obsnum.
        '''
        obsnums = self.pointing_tables['Dcs']['ObsNum']
        data = dict()
        for obsnum in obsnums:
            t = self.get_filtered_tables('Dcs', 'ObsNum', obsnum)
            obs = str(obsnum)
            data[obs] = dict()
            data[obs]['date'] = t['Scans']['Date'][0]
            data[obs]['time'] = t['Scans']['Time'][0]
            data[obs]['obsnum'] = obsnum
            data[obs]['source name'] = t['Source']['Source_SourceName'][0]
            rx = self.determine_receivers(t)
            data[obs]['instrument'] = rx
            rxr = rx+'Receiver'
            data[obs]['Az Off'] = t[rxr][rxr+'_AzPointOff'][0]
            data[obs]['El Off'] = t[rxr][rxr+'_ElPointOff'][0]
        return data
    

    def get_mapping_mode(self, t):
        has_lissajous = 'Lissajous' in t
        has_map = 'Map' in t
        if has_lissajous and has_map:
            return 'Rastajous'
        elif has_lissajous:
            return 'Lissajous'
        elif has_map:
            return 'Raster'
        else:
            return '--'
    

    def get_obsnums(self, tables=None):
        if tables is None:
            tables = self.tables
        return tables['Dcs']['ObsNum'].data


    def get_sources(self, tables=None):
        if tables is None:
            tables = self.tables
        return tables['Source']['Source_SourceName'].data
            

    def get_filtered_tables(self, tableName, key, value):
        filtered_tables = {}
        if tableName in self.tables:
            # Step 1: Find matching 'ObsNum' entries in the specified table
            matching_rows = self.tables[tableName][key] == value
            matching_obsnums = self.tables[tableName]['ObsNum'][matching_rows]

            # Convert to a list for compatibility with Astropy Table
            matching_obsnums_list = list(matching_obsnums)

            # Step 2: Filter all tables to include only rows with these 'ObsNums'
            for table_name, table_data in self.tables.items():
                if 'ObsNum' in table_data.colnames:
                    # Filter rows where 'ObsNum' is in the matching list
                    filtered_tables[table_name] = table_data[[
                        obsnum in matching_obsnums_list for obsnum in table_data['ObsNum']]]
                else:
                    print(f"Table {table_name} does not have an 'ObsNum' column.")
        else:
            print(f"{tableName} table not found in the tables.")
        return filtered_tables

    

    def determine_receivers(self, tables=None):
        if(tables is None):
            tables = self.tables
        rxs = ['Aztec', 'Toltec', 'Redshift', 'B4r', 'Msip', 'Sequoia', 'Vlbi']
        # Possible receivers have non-None elements in their tables
        receivers = []
        for t in tables.keys():
            for r in rxs:
                if r in t:
                    if tables[t] is not None:
                        if r not in receivers:
                            receivers.append(r)
        if len(receivers) == 1:
            receivers = receivers[0]
        return receivers

    


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def sqlite_to_astropy_tables(db_file, verbose=False):
    tables = {}
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Fetch the list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row['name'] for row in cursor.fetchall()] 

        if(verbose):
            print(f"Found tables: {table_names}")

        # For each table, convert data to an Astropy Table
        for table_name in table_names:
            query = f'SELECT * FROM "{table_name}"'
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows:
                tables[table_name] = Table(rows=[list(row.values()) for row in rows],
                                           names=list(rows[0].keys()))
        print("Read {0:} tables from {1:}.".format(len(tables), db_file))                
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        conn.close()

    return tables


def cull_tables(tables):
    # A list of tables we don't want to include in the dict of tables
    tables_to_remove = ['ApwSm_0_', 'ApwSm_1_', 'ApwSm_2_', 'ApwSm_3_', 'HoloBackend',
                        'HoloMap', 'HoloReceiver', 'M1', 'M1ActPos', 'M1CmdPos', 'M1PreSplit',
                        'M1Status', 'M3', 'OptBackend', 'PvCamera', 'RedshiftChassis_0_',
                        'RedshiftChassis_1_', 'RedshiftChassis_2_', 'RedshiftChassis_3_',
                        'Tiltmeter_0_', 'Tiltmeter_1_']
    for table in tables_to_remove:
        tables.pop(table, None)
    return tables


if __name__ == "__main__":
    p = Project('test_data/2018-S1-MU-77.sqlite')
