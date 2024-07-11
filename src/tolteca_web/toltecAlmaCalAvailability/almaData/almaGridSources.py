from astropy.io.votable import parse_single_table
from astropy.table import Table, vstack, hstack
from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
import numpy as np
import requests
import json
import io

class AlmaGridSources:
    
    def __init__(self, brightFilename='AlmaGridSources_bright.ecsv',
                 gridFilename='alma_grid.json',
                 minFlux=1.0,
                 regenerateCatalog=False):

        # Get the directory where almaGridSources.py is located
        almaData_dir = Path(__file__).resolve().parent
        
        # Construct the full paths to the data files
        self.brightFilename = almaData_dir / brightFilename
        self.gridFilename = almaData_dir / gridFilename
        
        self.sources = None
        self.minFlux = minFlux
        current_date = datetime.now()

        # First, just try reading the output source file
        if self.brightFilename.exists() and not regenerateCatalog:
            s = self.readInputTable()
            one_week_ago = current_date - timedelta(weeks=1)
            fileDate = s['Creation Date'][0]
            fileDate = datetime.strptime(fileDate, '%Y-%m-%d %H:%M:%S')
            if fileDate > one_week_ago:
                self.sources = s

        # If that didn't work, just regenerate it yourself.
        if self.sources is None:
            if not self.gridFilename.exists():
                print(f"No grid file found: {self.gridFilename}")
                self.sources = None
            else:
                self.sources = self.generateSources(current_date)

        # Finally, add a column of coordinates
        self.sources = self.add_coords_column(self.sources)        
        return

    
    def add_coords_column(self, sources):
        ra = sources['raDeg']
        dec = sources['decDeg']
        coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        sources['coords'] = coords
        return sources
    
    def generateSources(self, date):
        print(f"Generating sources from {self.gridFilename}")
        print(f"for {date.strftime('%d-%b-%Y')}")
        print("This may take a moment...")
        gg = self.readGrid()
        sources = self.extractSourceInfo(gg)
        sources = self.cullByDec(sources)
        t = self.getFluxes(date.strftime("%d-%b-%Y"), sources)
        self.writeOutputTable(t)
        print("done")
        return t
        
    def readGrid(self):
        with self.gridFilename.open('r') as file:
            g = json.load(file)
        return g

    def extractSourceInfo(self, gg):
        sources = []
        names = []
        gg.reverse()
        for g in gg:
            name = g['source']['names'][0]['name']
            if name not in names:
                sources.append({
                    'name': g['source']['names'][0]['name'],
                    'decDeg': g['source']['decDeg'],
                    'raDeg': g['source']['raDeg'],
                })
                names.append(name)
        return sources

    def cullByDec(self, sources, decLim=-20.):
        ss = []
        for s in sources:
            if s['decDeg'] > decLim:
                ss.append(s)
        return ss

    def fetch_data(self, date, frequency, name):
        url = f"https://almascience.nrao.edu/sc/flux?DATE={date}"
        url += f"&FREQUENCY={frequency:.1e}&NAME={name}"
        response = requests.get(url)
        return parse_single_table(io.BytesIO(response.content)).to_table()

    def getFluxes(self, date, sources):
        creationDate = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ss = []
        for i, s in enumerate(sources):
            print(f"Working on source {i+1} of {len(sources)} candidate grid sources.")
            t = self.fetch_data(date, 280.e9, s['name'])
            if t['FluxDensity'].data.data[0] > self.minFlux:
                ss.append(hstack([s, t]))
        ss = vstack(ss)
        ss['Creation Date'] = creationDate
        return ss

    def writeOutputTable(self, t):
        t.write(self.brightFilename, format='ascii.ecsv', overwrite=True)
        return

    def readInputTable(self):
        t = Table.read(self.brightFilename, format='ascii.ecsv')
        return t


if __name__ == '__main__':
    ags = AlmaGridSources(regenerateCatalog=True)
