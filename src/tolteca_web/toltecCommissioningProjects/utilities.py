from astropy.coordinates import SkyCoord
from astropy.table import Table
import pandas as pd

def readTargetData():
    # Read the data from the published Google Sheet
    targetListURL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR9veD_j6QAeLbO15UaE3t7ETgValcnzEMBB3f4PNR_LrF-bMmMFgaAScWfv3zNKbR_FW-0ZSs--M_D/pub?gid=0&single=true&output=csv"
    targetList = pd.read_csv(targetListURL)
    return targetList


def convertPandas2Table(targetList):
    targetList = Table.from_pandas(targetList)
    # Generate coordinates
    c = []
    for row in targetList:
        c.append(SkyCoord(ra=row['RA'], dec=row['Dec'], unit='deg'))
    targetList['coords'] = c
    return targetList
    

# Given a project ID and a target list, fetch only the data associated
# with the project ID.
def fetchProjectID(pid, targetList):
    t = targetList[targetList['Proposal Id'] == pid]
    return t
    

