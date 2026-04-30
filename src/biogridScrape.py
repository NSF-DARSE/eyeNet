#Script to check if file exists and download updated biogrid files
#REST service access key 9fd864ebf56bbce8ec908e2fa35cee12
import os
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import json

accessKey = '9fd864ebf56bbce8ec908e2fa35cee12'
df = pd.read_excel('data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx')
regulators = df['Regulator'].unique()
targets = df['New gene symbol for taget'].unique()
regTarg = np.concatenate((regulators, targets))
regTarg = np.unique(regTarg)
print(regTarg)

BASE_URL = "https://downloads.thebiogrid.org/BioGRID/Current-Release/"
FILENAME = "BIOGRID-ALL-LATEST.tab3.zip"
METADATA_FILE = 'data/localVersion'
output_path = 'data/biogridScrape/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
serviceVersion = requests.get('https://webservice.thebiogrid.org/version?accesskey='+accessKey+'&format=json').content.decode("utf-8")

localVersion = os.path.exists(METADATA_FILE)

"""Compare remote vs local metadata."""
if not localVersion:
    test = True
else:
    test = serviceVersion != open(METADATA_FILE).read()

if test:
    with requests.get(BASE_URL, stream=True) as r:
        r.raise_for_status()
        with open(output_path+FILENAME, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                print(chunk)
                if chunk:
                    f.write(chunk)

url = "https://thebiogrid.org"
output_file = "biogrid_latest.tab3"
print(f"Downloading latest BioGRID TAB3 release...")

response = requests.get('https://downloads.thebiogrid.org/BioGRID/Release-Archive', stream=True)
if response.status_code == 200:
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            print(chunk)
            f.write(chunk)
    print("Download complete.")
else:
    print(f"Failed to download. HTTP Status: {response.status_code}")

tables = pd.read_html(chunk)
versionList = tables[0].T.values.tolist()
