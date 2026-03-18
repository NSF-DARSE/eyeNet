#Script to check if file exists and download updated biogrid files
#REST service access key 9fd864ebf56bbce8ec908e2fa35cee12
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import json

accessKey = '9fd864ebf56bbce8ec908e2fa35cee12'
df = pd.read_excel('../data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx')
regulators = df['Regulator'].unique()
targets = df['New gene symbol for taget'].unique()
regTarg = np.concatenate((regulators, targets))
regTarg = np.unique(regTarg)
print(regTarg)

BASE_URL = "https://downloads.thebiogrid.org/BioGRID/Latest-Release/"
FILENAME = "BIOGRID-ALL-LATEST.psi.zip"
METADATA_FILE = "biogrid_psi_metadata.json"


def get_remote_metadata(url):
    """Fetch remote file headers (Last-Modified, ETag)."""
    response = requests.head(url, allow_redirects=True)
    response.raise_for_status()
    return {
        "last_modified": response.headers.get("Last-Modified"),
        "etag": response.headers.get("ETag")
    }

def load_local_metadata():
    if Path(METADATA_FILE).exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_local_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def is_newer(remote, local):
    """Compare remote vs local metadata."""
    if not local:
        return True
    return (
        remote.get("etag") != local.get("etag") or
        remote.get("last_modified") != local.get("last_modified")
    )

def download_file(url, output_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)