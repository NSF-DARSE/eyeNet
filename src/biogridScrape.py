#Script to check if file exists and download updated biogrid files
#REST service access key 9fd864ebf56bbce8ec908e2fa35cee12
import numpy as np
import pandas as pd
import requests as requests

accessKey = '9fd864ebf56bbce8ec908e2fa35cee12'
df = pd.read_excel('../data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx')
regulators = df['Regulator'].unique()
targets = df['New gene symbol for taget'].unique()
regTarg = np.concatenate((regulators, targets))
regTarg = np.unique(regTarg)
print(regTarg)
url = 'https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-5.0.255/BIOGRID-ALL-5.0.255.mitab.zip'
filename = '../data/'

