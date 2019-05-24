import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import scipy

'''
Data is the power spectral density (PSD) split into 5 bands ('Brain Wave Categories' below)
Measured in dB/Hz
Measurements are recorded at 1s intervals.
Timestep is the index from the beginning of the trial. Trials last 240s (4min)

Brain Wave Categories:
Delta = 0.5 - 4.0 hertz (deep stage 3 of NREM sleep, also known as slow-wave sleep)
Theta = 6.0 - 10.0 hertz (hippocampal and other subcortical activity)
Alpha = 8.0 - 12.0 hertz (thalamic pacemaker waves, Berger's waves)
Beta = 12.5 - 30.0 hertz (waking consciouscness)
Gamma = 32.0 - 100 hertz (unity of conscious perception)

Note that these categories are not collectively exhaustive or mutually exclusive,
and we do not have further information from the manufacture as to what exactly each band means.
'''

# import all data
# columns = 'Filename', 'Date', 'Time', 'Timestep', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'
data = pd.read_excel('./data/all_labeled_4min.xlsx')

# parse on trial and wave type (example)
# print(data.loc[data['Filename']=='eye2f.xlsx']['Delta'].head)

# Normalize
# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(data['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
# normalized_delta = pd.DataFrame(np_scaled)
# data['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] = normalized_delta
# print(data.head)

fig = plt.figure(figsize=(20,10))
plt.plot(data.loc[data['Filename']=='eye2f.xlsx']['Delta'])
plt.plot(data.loc[data['Filename']=='eye2f.xlsx']['Theta'])
plt.plot(data.loc[data['Filename']=='eye2f.xlsx']['Alpha'])
plt.plot(data.loc[data['Filename']=='eye2f.xlsx']['Beta'])
plt.plot(data.loc[data['Filename']=='eye2f.xlsx']['Gamma'])
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.xlim((0,240))
plt.xticks
plt.xticks(np.arange(0,240,step=5))
plt.xlabel('Time (s)')
plt.title('4 Minute EEG While Concentrating on the Word "Eye"')
plt.legend()
plt.show()