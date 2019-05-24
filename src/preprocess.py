import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

eye24f = pd.read_excel('../data/eye24fmv.xlsx')
eye24f['Delta'].plot()
eye24f['Theta'].plot()
eye24f['Alpha'].plot()
eye24f['Beta'].plot()
eye24f['Gamma'].plot()
plt.show()