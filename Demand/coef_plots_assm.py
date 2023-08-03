import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


coefs = #TODO: list of coefficients from the npy files

nsplits = 3

# plot coefficients over iterations
transposed_coefs = list(zip(*coefs))

plt.figure(dpi=200)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for i, line in enumerate(transposed_coefs[::-1]):
    plt.plot(line, label=f'HPI Q{nsplits-i}')

plt.xlabel('Iteration')
plt.title('Distance coefficient')
plt.legend(bbox_to_anchor=(1.05, 1))
# plt.show()



# comparing capacities
capacities = 
coefs = 
transposed_coefs = list(zip(*coefs))
plt.figure(dpi=200)
for i, line in enumerate(transposed_coefs[::-1]):
    plt.plot(line, label=f'HPI Quantile {nsplits-i}')

plt.xlabel('Capacity')
plt.title('Distance coefficient')
plt.xticks(range(len(capacities)), capacities)
plt.legend(bbox_to_anchor=(1.05, 1))
# plt.show()

