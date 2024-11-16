import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('NN_results.txt')
df.columns = ['x', 'target', 'pred']

#print(df)

plt.plot(df['x'], df['target'], 'r-')
plt.plot(df['x'], df['pred'], 'b-')
plt.grid()
plt.savefig('plot_predictions.png')
plt.show()
