import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('NN_err.txt')
df.columns = ['N', 'err']

#print(df)

plt.semilogy(df['N'], df['err'], 'r-')
#plt.plot(df['err'], 'r-')

plt.xlabel('Function Evaluations')
plt.ylabel('Error')
plt.grid()
plt.savefig('plot_err.png')
plt.show()
