import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/UBER.csv")
plt.plot(df['Close'])
plt.show()