import matplotlib.pyplot as plt
import numpy as np  
import seaborn as sns

with open("sk328cip0min_cumulative_frequency_one.txt") as f:
    data_1 = [[float(i.replace("\n","")) for i in i.split(",")] for i in f.readlines()]

with open("sk328cip30min_cumulative_frequency_one.txt") as f:
    data_2 = [[float(i.replace("\n","")) for i in i.split(",")] for i in f.readlines()]

with open("sk328cip60min_cumulative_frequency_one.txt") as f:
    data_3 = [[float(i.replace("\n","")) for i in i.split(",")] for i in f.readlines()]

with open("sk328cip90min_cumulative_frequency_one.txt") as f:
    data_4 = [[float(i.replace("\n","")) for i in i.split(",")] for i in f.readlines()]

with open("sk328cip120min_cumulative_frequency_one.txt") as f:
    data_5 = [[float(i.replace("\n","")) for i in i.split(",")] for i in f.readlines()]



fig = plt.figure(figsize=(6,6))
sns.set()
ax = fig.add_subplot(111)
for n, a,b,c,d,e in zip([i for i in range(0,256)],data_1,data_2,data_3,data_4,data_5):
    if n == 0:
        # ax.plot([i for i in range(0,256)],a,label="0 min",color = "red",linewidth=1)
        # ax.plot([i for i in range(0,256)],b,label="30 min",color = "orange",linewidth=1)
        # ax.plot([i for i in range(0,256)],c,label="60 min",color = "green",linewidth=1)
        # ax.plot([i for i in range(0,256)],d,label="90 min",color = "blue",linewidth=1)
        ax.plot([i for i in range(0,256)],e,label="120 min",color = "purple",linewidth=1) 
    else:
        # ax.plot([i for i in range(0,256)],a,color = "red",linewidth=1)
        # ax.plot([i for i in range(0,256)],b,color = "orange",linewidth=1)
        # ax.plot([i for i in range(0,256)],c,color = "green",linewidth=1)
        # ax.plot([i for i in range(0,256)],d,color = "blue",linewidth=1)
        ax.plot([i for i in range(0,256)],e,color = "purple",linewidth=1)


y_min = np.min(data_1, axis=0)
y_max = np.max(data_1, axis=0)
x = np.arange(256)
plt.fill_between(x, y_min, y_max, color='red', alpha=0.4,zorder = 100)


ax.set_ylabel("Cumulative frequency(-)")
ax.set_xlabel("Fluo. intensity(-)")
ax.legend()
plt.savefig("cumulative_frequency_120min.png",dpi = 500)
