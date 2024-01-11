# This is a template for 8-bit depth CDF analysis
import matplotlib.pyplot as plt
import numpy as np 

with open("sk328gen0min_cumulative_frequency_one.txt") as f:
    data1 = [[float(i.replace("\n","")) for i in j.split(',')] for j in f.readlines()] 

with open("sk328gen30min_cumulative_frequency_one.txt") as f:
    data2 = [[float(i.replace("\n","")) for i in j.split(',')] for j in f.readlines()] 

with open("sk328gen60min_cumulative_frequency_one.txt") as f:
    data3 = [[float(i.replace("\n","")) for i in j.split(',')] for j in f.readlines()]

with open("sk328gen90min_cumulative_frequency_one.txt") as f:
    data4 = [[float(i.replace("\n","")) for i in j.split(',')] for j in f.readlines()]

with open("sk328gen120min_cumulative_frequency_one.txt") as f:
    data5 = [[float(i.replace("\n","")) for i in j.split(',')] for j in f.readlines()]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)   
ax.set_title('Cumulative Frequency')
ax.set_xlabel('Pixel Value')
ax.set_ylabel('Frequency')
ax.set_xlim([0, 255])
ax.set_ylim([0, 1])

n = 0
line_width = 0.3

for a,b,c,d,e in zip(data1,data2,data3,data4,data5):
    # ax.plot([i for i in range(0,256)],a,label='0min' if n == 0 else None,color ='red',linewidth=line_width)
    ax.plot([i for i in range(0,256)],b,label='30min' if n == 0 else None,color ='orange',linewidth=line_width)
    ax.plot([i for i in range(0,256)],c,label='60min' if n == 0 else None,color ='green',linewidth=line_width)
    ax.plot([i for i in range(0,256)],d,label='90min' if n == 0 else None,color ='blue',linewidth=line_width)
    ax.plot([i for i in range(0,256)],e,label='120min' if n == 0 else None,color ='purple',linewidth=line_width)
    n += 1
# 透明度のためのalpha値を設定
fill_alpha = 0.4

# 各データセットの最小値と最大値を求め、塗りつぶしを追加
plt.fill_between(range(256), np.min(data1, axis=0), np.max(data1, axis=0), color="red", alpha=fill_alpha)
# plt.fill_between(range(256), np.min(data2, axis=0), np.max(data2, axis=0), color="orange", alpha=fill_alpha)
# plt.fill_between(range(256), np.min(data3, axis=0), np.max(data3, axis=0), color="green", alpha=fill_alpha)
# plt.fill_between(range(256), np.min(data4, axis=0), np.max(data4, axis=0), color="blue", alpha=fill_alpha)
# plt.fill_between(range(256), np.min(data5, axis=0), np.max(data5, axis=0), color="purple", alpha=fill_alpha)



ax.legend(loc='upper left')
ax.grid()
plt.savefig('result_CDF.png',dpi = 500)