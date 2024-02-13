import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

sns.set()

paths = [i for i in os.listdir() if i.endswith(".txt") and i[0] == "F"]

length_dict = {}
for path in paths:
    print(path)
    with open(path, "r") as f:
        data_i = [line for line in f.readlines()[1:]]
        data_i = [i.split(",") for i in data_i]
        data_i = [[float(i[0]), float(i[1].replace("\n", ""))] for i in data_i]
        longer_axis_lengths = [i[0] for i in data_i]
        shorter_axis_lengths = [i[1] for i in data_i]
        length_dict[path.split(".")[0]] = [longer_axis_lengths, shorter_axis_lengths]


fig = plt.figure(figsize=[28, 9])

labels = length_dict.keys()
data_long = [length_dict[i][0] for i in labels]
data_short = [length_dict[i][1] for i in labels]


data = data_long
plt.boxplot(data, sym="")
for i, d in enumerate(data, start=1):
    x = np.random.normal(i, 0.04, size=len(d))
    plt.plot(x, d, "o", alpha=0.5)

plt.xticks(
    [i + 1 for i in range(len(data))],
    [f"{i.split('_')[0].split('b')[-1]}" for i in labels],
)

plt.xlabel("Samples")
plt.ylabel("Lengths (pixel)")
plt.grid(True)

fig.savefig("result.png", dpi=500)


# temp string
