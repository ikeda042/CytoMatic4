import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def skewness_analysis(data_all:list[list[list[float]]]) -> None:
    sns.set()
    fig = plt.figure(figsize=[8, 6])
    skewnesses_all = []
    for data in data_all:
        skewnesses = []
        for peak_path_i in data:
            skewness = stats.skew(peak_path_i)
            if skewness < 0 or True:
                color = "red" if skewness > 0 else "blue"
                plt.plot(
                    [i for i in range(len(peak_path_i))],
                    peak_path_i,
                    alpha=0.9,
                    color=color,
                    linewidth=0.1,
                )
            skewnesses.append(skewness)
        skewnesses_all.append(skewnesses)

    plt.xlabel("location along the peak path (a.u.)")
    plt.ylabel("Normalized fluo. intensity (a.u.)")
    plt.grid(True)
    fig.savefig("result_peak_path_by_skewness.png", dpi=500)
    plt.close()

    fig = plt.figure(figsize=[8,6])
    plt.boxplot(skewnesses_all,sym="")
    for i, data in enumerate(skewnesses_all, start=1):
        x = np.random.normal(i, 0.04, size=len(data))
        plt.plot(x, data, 'o', alpha=0.4)  

    plt.xticks([i+1 for i in range(len(skewnesses_all))], [f'{i*30} (n={len(data_i)})' for i,data_i in enumerate(skewnesses_all,start=0)])

    plt.xlabel('Triclosan exposure time (min)')
    plt.ylabel("Skewness(-)")
    plt.grid(True)
    fig.savefig("result_skewness_boxplot.png",dpi = 500)
    for skewnesses in skewnesses_all:
        print(len([i for i in skewnesses if i > 0]) / len(skewnesses))

