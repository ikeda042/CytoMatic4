import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# with open("temp_peak_analysis_tri/sk328tri0min_peak_points_ys.txt", "r") as f:
#     ys_1 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_1_normalized = []
#     for i in ys_1:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_1_normalized.append(i.tolist())

# with open("temp_peak_analysis_tri/sk328tri30min_peak_points_ys.txt", "r") as f:
#     ys_2 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_2_normalized = []
#     for i in ys_2:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_2_normalized.append(i.tolist())

# with open("temp_peak_analysis_tri/sk328tri60min_peak_points_ys.txt", "r") as f:
#     ys_3 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_3_normalized = []
#     for i in ys_3:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_3_normalized.append(i.tolist())

# with open("temp_peak_analysis_tri/sk328tri90min_peak_points_ys.txt", "r") as f:
#     ys_4 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_4_normalized = []
#     for i in ys_4:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_4_normalized.append(i.tolist())

# with open("temp_peak_analysis_tri/sk328tri120min_peak_points_ys.txt", "r") as f:
#     ys_5 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_5_normalized = []
#     for i in ys_5:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_5_normalized.append(i.tolist())


with open("peakys2/sk25_bta_FITC_c_peak_points_ys.txt", "r") as f:
    ys_1 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_1_normalized = []
    for i in ys_1:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_1_normalized.append(i.tolist())

with open("peakys2/sk25_bta_FITC_10_peak_points_ys.txt", "r") as f:
    ys_2 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_2_normalized = []
    for i in ys_2:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_2_normalized.append(i.tolist())

with open("peakys2/sk25_bta_FITC_13_peak_points_ys.txt", "r") as f:
    ys_3 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_3_normalized = []
    for i in ys_3:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_3_normalized.append(i.tolist())

with open("peakys2/sk25_bta_FITC_15_peak_points_ys.txt", "r") as f:
    ys_4 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_4_normalized = []
    for i in ys_4:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_4_normalized.append(i.tolist())

with open("peakys2/sk25_bta_FITC_18_peak_points_ys.txt", "r") as f:
    ys_5 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_5_normalized = []
    for i in ys_5:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_5_normalized.append(i.tolist())

with open("peakys2/sk25_bta_FITC_20_peak_points_ys.txt", "r") as f:
    ys_6 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_6_normalized = []
    for i in ys_6:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_6_normalized.append(i.tolist())


print(ys_1_normalized[1])
print(ys_2_normalized[1])
print(ys_3_normalized[1])
print(ys_4_normalized[1])
print(ys_5_normalized[1])
print(ys_6_normalized[1])
class HeadmapVector:
    def __init__(self, heatmap_vector:np.ndarray, sample_num:int):
        self.heatmap_vector:np.ndarray = heatmap_vector
        self.sample_num:int = sample_num

    def __gt__(self, other):
        # self_sum = np.sum([i for i in self.heatmap_vector if i >0.6])
        # other_sum = np.sum([i for i in other.heatmap_vector if i >0.6])
        # self_v = np.max(self.heatmap_vector) - np.median(self.heatmap_vector)
        # other_v = np.max(other.heatmap_vector) - np.median(other.heatmap_vector)
        self_v = np.sum(self.heatmap_vector)
        other_v = np.sum(other.heatmap_vector)
        return self_v < other_v

# vectors = [HeadmapVector(i,1) for i in ys_1_normalized] + [HeadmapVector(i,2) for i in ys_2_normalized] + [HeadmapVector(i,3) for i in ys_3_normalized] + [HeadmapVector(i,4) for i in ys_4_normalized] + [HeadmapVector(i,5) for i in ys_5_normalized]

vectors = sorted([HeadmapVector(i,1) for i in ys_1_normalized]) + sorted([HeadmapVector(i,2) for i in ys_2_normalized]) + sorted([HeadmapVector(i,3) for i in ys_3_normalized]) + sorted([HeadmapVector(i,4) for i in ys_4_normalized]) + sorted([HeadmapVector(i,5) for i in ys_5_normalized])+ sorted([HeadmapVector(i,6) for i in ys_6_normalized])

# vectors = sorted([HeadmapVector(i,1) for i in ys_1]) + sorted([HeadmapVector(i,2) for i in ys_2]) + sorted([HeadmapVector(i,3) for i in ys_3]) + sorted([HeadmapVector(i,4) for i in ys_4]) + sorted([HeadmapVector(i,5) for i in ys_5])+ sorted([HeadmapVector(i,6) for i in ys_6])

# vectors =  sorted([HeadmapVector(i,4) for i in ys_4_normalized]) + sorted([HeadmapVector(i,5) for i in ys_5_normalized])+ sorted([HeadmapVector(i,6) for i in ys_6_normalized])
title = "sk328tri (sorted by sum of Normalized fluo. intensity)"
# vectors = sorted(vectors,reverse=True)
concatenated_samples = np.column_stack([i.heatmap_vector for i in vectors])  # ベクトルを横に並べて結合

# 追加する横向きヒートマップのデータ（サンプルデータ)
additional_row = np.array([i.sample_num/4 for i in vectors])[None, :]

plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[30, 1], height_ratios=[1, 10], hspace=0.05, wspace=0.05)

# 追加行のヒートマップ
ax0 = plt.subplot(gs[0, 0])
ax0.imshow(additional_row, aspect='auto', cmap='inferno', extent=[0, concatenated_samples.shape[1], 0, 1])
ax0.set_xticks([])
ax0.set_yticks([])

# 元のヒートマップ
ax1 = plt.subplot(gs[1, 0])
im = ax1.imshow(concatenated_samples, aspect='auto', cmap='inferno')
ax1.set_xlabel(f'Sample Number {title}')
ax1.set_ylabel('Split index')

# カラーバーを二つのグラフの外側に配置
ax2 = plt.subplot(gs[:, 1])
plt.colorbar(im, cax=ax2)
ax2.set_ylabel('Normalized fluo. intensity', rotation=270, labelpad=15)

plt.savefig("heatmap_FITC2.png")

# print(f"len_ys_1_normalized: {len(ys_1_normalized)}")
# print(f"len_ys_2_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)}")
# print(f"len_ys_3_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)+ len(ys_3_normalized)}")
# print(f"len_ys_4_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)+ len(ys_3_normalized)+ len(ys_4_normalized)}")
# print(f"len_ys_5_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)+ len(ys_3_normalized)+ len(ys_4_normalized)+ len(ys_5_normalized)}")

# for i in ys_5_normalized:
#     print(len(i))