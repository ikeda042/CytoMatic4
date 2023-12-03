import numpy as np
import matplotlib.pyplot as plt

# with open("tmp_peak_analysis/sk328cip0min_peak_points_xs.txt", "r") as f:
#     xs_1 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     xs_1_normalized = []
#     for i in xs_1:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         xs_1_normalized.append(i.tolist())
# with open("tmp_peak_analysis/sk328cip0min_peak_points_ys.txt", "r") as f:
#     ys_1 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_1_normalized = []
#     for i in ys_1:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_1_normalized.append(i.tolist())

# with open("tmp_peak_analysis/sk328cip30min_peak_points_xs.txt", "r") as f:
#     xs_2 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     xs_2_normalized = []
#     for i in xs_2:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         xs_2_normalized.append(i.tolist())


# with open("tmp_peak_analysis/sk328cip30min_peak_points_ys.txt", "r") as f:
#     ys_2 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_2_normalized = []
#     for i in ys_2:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_2_normalized.append(i.tolist())

# with open("tmp_peak_analysis/sk328cip60min_peak_points_xs.txt", "r") as f:
#     xs_3 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     xs_3_normalized = []
#     for i in xs_3:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         xs_3_normalized.append(i.tolist())


# with open("tmp_peak_analysis/sk328cip60min_peak_points_ys.txt", "r") as f:
#     ys_3 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_3_normalized = []
#     for i in ys_3:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_3_normalized.append(i.tolist())


# with open("tmp_peak_analysis/sk328cip90min_peak_points_xs.txt", "r") as f:
#     xs_4 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     xs_4_normalized = []
#     for i in xs_4:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         xs_4_normalized.append(i.tolist())


# with open("tmp_peak_analysis/sk328cip90min_peak_points_ys.txt", "r") as f:
#     ys_4 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_4_normalized = []
#     for i in ys_4:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_4_normalized.append(i.tolist())

# with open("tmp_peak_analysis/sk328cip120min_peak_points_xs.txt", "r") as f:
#     xs_5 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     xs_5_normalized = []
#     for i in xs_5:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         xs_5_normalized.append(i.tolist())


# with open("tmp_peak_analysis/sk328cip120min_peak_points_ys.txt", "r") as f:
#     ys_5 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
#     ys_5_normalized = []
#     for i in ys_5:
#         i = np.array(i)
#         i = (i - i.min()) / (i.max() - i.min())
#         ys_5_normalized.append(i.tolist())



with open("temp_peak_analysis_tri/sk328tri0min_peak_points_xs.txt", "r") as f:
    xs_1 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    xs_1_normalized = []
    for i in xs_1:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        xs_1_normalized.append(i.tolist())

with open("temp_peak_analysis_tri/sk328tri0min_peak_points_ys.txt", "r") as f:
    ys_1 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_1_normalized = []
    for i in ys_1:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_1_normalized.append(i.tolist())
    
with open("temp_peak_analysis_tri/sk328tri30min_peak_points_xs.txt", "r") as f:
    xs_2 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    xs_2_normalized = []
    for i in xs_2:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        xs_2_normalized.append(i.tolist())

with open("temp_peak_analysis_tri/sk328tri30min_peak_points_ys.txt", "r") as f:
    ys_2 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_2_normalized = []
    for i in ys_2:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_2_normalized.append(i.tolist())

with open("temp_peak_analysis_tri/sk328tri60min_peak_points_xs.txt", "r") as f:
    xs_3 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    xs_3_normalized = []
    for i in xs_3:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        xs_3_normalized.append(i.tolist())
    
with open("temp_peak_analysis_tri/sk328tri60min_peak_points_ys.txt", "r") as f:
    ys_3 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_3_normalized = []
    for i in ys_3:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_3_normalized.append(i.tolist())
    
with open("temp_peak_analysis_tri/sk328tri90min_peak_points_xs.txt", "r") as f:
    xs_4 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    xs_4_normalized = []
    for i in xs_4:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        xs_4_normalized.append(i.tolist())
    
with open("temp_peak_analysis_tri/sk328tri90min_peak_points_ys.txt", "r") as f:
    ys_4 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_4_normalized = []
    for i in ys_4:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_4_normalized.append(i.tolist())

with open("temp_peak_analysis_tri/sk328tri120min_peak_points_xs.txt", "r") as f:
    xs_5 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    xs_5_normalized = []
    for i in xs_5:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        xs_5_normalized.append(i.tolist())

with open("temp_peak_analysis_tri/sk328tri120min_peak_points_ys.txt", "r") as f:
    ys_5 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_5_normalized = []
    for i in ys_5:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_5_normalized.append(i.tolist())


for i in ys_5_normalized:
    print(len(i))


# samples = ys_1_normalized+ys_2_normalized+ys_3_normalized+ys_4_normalized+ys_5_normalized

# concatenated_samples = np.column_stack(samples)

# plt.figure(figsize=(10, 6)) 
# plt.imshow(concatenated_samples, aspect='auto', cmap='inferno')
# plt.xlabel('Sample Number')
# cbar = plt.colorbar()
# cbar.set_label('Normalized fluo. intensity', rotation=270, labelpad=15)

# plt.ylabel('Split index')
# plt.savefig("heatmap.png", dpi=300)
# plt.close()



# print(f"len_ys_1_normalized: {len(ys_1_normalized)}")
# print(f"len_ys_2_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)}")
# print(f"len_ys_3_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)+ len(ys_3_normalized)}")
# print(f"len_ys_4_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)+ len(ys_3_normalized)+ len(ys_4_normalized)}")
# print(f"len_ys_5_normalized: {len(ys_1_normalized)+ len(ys_2_normalized)+ len(ys_3_normalized)+ len(ys_4_normalized)+ len(ys_5_normalized)}")

