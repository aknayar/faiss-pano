import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data
levels = np.arange(16)

ALL_DATASETS = ['gist1m', 'openai', 'ada002', 'cifar10', 'fmnist', 'sift100m']

palette = sns.color_palette("muted", len(ALL_DATASETS))
COLORS = {}
for i, dataset in enumerate(ALL_DATASETS):
  COLORS[dataset] = palette[i]
COLORS['sift10m'] = COLORS['sift100m']

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # Axis label font size
    'xtick.labelsize': 12,     # X-tick label size
    'ytick.labelsize': 12,     # Y-tick label size
    'legend.fontsize': 12,      # Legend font size
    'figure.titlesize': 18,    # Figure title size
    'lines.linewidth': 2.5,    # Thicker lines
    'lines.markersize': 8,     # Larger markers
    'grid.linewidth': 0.8,     # Grid line width
    'axes.linewidth': 1.2,     # Axes border width
    'font.family': 'serif',    # Academic font
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,      # Set to True if you have LaTeX installed
    'figure.dpi': 150,         # High DPI for crisp text
    'savefig.dpi': 300,        # High DPI for saved figures
    'savefig.bbox': 'tight',   # Tight bounding box
    'savefig.pad_inches': 0.0  # Small padding
})

MEANS = {
'fmnist': np.array([
    0.859354, 0.914717, 0.939521, 0.957099, 0.969248, 0.978117,
    0.98532, 0.989339, 0.992475, 0.994735, 0.996692, 0.998175,
    0.999242, 0.99988, 1.0, 1.0
]),
'ada002': np.array([
    0.581766, 0.75009, 0.827124, 0.876467, 0.908518, 0.933796,
    0.953416, 0.968043, 0.97873, 0.984813, 0.990145, 0.992652,
    0.995929, 0.998257, 0.999815, 1.0
]),
'cifar10': np.array([
    0.911152, 0.963258, 0.980995, 0.989213, 0.993947, 0.996868,
    0.998507, 0.999454, 0.999829, 0.999956, 0.999986, 0.999996,
    0.999999, 1.0, 1.0, 1.0
]),
'openai': np.array([
    0.616568, 0.745965, 0.815452, 0.865332, 0.898637, 0.924804,
    0.944917, 0.959483, 0.970237, 0.978827, 0.984965, 0.990501,
    0.993083, 0.997171, 0.999499, 1.0
]),
'gist1m': np.array([
    0.843764, 0.9072, 0.938082, 0.957089, 0.971187, 0.980759,
    0.987161, 0.991078, 0.993944, 0.995952, 0.997824, 0.999018,
    0.999563, 0.999878, 0.99999, 1.0
]),
'sift100m': np.array([
    0.44304, 0.634245, 0.747081, 0.797146, 0.853415, 0.883571,
    0.906163, 0.929874, 0.941732, 0.953818, 0.970477, 0.979637,
    0.985992, 0.992041, 0.996273, 1.0
])
}

VARS = {
'fmnist': np.array([
    0.00847863, 0.00352366, 0.00187506, 0.0009559, 0.000503826,
    0.00025818, 0.000118989, 6.30252e-05, 3.16348e-05, 1.61693e-05,
    6.89624e-06, 2.50971e-06, 8.56907e-07, 1.75402e-07,
    8.47429e-10, 4.5616e-14
]),
'ada002': np.array([
    0.00201761, 0.000915582, 0.000477913, 0.000263287, 0.000153175,
    8.40081e-05, 4.28954e-05, 2.00979e-05, 8.80735e-06, 4.41233e-06,
    2.29345e-06, 1.18858e-06, 3.5994e-07, 3.9592e-07,
    4.09429e-07, 4.19802e-14
]),
'cifar10': np.array([
    0.00116254, 0.000236223, 7.17664e-05, 2.61046e-05, 9.3906e-06,
    2.86126e-06, 9.51557e-07, 3.80182e-07, 1.6706e-07, 5.64592e-08,
    2.04239e-08, 6.10023e-09, 1.54389e-09, 2.49955e-10,
    2.46804e-11, 2.24635e-13
]),
'openai': np.array([
    0.00125393, 0.000692588, 0.000408969, 0.000229862, 0.00013573,
    7.64999e-05, 4.20872e-05, 2.30265e-05, 1.25444e-05, 6.36111e-06,
    3.21501e-06, 1.77049e-06, 9.96141e-07, 5.72913e-07,
    5.94154e-07, 5.57904e-14
]),
'gist1m': np.array([
    0.00204056, 0.000735504, 0.000350287, 0.000180456, 9.13087e-05,
    4.51793e-05, 2.34341e-05, 1.2886e-05, 6.27366e-06, 3.43881e-06,
    1.8668e-06, 1.11126e-06, 7.34443e-07, 2.99274e-07,
    3.30462e-08, 3.7413e-14
]),
'sift100m': np.array([
    0.0183613, 0.0109561, 0.0058918, 0.00402433, 0.002284,
    0.00146416, 0.00100405, 0.000583703, 0.000426503, 0.000289279,
    0.000127149, 6.83345e-05, 3.67326e-05, 1.40825e-05,
    5.56309e-06, 2.1275e-14
])
}

DATASET_NAMES = {
  'gist1m': 'GIST',
  'sift100m': 'SIFT',
  'sift10m': 'SIFT',
  'fmnist': 'FMNIST',
  'cifar10': 'CIFAR-10',
  'ada002': 'Ada',
  'openai': 'Large'
}

fig, axes = plt.subplots(2, 3, figsize=(2.5*3, 2.5*2))

for k, dataset in enumerate(ALL_DATASETS):
    i = k // 3
    j = k % 3
    ax = axes[i, j]

    avg = MEANS[dataset]
    std = VARS[dataset]

    alpha = 0
    for i in range(1, 10):
        alpha += -np.log(1 - avg[i]) * 16 / i
    alpha /= 9``

    print(f"{dataset} alpha@8: {np.mean(alpha)}")
    
    color = COLORS[dataset]
    ax.errorbar(
        levels,
        avg,
        yerr=std,
        fmt='o-',
        capsize=4,
        markersize=5, 
        linewidth=1.5,
        color=color
    )

    if i == 1 and j == 1:
        ax.set_xlabel("Level", fontweight='bold', fontsize=14)
    if j == 0 and i == 0:
        ax.set_ylabel('LB/Exact Distance', fontweight='bold', fontsize=14)
        x, _ = ax.yaxis.get_label().get_position()
        ax.yaxis.set_label_coords(x -.4, -.2)
    ax.set_title(DATASET_NAMES[dataset], fontweight='bold', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(np.arange(0, 16, 4))

fig.tight_layout()
fig.savefig("experiment.pdf", dpi=300, pad_inches=0.01)
fig.savefig("experiment.png", dpi=300, pad_inches=0.01)
