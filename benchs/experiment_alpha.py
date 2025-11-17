import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data from experiment.py
ALL_DATASETS = ['gist1m', 'openai', 'ada002', 'cifar10', 'fmnist', 'sift100m']

palette = sns.color_palette("muted", len(ALL_DATASETS))
COLORS = {}
for i, dataset in enumerate(ALL_DATASETS):
    COLORS[dataset] = palette[i]

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'grid.linewidth': 0.8,
    'axes.linewidth': 1.2,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.0
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

DATASET_NAMES = {
    'gist1m': 'GIST',
    'sift100m': 'SIFT',
    'fmnist': 'FMNIST',
    'cifar10': 'CIFAR-10',
    'ada002': 'Ada',
    'openai': 'Large'
}

levels = np.arange(16)
Levels = 15  # Maximum level (0-indexed, so 16 levels total)

# Compute alpha for each dataset and level
# From: 1 - e^(-alpha*level/Levels) = LB_level/|x-q|
# => e^(-alpha*level/Levels) = 1 - MEANS[dataset][level]
# => -alpha*level/Levels = log(1 - MEANS[dataset][level])
# => alpha = -Levels * log(1 - MEANS[dataset][level]) / level

alphas = {}
for dataset in ALL_DATASETS:
    alpha_values = []
    for level in levels:
        if level == 0:
            # Can't compute alpha at level 0 (division by zero)
            alpha_values.append(np.nan)
        else:
            lb_ratio = MEANS[dataset][level]
            if lb_ratio >= 1.0:
                # log(0) or log(negative) - cap at very small value
                alpha_values.append(np.nan)
            else:
                alpha = -Levels * np.log(1 - lb_ratio) / level
                alpha_values.append(alpha)
    alphas[dataset] = np.array(alpha_values)

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Alpha vs Level
ax1.set_title('Alpha vs Level', fontweight='bold', fontsize=16)
ax1.set_xlabel('Level', fontweight='bold', fontsize=14)
ax1.set_ylabel(r'$\alpha$', fontweight='bold', fontsize=14)

for dataset in ALL_DATASETS:
    valid_levels = levels[1:]  # Skip level 0
    valid_alphas = alphas[dataset][1:]
    # Filter out NaN values
    mask = ~np.isnan(valid_alphas)
    ax1.plot(valid_levels[mask], valid_alphas[mask], 'o-', 
             label=DATASET_NAMES[dataset], color=COLORS[dataset],
             linewidth=2, markersize=6)

ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='best', frameon=True)
ax1.set_xticks(np.arange(0, 16, 2))

# Plot 2: Theoretical vs Empirical R_level/R_Levels
# Empirical: R_level/R_Levels = 1 - MEANS[dataset][level]
# Theoretical: e^(-alpha*level/Levels) using computed alpha

ax2.set_title('Model Verification: $R_{level}/R_{Levels}$', fontweight='bold', fontsize=16)
ax2.set_xlabel('Level', fontweight='bold', fontsize=14)
ax2.set_ylabel('$R_{level}/R_{Levels}$', fontweight='bold', fontsize=14)

for dataset in ALL_DATASETS:
    # Empirical values
    empirical = 1 - MEANS[dataset]
    
    # Theoretical values using computed alpha
    theoretical = []
    for level in levels:
        if level == 0:
            theoretical.append(1.0)  # At level 0, R_0/R_Levels = 1
        else:
            alpha = alphas[dataset][level]
            if np.isnan(alpha):
                theoretical.append(np.nan)
            else:
                theoretical.append(np.exp(-alpha * level / Levels))
    theoretical = np.array(theoretical)
    
    # Plot empirical (solid line)
    ax2.plot(levels, empirical, 'o-', 
             label=f'{DATASET_NAMES[dataset]} (empirical)', 
             color=COLORS[dataset], linewidth=2, markersize=6)
    
    # Plot theoretical (dashed line)
    mask = ~np.isnan(theoretical)
    ax2.plot(levels[mask], theoretical[mask], '--', 
             color=COLORS[dataset], linewidth=2, alpha=0.7)

ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='best', frameon=True, fontsize=9)
ax2.set_xticks(np.arange(0, 16, 2))

fig.tight_layout()
fig.savefig("/home/akash/faiss-pano/benchs/experiment_alpha.pdf", dpi=300, pad_inches=0.01)
fig.savefig("/home/akash/faiss-pano/benchs/experiment_alpha.png", dpi=300, pad_inches=0.01)

print("Alpha analysis plots saved!")

# Print some statistics
print("\n=== Alpha Statistics ===")
for dataset in ALL_DATASETS:
    valid_alphas = alphas[dataset][~np.isnan(alphas[dataset])]
    if len(valid_alphas) > 0:
        print(f"{DATASET_NAMES[dataset]:10s}: mean={np.mean(valid_alphas):.3f}, std={np.std(valid_alphas):.3f}, "
              f"min={np.min(valid_alphas):.3f}, max={np.max(valid_alphas):.3f}")

