import matplotlib.pyplot as plt
import numpy as np

# Data
levels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
avg = np.array([0.90644, 0.933865, 0.976473, 0.990427, 0.995739, 0.998342, 0.999435, 1])
var = np.array([0.000737578, 0.000869074, 0.000154137, 2.92155e-05, 7.2505e-06, 1.24049e-06, 1.51959e-07, 5.58552e-14])

plt.figure(figsize=(8, 5))

# Plot with error bars
plt.errorbar(
    levels,
    avg,
    yerr=var,
    fmt='o-',
    capsize=4,
    linewidth=1.5
)

plt.xlabel("Level")
plt.ylabel("Average")
plt.title("Average vs Level with Variance as Error Bars")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig("rebuttal.png")

plt.show()