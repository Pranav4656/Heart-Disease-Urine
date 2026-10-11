import matplotlib.pyplot as plt
import numpy as np
import os

# Reference papers in sequence
labels = [
    "[4] Akella & Akella",
    "[5] Bharti et al.",
    "[6] Louridi et al.",
    "[9] Maini et al.",
    "Proposed Model"
]

# Reported accuracy values (%)
# Proposed Model = highest accuracy in this study: LightGBM
accuracy = [
    93.03,
    94.20,
    95.83,
    93.80,
    98.47
]

x = np.arange(len(labels))

plt.figure(figsize=(10, 5.5))

plt.plot(
    x,
    accuracy,
    marker="o",
    linewidth=2,
    markersize=7
)

# Add accuracy values above each point
for i, value in enumerate(accuracy):
    plt.text(
        i,
        value + 0.6,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xticks(x, labels, fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.xlabel("Reference / Model", fontsize=11)

plt.title(
    "Comparison of Reported Model Performance with Proposed Model",
    fontsize=12
)

plt.ylim(90, 100)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# Create comparison folder if it doesn't exist
os.makedirs("comparison", exist_ok=True)

# Save graph
output_path = os.path.join(
    "comparison",
    "reference_model_comparison_with_proposed_model.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nGraph saved to: {output_path}")