import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline = [92.58, 93.79, 94.44, 94.12]
ensemble = [94.10, 95.96, 95.00, 95.48]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

# IEEE standard styling (Grayscale/Blue for professional print)
rects1 = ax.bar(x - width/2, baseline, width, label='Random Forest Baseline', color='#A9A9A9')
rects2 = ax.bar(x + width/2, ensemble, width, label='Proposed Stacking Ensemble', color='#1f77b4')

ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Performance Metrics Comparison on Zero-Day Vault', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Set y-axis limits to highlight the difference
ax.set_ylim(85, 100)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print('Successfully generated performance_comparison.png')
