import matplotlib.pyplot as plt
import numpy as np

# 1. Define the Metrics and Data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
stacking_scores = [94.12, 92.11, 98.13, 95.02]
baseline_scores = [88.77, 89.69, 88.78, 89.23]

# 2. Set up the Bar Chart layout
x = np.arange(len(metrics))  # The label locations
width = 0.35  # The width of the bars

# 3. Create the Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plotting the bars (using IEEE-friendly muted colors: Dark Blue and Gray)
rects1 = ax.bar(x - width/2, stacking_scores, width, label='Proposed Stacking Ensemble', color='#1f497d')
rects2 = ax.bar(x + width/2, baseline_scores, width, label='Random Forest Baseline', color='#a5a5a5')

# 4. Format the Axes and Labels
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(85, 105) # Starting y-axis at 85 to emphasize the differences

# Add a subtle grid behind the bars for readability
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='gray', linestyle='dashed', alpha=0.3)

# 5. Add the exact numbers on top of the bars
def autolabel(rects):
    """Attach a text label above each bar, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# 6. Final Polish and Save
ax.legend(loc='upper right', framealpha=1)
fig.tight_layout()

# Save the figure as a high-resolution PNG (300 dpi is standard for IEEE)
plt.savefig('Performance Graph.png', dpi=300, bbox_inches='tight')
plt.show()