import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
sota_tfidf = [92.62, 97.60, 91.99, 94.71]
baseline_rf = [92.74, 97.25, 92.51, 94.82]
sota_cnn = [96.75, 97.57, 97.91, 97.74]
ensemble = [98.37, 98.94, 98.76, 98.85]

x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

# IEEE standard styling (Grayscale/Blue for professional print)
rects1 = ax.bar(x - 1.5*width, sota_tfidf, width, label='TF-IDF (NLP)', color='#D3D3D3')
rects2 = ax.bar(x - 0.5*width, baseline_rf, width, label='Random Forest (Heuristics)', color='#A9A9A9')
rects3 = ax.bar(x + 0.5*width, sota_cnn, width, label='Raw 1D-CNN', color='#696969')
rects4 = ax.bar(x + 1.5*width, ensemble, width, label='OmniPhish (Proposed)', color='#1f77b4')

ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Performance Metrics Comparison on Zero-Day Vault', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)

# Set y-axis limits to highlight the difference tightly
ax.set_ylim(90, 100)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.savefig('visualizations/performance_comparison.png', dpi=300, bbox_inches='tight')
print('Successfully generated visualizations/performance_comparison.png')
