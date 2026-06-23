import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def generate_detailed_comparison():
    # ---------------------------------------------------------
    # EXACT METRICS EXTRACTED FROM PIPELINE LOGS
    # ---------------------------------------------------------
    
    # CNN CM: TN=185, FP=44, FN=0, TP=570
    cnn_tn, cnn_fp, cnn_fn, cnn_tp = 185, 44, 0, 570
    
    # GNN CM (calculated from rates): Support Benign=211, Phishing=588
    gnn_tn, gnn_fp, gnn_fn, gnn_tp = 184, 27, 21, 567
    
    # Calculate Detailed Statistical Rates
    def calc_rates(tn, fp, fn, tp):
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate (Miss Rate)
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0  # False Discovery Rate
        f1 = 2 * (precision * recall) / (precision + recall)
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        return {
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall (Sensitivity)': recall * 100,
            'Specificity (TNR)': specificity * 100,
            'False Positive Rate': fpr * 100,
            'False Negative Rate (Miss Rate)': fnr * 100,
            'False Discovery Rate': fdr * 100,
            'F1-Score': f1 * 100,
            'MCC (Correlation)': mcc
        }
        
    cnn_stats = calc_rates(cnn_tn, cnn_fp, cnn_fn, cnn_tp)
    gnn_stats = calc_rates(gnn_tn, gnn_fp, gnn_fn, gnn_tp)
    
    # ---------------------------------------------------------
    # COMPUTE DETAILED COMPARISON TABLE
    # ---------------------------------------------------------
    df = pd.DataFrame({
        'Metric': list(cnn_stats.keys()),
        'OmniPhish CNN': list(cnn_stats.values()),
        'OmniPhish GNN': list(gnn_stats.values())
    })
    
    # Add computational stats
    df.loc[len(df.index)] = ['Latency (ms/URL)', 93.79, 95.28]
    df.loc[len(df.index)] = ['Peak VRAM (MB)', 1798.29, 2964.40]
    
    # Format and save to Markdown
    os.makedirs("visualizations", exist_ok=True)
    with open("visualizations/ARCHITECTURE_COMPARISON.md", "w", encoding="utf-8") as f:
        f.write("# Hyper-Detailed Architectural Comparison (CNN vs GNN)\n\n")
        f.write("This table contains advanced statistical rates derived directly from the confusion matrices of the Zero-Day vault evaluation.\n\n")
        f.write("| Statistical Metric | OmniPhish CNN | OmniPhish GNN | Difference | Winner |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for i, row in df.iterrows():
            metric = row['Metric']
            val_cnn = row['OmniPhish CNN']
            val_gnn = row['OmniPhish GNN']
            
            diff = abs(val_cnn - val_gnn)
            
            # Determine winner (Lower is better for FPR, FNR, FDR, Latency, VRAM)
            lower_is_better = ['False Positive Rate', 'False Negative Rate (Miss Rate)', 'False Discovery Rate', 'Latency (ms/URL)', 'Peak VRAM (MB)']
            if metric in lower_is_better:
                winner = "CNN" if val_cnn < val_gnn else "GNN"
            else:
                winner = "CNN" if val_cnn > val_gnn else "GNN"
                
            # Formatting
            if "MCC" in metric:
                f.write(f"| **{metric}** | {val_cnn:.4f} | {val_gnn:.4f} | {diff:.4f} | 🏆 **{winner}** |\n")
            elif "VRAM" in metric:
                f.write(f"| **{metric}** | {val_cnn:.2f} MB | {val_gnn:.2f} MB | {diff:.2f} MB | 🏆 **{winner}** |\n")
            elif "Latency" in metric:
                f.write(f"| **{metric}** | {val_cnn:.2f} ms | {val_gnn:.2f} ms | {diff:.2f} ms | 🏆 **{winner}** |\n")
            else:
                f.write(f"| **{metric}** | {val_cnn:.2f}% | {val_gnn:.2f}% | {diff:.2f}% | 🏆 **{winner}** |\n")
                
    # ---------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # ---------------------------------------------------------
    
    # 1. Advanced Radar Chart for Classification Metrics
    categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    cnn_radar = [cnn_stats['Accuracy'], cnn_stats['Precision'], cnn_stats['Recall (Sensitivity)'], cnn_stats['Specificity (TNR)'], cnn_stats['F1-Score']]
    cnn_radar += cnn_radar[:1]
    
    gnn_radar = [gnn_stats['Accuracy'], gnn_stats['Precision'], gnn_stats['Recall (Sensitivity)'], gnn_stats['Specificity (TNR)'], gnn_stats['F1-Score']]
    gnn_radar += gnn_radar[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=12, fontweight='bold')
    
    ax.plot(angles, cnn_radar, linewidth=2, linestyle='solid', label='OmniPhish CNN', color='#1f77b4')
    ax.fill(angles, cnn_radar, '#1f77b4', alpha=0.1)
    
    ax.plot(angles, gnn_radar, linewidth=2, linestyle='solid', label='OmniPhish GNN', color='#ff7f0e')
    ax.fill(angles, gnn_radar, '#ff7f0e', alpha=0.1)
    
    plt.ylim(85, 100)  # Focus on the high-performance area
    plt.title('Radar Comparison: CNN vs GNN Deep Statistical Metrics', size=16, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('visualizations/architecture_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error Rate Bar Chart (Focusing on mistakes)
    error_metrics = ['False Positive Rate\n(Safe blocked)', 'False Negative Rate\n(Phish missed)']
    cnn_errors = [cnn_stats['False Positive Rate'], cnn_stats['False Negative Rate (Miss Rate)']]
    gnn_errors = [gnn_stats['False Positive Rate'], gnn_stats['False Negative Rate (Miss Rate)']]
    
    x = np.arange(len(error_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, cnn_errors, width, label='OmniPhish CNN', color='#d62728')
    ax.bar(x + width/2, gnn_errors, width, label='OmniPhish GNN', color='#2ca02c')
    
    ax.set_ylabel('Error Rate Percentage (%)', fontweight='bold')
    ax.set_title('Critical Error Rate Comparison (Lower is Better)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(error_metrics, fontweight='bold')
    ax.legend()
    
    # Add labels on top of bars
    for i, v in enumerate(cnn_errors):
        ax.text(i - width/2, v + 0.2, f"{v:.2f}%", ha='center', fontweight='bold')
    for i, v in enumerate(gnn_errors):
        ax.text(i + width/2, v + 0.2, f"{v:.2f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('visualizations/architecture_error_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("[*] Successfully generated detailed statistical comparisons in visualizations/")

if __name__ == "__main__":
    generate_detailed_comparison()
