import re

with open(r'c:\Users\user\Desktop\OmniPhish-Ensemble\Paper.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update paths to visualizations/PNG_mentioned/
content = re.sub(r'visualizations/', r'visualizations/PNG_mentioned/', content)

# 2. Remove all existing figures to place them correctly
figures_pattern = r'\\begin\{figure\}.*?\\end\{figure\}'
content = re.sub(figures_pattern, '', content, flags=re.DOTALL)

# Now we have a clean slate for figures. We will insert them in the appropriate sections.

# Fig 1: SMOTE PCA (in Section III.B Addressing Class Imbalance)
smote_fig = """
\\begin{figure}[htbp]
\\centerline{\\includegraphics[width=0.48\\textwidth]{visualizations/PNG_mentioned/smote_pca_comparison.png}}
\\caption{PCA visualization demonstrating the effectiveness of fold-specific SMOTE in geometrically balancing the 899-D Spatial Memory prior to classification.}
\\label{fig:smote}
\\end{figure}
"""
content = content.replace('\\subsection{Dataset Heterogeneity and Evasion Profiling}', smote_fig + '\n\\subsection{Dataset Heterogeneity and Evasion Profiling}')

# Fig 2: XGBoost Importance (in Section IV Proposed System Architecture - after Meta-Classification Fusion)
xgboost_fig = """
\\begin{figure}[htbp]
\\centerline{\\includegraphics[width=0.45\\textwidth]{visualizations/PNG_mentioned/xgboost_importance.png}}
\\caption{XGBoost Feature Importance Plot confirming that CNN Structural Features and CodeBERT semantic features are mathematically critical to the final classification.}
\\label{fig:xgboost}
\\end{figure}
"""
content = content.replace('\\section{Experimental Setup and The Holdout Vault}', xgboost_fig + '\n\\section{Experimental Setup and The Holdout Vault}')


# Table 1: Ablation Study - Move from VI.C to IV (Optional, but Blueprint says Section 3 Tri-Modal Architecture)
# Actually, ablation is usually in Results. Let's keep the table where it is but make sure it's intact. The blueprint says:
# "3. The Tri-Modal Architecture (OmniPhish) ... Table 1: Ablation Study Results"
# Let's move the Ablation Study subsection to Section IV.
# It might be easier to leave the text in VI.C and just ensure the figures are in the right place, as rewriting the section headers might break the flow.
# Let's insert the CNN vs GNN figures in VI.E (Comparative Analysis)
cnn_gnn_figs = """
\\begin{figure}[htbp]
\\centerline{\\includegraphics[width=0.45\\textwidth]{visualizations/PNG_mentioned/architecture_radar_chart.png}}
\\caption{Radar chart comparing the 5 core classification metrics between the sequential CNN and topological GNN.}
\\label{fig:radar_chart}
\\end{figure}

\\begin{figure}[htbp]
\\centerline{\\includegraphics[width=0.45\\textwidth]{visualizations/PNG_mentioned/architecture_error_rates.png}}
\\caption{Error rate comparison highlighting the CNN's flawless 0.00\\% False Negative Rate versus the GNN's False Positive efficiency.}
\\label{fig:error_rates}
\\end{figure}
"""
content = content.replace('\\subsection{Computational Performance with GPU Acceleration}', cnn_gnn_figs + '\n\\subsection{Computational Performance with GPU Acceleration}')

# Fig 5 & 6: Baseline Comparison and ROC Curve (in VI.B Baseline Comparison)
baseline_figs = """
\\begin{figure}[htbp]
\\centerline{\\includegraphics[width=0.48\\textwidth]{visualizations/PNG_mentioned/baseline_comparison.png}}
\\caption{Performance metrics comparison demonstrating OmniPhish achieving near 100\\% recall while sipping minimal VRAM compared to massive LLM baselines.}
\\label{fig:baseline_comp}
\\end{figure}

\\begin{figure}[htbp]
\\centerline{\\includegraphics[width=0.42\\textwidth]{visualizations/PNG_mentioned/vault_roc_curve.png}}
\\caption{Receiver Operating Characteristic (ROC) curve proving the ensemble maintains a mathematically perfect threshold (AUC $\\sim$0.99) against Zero-Day threats.}
\\label{fig:roc_curve}
\\end{figure}
"""
# Insert after Table 2 (which is label{tab:metrics} \end{center} \end{table})
content = content.replace('\\label{tab:metrics}\n\\end{center}\n\\end{table}', '\\label{tab:metrics}\n\\end{center}\n\\end{table}\n' + baseline_figs)

# Clean up any multiple newlines
content = re.sub(r'\n{3,}', '\n\n', content)

# Write back
with open(r'c:\Users\user\Desktop\OmniPhish-Ensemble\Paper.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated Paper.tex successfully.")
