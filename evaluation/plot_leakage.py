import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_privacy_leakage(full_model_file, ensemble_model_file, output_filename):
    column_names = [
        'epsilon', 'scale', 'precision', 'recall', 'accuracy',
        'tpr', 'fpr', 'privacy_leakage'
    ]
    df_full = pd.read_csv(full_model_file, sep='\s+', header=None, names=column_names)
    df_ens = pd.read_csv(ensemble_model_file, sep='\s+', header=None, names=column_names)
    stats_full = df_full.groupby('epsilon')['privacy_leakage'].agg(
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    ).reset_index()
    stats_ens = df_ens.groupby('epsilon')['privacy_leakage'].agg(
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    ).reset_index()
    full_y_err = [stats_full['median'] - stats_full['q1'], stats_full['q3'] - stats_full['median']]
    ens_y_err = [stats_ens['median'] - stats_ens['q1'], stats_ens['q3'] - stats_ens['median']]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(stats_full['epsilon'], stats_full['median'], yerr=full_y_err,
                fmt='-o', capsize=4, label='Full Model', alpha=0.8)
    ax.errorbar(stats_ens['epsilon'], stats_ens['median'], yerr=ens_y_err,
                fmt='-s', capsize=4, label='Ensemble Model', alpha=0.8)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, label='No Leakage')
    ax.set_xscale('log')
    ax.set_xlabel('Privacy Budget ε (Epsilon)')
    ax.set_ylabel('Privacy Leakage (TPR - FPR)')
    ax.set_title('Privacy Leakage for Membership Inference Attacks')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    ax.set_ylim(-0.05, 0.2)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as '{output_filename}'")

if __name__ == "__main__":
    plot_privacy_leakage(
        full_model_file='full_model_leakage.txt',
        ensemble_model_file='ensemble_leakage.txt',
        output_filename='privacy_leakage_plot.png'
    )