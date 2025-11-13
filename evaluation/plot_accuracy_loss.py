import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

def plot_accuracy_privacy_tradeoff(data_file, output_filename):
    interval_data_blocks = []
    current_block = []
    interval_titles = []
    title_map = {
        "-4.250000 to -3.750000": "(a) 4h prediction",
        "-8.250000 to -7.750000": "(b) 8h prediction",
        "-12.250000 to -11.750000": "(c) 12h prediction",
        "-12.250000 to -7.750000": "(d) 12-8h prediction",
        "-24.250000 to -11.750000": "(e) 24-12h prediction"
    }

    with open(data_file, 'r') as f:
        for line in f:
            if line.startswith("#Interval:"):
                if current_block:
                    interval_data_blocks.append("".join(current_block))
                    current_block = []
                interval_key = line.split("#Interval: ")[1].strip()
                interval_titles.append(title_map.get(interval_key, interval_key))
            elif not line.startswith('#'):
                current_block.append(line)
    if current_block:
        interval_data_blocks.append("".join(current_block))
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    column_names = [
        'epsilon', 'scale_full', 'auroc_full_noise', 'acc_loss_full',
        'scale_ens', 'auroc_ens_noise', 'acc_loss_ens'
    ]

    for i, block in enumerate(interval_data_blocks):
        ax = axes[i]
        df = pd.read_csv(io.StringIO(block), sep='\s+', header=None, names=column_names)
        stats_full = df.groupby('epsilon')['acc_loss_full'].agg(
            median='median',
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        ).reset_index()
        full_y_err = [stats_full['median'] - stats_full['q1'], stats_full['q3'] - stats_full['median']]
        stats_ens = df.groupby('epsilon')['acc_loss_ens'].agg(
            median='median',
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        ).reset_index()
        ens_y_err = [stats_ens['median'] - stats_ens['q1'], stats_ens['q3'] - stats_ens['median']]
        ax.errorbar(stats_full['epsilon'], stats_full['median'], yerr=full_y_err,
                    fmt='-o', capsize=3, label='Full Model')
        ax.errorbar(stats_ens['epsilon'], stats_ens['median'], yerr=ens_y_err,
                    fmt='-s', capsize=3, label='Ensemble Model')
        ax.set_xscale('log')
        ax.set_title(interval_titles[i])
        ax.grid(True, which="both", ls="--")
    fig.delaxes(axes[-1])
    fig.text(0.5, 0.06, 'Privacy Budget ε (Epsilon)', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Accuracy Loss', ha='center', va='center', rotation='vertical', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.85, 0.15), fontsize=12)
    plt.suptitle('Accuracy/Privacy Trade-off', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0.08, 0.08, 1, 0.93])
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved successfully as '{output_filename}'")

if __name__ == "__main__":
    plot_accuracy_privacy_tradeoff(
        data_file='accuracy_loss_data.txt',
        output_filename='accuracy_loss_plot.png'
    )