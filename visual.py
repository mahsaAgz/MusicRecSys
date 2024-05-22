import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_smooth_data(file_path, window_size=10):
    # Load the CSV file
    data = pd.read_csv(file_path)
    # Assuming the second column is the desired metric
    metric_column = data.columns[1]
    # Apply a rolling average for smoothing
    smoothed_data = data[metric_column].rolling(window=window_size).mean()
    return data['Step'], smoothed_data


def plot_and_save(files, labels, window_size=10, save_formats=['pdf', 'svg', 'jpg'], plot_title='HR'):
    plt.figure(figsize=(10, 6))

    for file_path, label in zip(files, labels):
        step, smoothed_data = load_and_smooth_data(file_path, window_size)
        plt.plot(step, smoothed_data, label=label)

    plt.xlabel('Step')
    plt.ylabel('Metric Value')
    plt.title('Comparison of Smoothed Metric Trends Over Steps')
    plt.legend()
    plt.grid(True)
    # plt.show()
    for fmt in save_formats:
        if not os.path.exists('results/plots'):
            os.makedirs('results/plots')
        plt.savefig(f'results/plots/{plot_title}.{fmt}', format=fmt)


# List of file paths
import os
file_paths = [
    'results/yahoo_hr_20.csv',
    'results/yahoo_hr_10.csv',
    'results/yahoo_hr_5.csv',
    # 'results/yahoo_ndgc_20.csv',
    # 'results/yahoo_ndgc_10.csv',
    # 'results/yahoo_ndgc_5.csv',
    # 'results/total_loss.csv',
    # 'results/contrast_loss.csv',
]

# Corresponding labels for each file
labels = [
    'HR@20',
    'HR@10',
    'HR@5',
    # 'NDCG@20',
    # 'NDCG@10',
    # 'NDCG@5',
    # 'Total Loss',
    # 'Contrastive Loss',
]

# Call the function with the list of file paths and labels
plot_and_save(file_paths, labels, save_formats=['pdf', 'svg', 'jpg'], plot_title='HR')
