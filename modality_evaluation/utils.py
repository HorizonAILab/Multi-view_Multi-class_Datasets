import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_multi_view_data(basic_path, dataset_name, normalization=None):
    ''' Load data from multiple views
    params:
    -------
    basic_path: str
        The basic path of the dataset.
    dataset_name: str
        The name of the dataset.
    normalization: str
        The normalization method. Default is None.

    return:
    -------
    X: list (len(X) = num_views)
        The data from multiple views.
    gnd: array, (num_samples, 1)
        The ground truth of the dataset.
    num_class: int
        The number of classes in the dataset.
    '''

    data = sio.loadmat(f"{basic_path}/{dataset_name}.mat")

    # data preprocessing
    scaler = None
    if normalization == 'MinMax':
        scaler = MinMaxScaler()
    elif normalization == 'Standard':
        scaler = StandardScaler()
    else:
        print("No normalization method is applied!")

    X = []
    for view in range(data["X"].shape[1]):
        if scaler is None:
            X.append(data["X"][0, view])
        else:
            X.append(scaler.fit_transform(data['X'][0, view]))

    gnd = data["gnd"]
    if gnd.min() == 1:
        gnd -= 1

    num_class = len(np.unique(gnd))

    return X, gnd, num_class


def plot_radar_chart(
    views, dataset_name, scores_dict, save_path=None
) -> None:
    ''' Plot radar chart for NMI scores of different views
    params:
    -------
    views: list
        The names of views.
    dataset_name: str
        The name of the dataset.
    scores_dict: dictionary
        {'metric1': [score1, score2, ...], 'metric2': [score1, score2, ...]}
    save_path: str
        The path to save the radar chart. Default is None.
    '''
    color_palette = ['#f07167', '#0081a7', '#00afb9', '#fed9b7', '#fdfcdc', '#FFFFFF']
    num_views = len(views)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, (metric, scores) in enumerate(scores_dict.items()):
        scores += scores[:1]
        ax.fill(angles, scores, color=color_palette[i], alpha=0.35, label=metric)
        ax.plot(angles, scores, color=color_palette[i], linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(views, fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', colors='black', labelsize=16)

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

    plt.title(f'{dataset_name}: {metric}', fontsize=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=14)
    plt.xticks(size=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            f'{save_path}/{dataset_name}.png',
            bbox_inches='tight',
            pad_inches=0,
            dpi=300
        )
    else:
        plt.show()