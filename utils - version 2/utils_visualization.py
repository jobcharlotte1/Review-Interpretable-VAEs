import numpy as np
import umap


def plot_umap(embedding:np.array, y_labels, name_labels):
    umap_df = pd.DataFrame({'UMAP-1':embedding[:,0], 'UMAP-2':embedding[:,1],
                        name_labels:y_labels})
    plt.figure(figsize=[7,7])
    sns.scatterplot(x='UMAP-1', y='UMAP-2', hue=name_labels, data=umap_df,
                    linewidth=0, alpha=0.7, s=5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=20, frameon=False, markerscale=2)
    plt.xlabel('UMAP-1', fontsize=20)
    plt.ylabel('UMAP-2', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()