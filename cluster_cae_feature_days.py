import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
# from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE

from utils import save_analysis_to_xlsx

sample_point_count = 178
origin_feature_names = ["TST", "SOL", "SE", "WASO", "AR", "N3", "N12", "REM", "Total SW", "N3 SW", "Total SP", "N12 SP",
                 "avg_n12_delta", "avg_n12_beta", "avg_n12_alpha", "avg_n3_delta", "avg_n3_beta", "avg_n3_alpha",
                 "n3_delta", "n3_theta", "n3_alpha", "n3_spindle", "n3_beta1", "n3_beta2", "n3_gamma", "n12_delta",
                 "n12_theta", "n12_alpha", "n12_spindle", "n12_beta1", "n12_beta2", "n12_gamma", "rem_delta",
                 "rem_theta", "rem_alpha", "rem_spindle", "rem_beta1", "rem_beta2", "rem_gamma"]
origin_feature_count = origin_feature_names.__len__()
n_clusters = 4
cluster_method = "KMeans"
# cluster_method = "AgglomerativeClustering"
show_legend = False
show_sample_point_name = True



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dist_dir = "./results/cluster_cae_feature_by_days/trail_{}".format(timestamp)
os.mkdir(dist_dir)
save_path = dist_dir



data = pd.read_csv("./results/insomnia_normal_student_CAE-fs1-fn30-es15decay_weight_75_latent.csv")
data = data.reset_index(drop=True)

index = data.iloc[:, 0]
sources = data.iloc[:, 2].values
data = data.iloc[:, 3:16]


tsne = TSNE(n_components=2, perplexity=30)
embedded_data = tsne.fit_transform(data)


origin_sample = pd.read_csv("./data/data_for_CAE.csv")
origin_sample = origin_sample.iloc[:, 4:4+origin_feature_count]
origin_sample = origin_sample.reset_index(drop=True).values


# 计算Pearson相关系数
pearson_corr = {
    "1st Dimension": np.zeros(origin_feature_count),
    "2nd Dimension": np.zeros(origin_feature_count),
}
pearson_p = {
    "1st Dimension": np.zeros(origin_feature_count),
    "2nd Dimension": np.zeros(origin_feature_count),
}
for i in range(len(origin_feature_names)):
    correlation_1, p_1 = pearsonr(origin_sample[:, i].flatten(), embedded_data[:, 0].flatten())
    pearson_corr["1st Dimension"][i] = correlation_1
    pearson_p["1st Dimension"][i] = p_1


for i in range(len(origin_feature_names)):
    correlation_2, p_2 = pearsonr(origin_sample[:, i].flatten(), embedded_data[:, 1].flatten())
    pearson_corr["2nd Dimension"][i] = correlation_2
    pearson_p["2nd Dimension"][i] = p_2

pearson = {"corr": pearson_corr, "p": pearson_p}

# 计算Spearman相关系数
spearman_corr = {
    "1st Dimension": np.zeros(origin_feature_count),
    "2nd Dimension": np.zeros(origin_feature_count),
}
spearman_p = {
    "1st Dimension": np.zeros(origin_feature_count),
    "2nd Dimension": np.zeros(origin_feature_count),
}
for i in range(len(origin_feature_names)):
    correlation_1, p_1 = spearmanr(origin_sample[:, i].flatten(), embedded_data[:, 0].flatten())
    spearman_corr["1st Dimension"][i] = correlation_1
    spearman_p["1st Dimension"][i] = p_1

for i in range(len(origin_feature_names)):
    correlation_2, p_2 = spearmanr(origin_sample[:, i].flatten(), embedded_data[:, 1].flatten())
    spearman_corr["2nd Dimension"][i] = correlation_2
    spearman_p["2nd Dimension"][i] = p_2

spearman = {"corr": spearman_corr, "p": spearman_p}



# 创建KMeans对象并指定聚类数量
if cluster_method == "AgglomerativeClustering":
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
elif cluster_method == 'KMeans':
    clustering = KMeans(n_clusters=n_clusters)

# 对降维后的数据进行聚类
clustering.fit(embedded_data)

# 获取聚类结果
labels = clustering.labels_
# centroids = kmeans.cluster_centers_

colors = ['blue', 'red', 'green', 'pink']
center_points = {}
for i in range(n_clusters):
    xi_0 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 0)[0])]
    xi_1 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 1)[0])]
    xi_2 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 2)[0])]

    yi_0 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 0)[0])]
    yi_1 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 1)[0])]
    yi_2 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 2)[0])]

    plt.scatter(xi_0, yi_0, marker='o', color=colors[i], label='C{}-insomnia'.format(i))
    plt.scatter(xi_1, yi_1, marker='*', color=colors[i], label='C{}-normal'.format(i))
    plt.scatter(xi_2, yi_2, marker='s', color=colors[i], label='C{}-students'.format(i))


    cluster_i_points = np.squeeze(origin_sample)[np.where(labels == i)[0]]
    cluster_center = np.mean(cluster_i_points, axis=0)
    center_points["Cluster {}".format(i)] = cluster_center

save_analysis_to_xlsx(save_path, origin_feature_names, pearson, spearman, center_points)


plt.title('{} Clusters with TSNE(2 Dimension)'.format(n_clusters))
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
if show_legend:
    plt.legend()


if show_sample_point_name:
    x = np.squeeze(embedded_data[:, 0])
    y = np.squeeze(embedded_data[:, 1])
    # data_name = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
    for i in range(len(x)):
        plt.annotate("{}".format(index[i]), (x[i], y[i]), textcoords="offset points", xytext=(0, 10), fontsize=6,
                     ha='center')

plt.savefig(save_path+"/cluster_fig.png", dpi=300)
plt.show()
