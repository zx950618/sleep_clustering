from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from utils import save_analysis_to_xlsx, sample_day_to_sample_people

sample_point_count = 178
feature_names = ["TST", "SOL", "SE", "WASO", "AR", "N3", "N12", "REM", "Total SW", "N3 SW", "Total SP", "N12 SP",
                 "avg_n12_delta", "avg_n12_beta", "avg_n12_alpha", "avg_n3_delta", "avg_n3_beta", "avg_n3_alpha",
                 "n3_delta", "n3_theta", "n3_alpha", "n3_spindle", "n3_beta1", "n3_beta2", "n3_gamma", "n12_delta",
                 "n12_theta", "n12_alpha", "n12_spindle", "n12_beta1", "n12_beta2", "n12_gamma", "rem_delta",
                 "rem_theta", "rem_alpha", "rem_spindle", "rem_beta1", "rem_beta2", "rem_gamma"]
feature_count = feature_names.__len__()
n_clusters = 2
show_legend = False
show_sample_point_name = True



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dist_dir = "./results/cluster_origin_feature_by_people/trail_{}".format(timestamp)
os.mkdir(dist_dir)
save_path = dist_dir



data = pd.read_csv("./data/data_for_CAE.csv")
index = data.iloc[:, 0]
phone = data.iloc[:, 1]
sources = data.iloc[:, 3].values

data = data.iloc[:, 4:4+feature_count]

sample_people, sample_source, sample_phone = sample_day_to_sample_people(data, index, phone, sources)


df = DataFrame(sample_people)

for i in range(sample_people.shape[1]):
    feature_min = np.min(sample_people[:, i])
    feature_max = np.max(sample_people[:, i])
    for j in range(sample_people.shape[0]):
        sample_people[j, i] = (sample_people[j, i] - feature_min) / (feature_max - feature_min)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=50, n_iter=1000)
embedded_data = tsne.fit_transform(sample_people)


# 计算Pearson相关系数
pearson_corr = {
    "1st Dimension": np.zeros(feature_count),
    "2nd Dimension": np.zeros(feature_count),
}
pearson_p = {
    "1st Dimension": np.zeros(feature_count),
    "2nd Dimension": np.zeros(feature_count),
}
for i in range(len(feature_names)):
    correlation_1, p_1 = pearsonr(sample_people[:, i].flatten(), embedded_data[:, 0].flatten())
    pearson_corr["1st Dimension"][i] = correlation_1
    pearson_p["1st Dimension"][i] = p_1


for i in range(len(feature_names)):
    correlation_2, p_2 = pearsonr(sample_people[:, i].flatten(), embedded_data[:, 1].flatten())
    pearson_corr["2nd Dimension"][i] = correlation_2
    pearson_p["2nd Dimension"][i] = p_2


pearson = {"corr": pearson_corr, "p": pearson_p}

# 计算Spearman相关系数
spearman_corr = {
    "1st Dimension": np.zeros(feature_count),
    "2nd Dimension": np.zeros(feature_count),
}
spearman_p = {
    "1st Dimension": np.zeros(feature_count),
    "2nd Dimension": np.zeros(feature_count),
}
for i in range(len(feature_names)):
    correlation_1, p_1 = spearmanr(sample_people[:, i].flatten(), embedded_data[:, 0].flatten())
    spearman_corr["1st Dimension"][i] = correlation_1
    spearman_p["1st Dimension"][i] = p_1


for i in range(len(feature_names)):
    correlation_2, p_2 = spearmanr(sample_people[:, i].flatten(), embedded_data[:, 1].flatten())
    spearman_corr["2nd Dimension"][i] = correlation_2
    spearman_p["2nd Dimension"][i] = p_2


spearman = {"corr": spearman_corr, "p": spearman_p}





# 创建KMeans对象并指定聚类数量
clustering = KMeans(n_clusters=n_clusters)

# 对降维后的数据进行聚类
clustering.fit(embedded_data)

# 获取聚类结果
labels = clustering.labels_

colors = ['blue', 'red', 'green', 'pink']

center_points = {}
for i in range(n_clusters):
    xi_0 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sample_source == 0)[0])]
    xi_1 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sample_source == 1)[0])]
    xi_2 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sample_source == 2)[0])]

    yi_0 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sample_source == 0)[0])]
    yi_1 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sample_source == 1)[0])]
    yi_2 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sample_source == 2)[0])]
    plt.scatter(xi_0, yi_0, marker='o', color=colors[i], label='C{}-insomnia'.format(i))
    plt.scatter(xi_1, yi_1, marker='*', color=colors[i], label='C{}-normal'.format(i))
    plt.scatter(xi_2, yi_2, marker='s', color=colors[i], label='C{}-students'.format(i))

    cluster_i_points = np.squeeze(sample_people)[np.where(labels == i)[0]]
    cluster_center = np.mean(cluster_i_points, axis=0)
    center_points["Cluster {}".format(i)] = cluster_center


save_analysis_to_xlsx(save_path, feature_names, pearson, spearman, center_points)


plt.title('{} Clusters with TSNE(2 Dimension)'.format(n_clusters))
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
if show_legend:
    plt.legend()

if show_sample_point_name:
    x = np.squeeze(embedded_data[:, 0])
    y = np.squeeze(embedded_data[:, 1])
    data_name = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 50, 51, 52, 53, 54, 55, 56,
                 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    for i in range(len(x)):
        plt.annotate("{}".format(data_name[i]), (x[i], y[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')


plt.savefig(save_path+"/cluster_fig.png", dpi=300)
plt.show()

