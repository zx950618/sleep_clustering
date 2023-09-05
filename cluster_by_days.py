import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
# from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


def sample_points_cluster(X, y, pca_components, n_clusters):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    y = y.to_numpy().astype(dtype=np.int)

    # 创建PCA对象并指定降维后的维度
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    # 创建KMeans对象并指定聚类数量
    clustering = AgglomerativeClustering(n_clusters=n_clusters)

    # 对降维后的数据进行聚类
    clustering.fit(X_reduced)

    # 获取聚类结果
    labels = clustering.labels_
    # centroids = kmeans.cluster_centers_

    if pca_components == 2:
        for i in range(n_clusters):
            xi = np.squeeze(X_reduced[:, 0])[np.where(y == i)]
            yi = np.squeeze(X_reduced[:, 1])[np.where(y == i)]
            plt.scatter(xi, yi, marker='o', label='Cluster {}'.format(i))

        plt.title('{} Clusters with PCA(2 Dimension)'.format(n_clusters))
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.legend()
        plt.show()

        pca1 = np.squeeze(X_reduced[:, 0])
        pca2 = np.squeeze(X_reduced[:, 1])

        feature_name = X.columns.to_numpy()
        for i in range(feature_name.__len__()):
            feature_i = X.values[:, i]
            corr, p = pearsonr(pca1, feature_i)
            print("{} with PCA1: corr = {}, p = {}".format(feature_name[i], corr, p))
            corr, p = pearsonr(pca2, feature_i)
            print("{} with PCA2: corr = {}, p = {}".format(feature_name[i], corr, p))

    return labels


def corr_analysis(x, x_name, y, y_name):
    a = np.array([x]).T

    b = y

    # correlation, p_value = pearsonr(a, b)
    # print("n3_time/n12_time vs scl_de: corr = {}, p = {}".format(correlation, p_value))

    # a = np.array([[1, 2],
    #               [2, 3],
    #               [3, 4],
    #               [4, 5]])
    # b = np.array([1,2,3,4])
    X = sm.add_constant(a)

    Y = b
    # 创建一个OLS模型
    model = sm.OLS(Y, X)

    # 拟合模型
    results = model.fit()

    print("p: {}, r2: {}".format(results.pvalues[1], results.rsquared))
    print(results.summary())

    fig, ax = plt.subplots()
    sns.set_style('white')
    color = 'b' if 1 else 'b'
    ax = sns.regplot(x=a, y=b, color=color, ax=ax)

    # x = range(-10, 800)
    # y = [i + 1 for i in x]
    #
    # plt.plot(x, y)

    ax.set_xlabel(x_name, fontsize=15)
    ax.set_ylabel(y_name, fontsize=15)
    # ax.set_xlim([0.2, 1.5])
    # ax.set_ylim([0, 20])
    ax.set_title("p: {}\nr2: {}".format(results.pvalues[1], results.rsquared), loc="right")
    plt.show()


def generate_dataset_with_name(data_path):
    df = pd.read_parquet(data_path)
    df = df.dropna(axis=0)
    cols_all = df.columns

    cols_n3 = cols_all[cols_all.str.startswith('n3')].tolist()
    cols_n12 = cols_all[cols_all.str.startswith('n12')].tolist()
    cols_rem = cols_all[cols_all.str.startswith('rem')].tolist()
    cols_demo = ['tst', 'sol', 'waso', 'se', 'sw_total', 'sw_n3', 'sp_total', 'sp_n12', 'age', 'sex']

    feature_columns = [] + cols_n12 + cols_rem + cols_n3 + cols_demo

    X = df[feature_columns]
    y = df['label']
    name = df['data_name']
    source = df['data_source']
    y.replace({0: '0', 1: '1'}, inplace=True)
    return X, y, name, source


if __name__ == '__main__':
    # 读取文件
    # sample_points, y, sample_points_name, source = generate_dataset_with_name(
    #     "E:/githome/insomnia_normal_divide/feature_data/data_v3.parquet")

    data = pd.read_csv("./results/insomnia_normal_CAE-fs1-fn30-es15decay_weight_75_latent.csv")
    # data = data.iloc[:44, :]
    data = data.reset_index(drop=True)

    index = data.iloc[:, 0]
    data = data.iloc[:, 2:]
    tsne = TSNE(n_components=2, perplexity=30)
    embedded_data = tsne.fit_transform(data)

    # print(embedded_data)

    # 创建KMeans对象并指定聚类数量
    clustering = AgglomerativeClustering(n_clusters=2, linkage='single')

    # 对降维后的数据进行聚类
    clustering.fit(embedded_data)

    # 获取聚类结果
    labels = clustering.labels_
    # centroids = kmeans.cluster_centers_
    sources = np.zeros(len(labels)).astype(dtype=np.int32)
    sources[44:] = 1
    colors = ['blue', 'red', 'green', 'pink']
    origin_sample = pd.read_csv("./data/data_for_CAE.csv")
    origin_sample = origin_sample.iloc[:, 3:]
    origin_sample = origin_sample.reset_index(drop=True).values
    for i in range(4):
        xi_0 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 0)[0])]
        xi_1 = np.squeeze(embedded_data[:, 0])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 1)[0])]

        yi_0 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 0)[0])]
        yi_1 = np.squeeze(embedded_data[:, 1])[np.intersect1d(np.where(labels == i)[0], np.where(sources == 1)[0])]
        plt.scatter(xi_0, yi_0, marker='o', color=colors[i], label='C{}-insomnia'.format(i))
        plt.scatter(xi_1, yi_1, marker='*', color=colors[i], label='C{}-normal'.format(i))

        # xi_0 = np.squeeze(embedded_data[:, 0])[np.where(labels == i)[0]]
        #
        # yi_0 = np.squeeze(embedded_data[:, 1])[np.where(labels == i)[0]]
        # plt.scatter(xi_0, yi_0, marker='o', color=colors[i], label='C{}'.format(i))

        cluster_i_points = np.squeeze(origin_sample)[np.where(labels == i)[0]]
        cluster_center = np.mean(cluster_i_points, axis=0)
        print("Cluster {} center point:\n {}".format(i + 1, cluster_center))


    plt.title('{} Clusters with TSNE(2 Dimension)'.format(4))
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    # plt.legend()


    x = np.squeeze(embedded_data[:, 0])
    y = np.squeeze(embedded_data[:, 1])
    # data_name = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
    for i in range(len(x)):
        plt.annotate("{}".format(index[i]), (x[i], y[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')


    plt.show()

    # # 降维和聚类
    # label = sample_points_cluster(sample_points, y, pca_components=None, n_clusters=2)
    #
    # sample_people_1 = sample_points[label == 0]
    # sample_people_2 = sample_points[label == 1]
    #
    # sample_people_cluster = [sample_people_1, sample_people_2]
    # feature_name = sample_points.columns.tolist()
    # for i in range(2):
    #     print("sample points mean and std in cluster {}".format(i + 1))
    #     for j in range(feature_name.__len__()):
    #         print("{}: {:.2f} ± {:.2f}".format(feature_name[j], np.mean(sample_people_cluster[i].values[:, j]),
    #                                            np.std(sample_people_cluster[i].values[:, j])))
    #
    # print("======================================================================")
    #
    # print("Sample point are clustered as:")
    # str_people = str(sample_points_name[0]) + "[" + str(label[0])
    # current_name = sample_points_name[0]
    # for i in range(1, len(sample_points)):
    #     if sample_points_name[i] == current_name:
    #         str_people = str_people + ", " + str(label[i])
    #     else:
    #         str_people = str_people + "]"
    #         print(str_people)
    #         str_people = str(sample_points_name[i]) + "[" + str(label[i])
    #         current_name = sample_points_name[i]
    # str_people = str_people + "," + str(label[-1]) + "]"
    # print(str_people)
    # # print("{} - {}: {}".format(sample_points_name[i], sample_points_dates[i], label[i]))
    #
    # print("======================================================================")
