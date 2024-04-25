import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


# 主成分分析，threshold: 阈值，解释方差比例的累积和大于等于该值时停止（保留多少数据）
def Principal_Component_Analysis(df: DataFrame, threshold=0.95):
    features = df.columns[:-1]
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=threshold)
    principal_components = pca.fit_transform(x)
    print("=" * 25 + " PCA " + "=" * 25)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

    print("PCA components (features' contributions to components):")
    components_df = pd.DataFrame(pca.components_, columns=features,
                                 index=[f'component{i + 1}' for i in range(pca.n_components_)])
    print(components_df)

    principalDf = pd.DataFrame(data=principal_components,
                               columns=[f'component{i + 1}' for i in range(pca.n_components_)])
    finalDf = pd.concat([principalDf, df[['Energy_Consumption']]], axis=1)
    return finalDf


# 因子分析；threshold: 阈值，解释方差比例的累积和大于等于该值时停止（保留多少数据）
def Factor_Analysis(df: DataFrame, threshold=0.4):
    features = df.columns[:-1]
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    n_factors = 1
    total_variance = 0
    total_components = len(features)

    while total_variance < threshold and n_factors < total_components:
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(x)
        explained_variance = np.sum(fa.noise_variance_)
        total_variance = 1 - explained_variance / np.var(x, axis=0).sum()
        print(f"Trying {n_factors} factors: Total Explained Variance = {total_variance}")
        if total_variance >= threshold:
            break
        n_factors += 1

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factors = fa.fit_transform(x)

    # 删除所有负荷量为0的因子
    non_zero_factors = np.any(fa.components_ != 0, axis=1)
    factors = factors[:, non_zero_factors]
    n_factors = np.sum(non_zero_factors)

    print("=" * 25 + " FACTOR ANALYSIS " + "=" * 25)
    loadings_df = pd.DataFrame(fa.components_[non_zero_factors], columns=features,
                               index=[f'factor{i + 1}' for i in range(n_factors)])
    print(loadings_df)

    factor_df = pd.DataFrame(data=factors, columns=[f'factor{i + 1}' for i in range(n_factors)])
    final_factor_df = pd.concat([factor_df, df[['Energy_Consumption']]], axis=1)
    return final_factor_df


# 相关性分析；method: pearson, kendall, spearman；threshold: 阈值，相关系数的绝对值小于该值的因素将被删除
def correlation_analysis(df: DataFrame, method='spearman', threshold=0.1):
    correlation_matrix = df.corr(method=method)

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Factors about Energy Consumption')
    plt.show()

    print("=" * 25 + " CORRELATION ANALYSIS " + "=" * 25)
    energy_corr = correlation_matrix['Energy_Consumption']
    significant_corr = energy_corr[energy_corr.abs() >= threshold]
    print(significant_corr.sort_values(ascending=False))

    columns_to_drop = energy_corr[energy_corr.abs() < threshold].index
    df_dropped = df.drop(columns=columns_to_drop)
    return df_dropped


def analysis_factors_main(csv_path, decomposition_method='PCA',
                          decomposition_threshold=0.9, correlation_method='spearman',
                          correlation_threshold=0.1, output_csv_name='output_factors.csv'):
    df = pd.read_csv(csv_path)
    if decomposition_method == 'PCA':
        decomposition_df = Principal_Component_Analysis(df, threshold=decomposition_threshold)
    elif decomposition_method == 'FA':
        decomposition_df = Factor_Analysis(df, threshold=decomposition_threshold)
    CA_df = correlation_analysis(decomposition_df, method=correlation_method, threshold=correlation_threshold)
    CA_df.to_csv(output_csv_name, index=False)


if __name__ == '__main__':
    csv_path = './modified_file.csv'
    correlation_analysis(pd.read_csv(csv_path))
    # analysis_factors_main(csv_path, decomposition_method='PCA')
