import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import pickle
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

# 正常文本显示
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


# 为了能在测试数据上使用相同的步骤得到和训练数据上相同的测试结果，需要保存模型
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# KMO 检验
def kmo_test(df: DataFrame):
    kmo_all, kmo_model = calculate_kmo(df)
    return kmo_model


# Bartlett 球形检验
def bartlett_test(df: DataFrame):
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    return chi_square_value, p_value


def create_eigenvalue_table(pca):
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    eigenvalue_table = pd.DataFrame({
        '特征根': eigenvalues,
        '方差百分比': explained_variance_ratio * 100,
        '累积': cumulative_variance_ratio * 100
    }, index=[f'成分{i + 1}' for i in range(len(eigenvalues))])

    return eigenvalue_table


# 绘制碎石图
def plot_scree_plot(pca):
    eigenvalues = pca.explained_variance_
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
    for i, txt in enumerate(np.round(eigenvalues, 3)):
        plt.annotate(txt, (i + 1, eigenvalues[i]), textcoords="offset points", xytext=(0, 5), ha='center')
    plt.title('碎石图')
    plt.xlabel('因子个数')
    plt.ylabel('特征根')
    plt.grid()
    plt.show()


# 计算综合得分
def calculate_comprehensive_score(df, explained_variance_ratio):
    component_scores = df.iloc[:, :-1]
    weights = explained_variance_ratio / explained_variance_ratio.sum()
    df['综合得分'] = component_scores.dot(weights)
    return df


# 绘制综合得分表
def plot_comprehensive_scores(df):
    df = df.sort_values(by='综合得分', ascending=False)
    df.reset_index(inplace=True)
    df.rename(columns={'index': '行索引'}, inplace=True)
    df.index += 1
    df.index.name = '排名'
    print(df)
    return df


# 主成分分析，threshold: 阈值，解释方差比例的累积和大于等于该值时停止（保留多少数据）
def Principal_Component_Analysis(df: DataFrame, threshold=0.95,
                                 save=False, model_path='pca_model.pkl', scaler_path='scaler.pkl'):
    features = df.columns[:-1]
    x = df.loc[:, features].values
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    if save:
        save_model(scaler, scaler_path)

    pca = PCA(n_components=threshold)
    principal_components = pca.fit_transform(x_scaled)

    if save:
        save_model(pca, model_path)

    print("=" * 25 + " PCA " + "=" * 25)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

    eigenvalue_table = create_eigenvalue_table(pca)
    print(eigenvalue_table)

    # 绘制碎石图
    plot_scree_plot(pca)

    print("PCA components (features' contributions to components):")
    components_df = pd.DataFrame(pca.components_, columns=features,
                                 index=[f'component{i + 1}' for i in range(pca.n_components_)])
    print(components_df)

    principalDf = pd.DataFrame(data=principal_components,
                               columns=[f'component{i + 1}' for i in range(pca.n_components_)])
    finalDf = pd.concat([principalDf, df[['Energy_Consumption']]], axis=1)
    return finalDf, pca.explained_variance_ratio_


def apply_PCA_to_test_data(df: DataFrame, pca_model_path='pca_model.pkl', scaler_path='scaler.pkl'):
    features = df.columns[:-1]
    x = df.loc[:, features].values

    # Load the scaler and PCA model
    scaler = load_model(scaler_path)
    pca = load_model(pca_model_path)

    # Standardize and apply PCA transformation
    x_scaled = scaler.transform(x)
    principal_components = pca.transform(x_scaled)

    # Convert the principal components into a DataFrame
    principalDf = pd.DataFrame(data=principal_components,
                               columns=[f'component{i + 1}' for i in range(pca.n_components_)])
    return principalDf


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
# 会保存我们删除的因素，以便在预测数据上使用相同的因素
def correlation_analysis(df: DataFrame, method='spearman', threshold=0.1, save_path='significant_features.pkl'):
    correlation_matrix = df.corr(method=method)

    # 添加显著性检验
    p_values_matrix = pd.DataFrame(np.ones(correlation_matrix.shape), columns=correlation_matrix.columns,
                                   index=correlation_matrix.index)
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if method == 'pearson':
                corr, p_value = pearsonr(df[col], df[row])
            elif method == 'spearman':
                corr, p_value = spearmanr(df[col], df[row])
            elif method == 'kendall':
                corr, p_value = kendalltau(df[col], df[row])
            p_values_matrix.loc[row, col] = p_value

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Factors about Energy Consumption')
    plt.show()

    print("=" * 25 + " CORRELATION ANALYSIS " + "=" * 25)
    energy_corr = correlation_matrix['Energy_Consumption']
    significant_corr = energy_corr[energy_corr.abs() >= threshold]
    print(significant_corr.sort_values(ascending=False))

    print("=" * 25 + " SIGNIFICANCE TEST " + "=" * 25)
    energy_p_values = p_values_matrix['Energy_Consumption']
    significant_p_values = energy_p_values[energy_p_values < 0.05]
    print(significant_p_values.sort_values())

    columns_to_keep = energy_corr[energy_corr.abs() >= threshold].index
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(columns_to_keep, f)

    columns_to_drop = energy_corr[energy_corr.abs() < threshold].index
    df_dropped = df.drop(columns=columns_to_drop)
    return df_dropped


def apply_correlation_to_test_data(df: DataFrame, features_path='significant_features.pkl'):
    with open(features_path, 'rb') as f:
        significant_features = pickle.load(f)

    # Ensure 'Energy_Consumption' is not expected in test set
    significant_features = [feat for feat in significant_features if feat in df.columns]

    df_filtered = df.loc[:, significant_features]
    return df_filtered


def analysis_factors_main(csv_path, decomposition_method='PCA',
                          decomposition_threshold=0.9, correlation_method='spearman',
                          correlation_threshold=0.1, output_csv_name='temp_csv/output_factors.csv'):
    df = pd.read_csv(csv_path)

    # KMO和Bartlett检验
    kmo_value = kmo_test(df.iloc[:, :-1])
    bartlett_chi_square_value, bartlett_p_value = bartlett_test(df.iloc[:, :-1])
    print(f"KMO Test Value: {kmo_value}")
    print(f"Bartlett's Test Chi-Square: {bartlett_chi_square_value}, p-value: {bartlett_p_value}")

    if decomposition_method == 'PCA':
        # 保存PCA模型，这样在测试数据上也可以用相同的处理方法，防止处理的数据不一致导致在测试数据上得到不一样的处理结果
        decomposition_df, explained_variance_ratio = Principal_Component_Analysis(df, threshold=decomposition_threshold,
                                                                                  save=False,
                                                                                  model_path='weight_ckpt/pca_model.pkl',
                                                                                  scaler_path='weight_ckpt/scaler.pkl')

    elif decomposition_method == 'FA':
        decomposition_df = Factor_Analysis(df, threshold=decomposition_threshold)

    decomposition_df_copy = decomposition_df.copy()
    comprehensive_score_df = calculate_comprehensive_score(decomposition_df_copy, explained_variance_ratio)
    comprehensive_score_df = plot_comprehensive_scores(comprehensive_score_df)

    # CA_df = correlation_analysis(decomposition_df, method=correlation_method, threshold=correlation_threshold,
    #                              save_path='weight_ckpt/significant_features.pkl')
    CA_df = correlation_analysis(decomposition_df, method=correlation_method, threshold=correlation_threshold,
                                 save_path=None)
    CA_df.to_csv(output_csv_name, index=False)



def analysis_factors_pred(csv_path, output_csv_name='temp_csv/pred_input_factors.csv',
                          features_path='weight_ckpt/significant_features.pkl'):
    df = pd.read_csv(csv_path)
    transformed_df = apply_PCA_to_test_data(df, pca_model_path='weight_ckpt/pca_model.pkl',
                                            scaler_path='weight_ckpt/scaler.pkl')
    test_final_df = apply_correlation_to_test_data(transformed_df, features_path=features_path)
    test_final_df.to_csv(output_csv_name, index=False)


if __name__ == '__main__':
    train_csv_path = 'temp_csv/modified_file.csv'
    # correlation_analysis(pd.read_csv(csv_path))
    analysis_factors_main(train_csv_path, decomposition_method='PCA')

    # test_csv_path = 'temp_csv/pred_input_sample.csv'
    # analysis_factors_pred(test_csv_path, output_csv_name='temp_csv/pred_input_factors.csv')
