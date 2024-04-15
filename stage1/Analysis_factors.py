import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def Principal_Component_Analysis(df: DataFrame, n_components=3):
    features = df.columns[:-1]
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)
    print("=" * 25 + " PCA " + "=" * 25)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    print("PCA components (features' contributions to components):")
    components_df = pd.DataFrame(pca.components_, columns=features,
                                 index=[f'component{i + 1}' for i in range(n_components)])
    print(components_df)

    principalDf = pd.DataFrame(data=principal_components, columns=[f'component{i + 1}' for i in range(n_components)])
    finalDf = pd.concat([principalDf, df[['Energy_Consumption']]], axis=1)
    return finalDf


def correlation_analysis(df: DataFrame, method='spearman'):
    correlation_matrix = df.corr(method=method)

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Factors about Energy Consumption')
    plt.show()

    print("=" * 25 + " CORRELATION ANALYSIS " + "=" * 25)
    energy_corr = correlation_matrix['Energy_Consumption']
    significant_corr = energy_corr[energy_corr.abs() >= 0.1]
    print(significant_corr.sort_values(ascending=False))

    columns_to_drop = energy_corr[energy_corr.abs() < 0.1].index
    df_dropped = df.drop(columns=columns_to_drop)
    return df_dropped


if __name__ == '__main__':
    file_path = './Electric_Vehicle_Trip_Energy_Consumption_Data.csv'
    df = pd.read_csv(file_path)
    PCA_df = Principal_Component_Analysis(df, n_components=10)
    CA_df = correlation_analysis(PCA_df)
