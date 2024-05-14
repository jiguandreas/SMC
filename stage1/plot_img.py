import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 正常文本显示
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def plot_rf_learning_curve(n_estimators_list, train_scores, test_scores, filename='RF_learning_curve.png'):
    plt.figure()
    plt.title("随机森林学习曲线")
    plt.xlabel("决策树数量")
    plt.ylabel("模型精度")

    plt.plot(n_estimators_list, train_scores, 'o-', color="r", label="训练集精度")
    plt.plot(n_estimators_list, test_scores, 'o-', color="g", label="验证集精度")

    plt.legend(loc="best")
    plt.grid()
    plt.savefig(filename)
    plt.show()


def rf_learning_curve(data, target_column='Energy_Consumption', filename='RF_learning_curve.png'):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    n_estimators_list = np.arange(10, 101, 10)
    train_scores = []
    test_scores = []

    for n_estimators in n_estimators_list:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(input_train, output_train)

        train_predictions = model.predict(input_train)
        test_predictions = model.predict(input_test)

        train_r2 = r2_score(output_train, train_predictions)
        test_r2 = r2_score(output_test, test_predictions)

        train_scores.append(train_r2)
        test_scores.append(test_r2)

    plot_rf_learning_curve(n_estimators_list, train_scores, test_scores, filename)


def plot_feature_importance(model,filename):
    plt.figure(figsize=(10, 6))
    ax = xgb.plot_importance(model, max_num_features=15)
    ax.set_xlabel('特征变量')
    ax.set_ylabel('重要程度')
    plt.title("XGB特征变量重要性")
    plt.savefig(filename)
    plt.show()


def plot_predictions_vs_actuals(predictions, actuals, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.3)
    plt.xlabel("实际值")
    plt.ylabel("预测值")
    plt.title("预测值与实际值对比图")
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red')
    plt.savefig(filename)
    plt.show()


def plot_residuals(predictions, actuals,filename):
    residuals = actuals - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, residuals, alpha=0.3)
    plt.xlabel("实际值")
    plt.ylabel("残差")
    plt.title("残差图")
    plt.axhline(y=0, color='red', linestyle='--')
    plt.savefig(filename)
    plt.show()
