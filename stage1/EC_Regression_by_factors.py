import numpy as np
import pandas as pd
from pyswarm import pso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deap import base, creator, tools, algorithms
import pickle
import xgboost as xgb
import lightgbm as lgb
from stage1.Analysis_factors import analysis_factors_pred
from stage1.plot_img import rf_learning_curve, plot_feature_importance, plot_predictions_vs_actuals, plot_residuals

"""
    随机森林回归，使用贝叶斯优化调参
    除了SVR每种模型我都用了至少一种优化算法，这个也可以吹，比如遗传算法、蚁群优化、贝叶斯优化等
    采用的集成学习方法：Random Forest 本身就是一个基于Bagging的集成模型，它通过从原始训练数据集中进行有放回的抽样（也称为自助采样法，bootstrap sampling）来创建多个决策树，
                    并通过投票（对于分类问题）或平均（对于回归问题）来合并这些树的预测结果，从而形成最终的输出。
                    Stacking：RF由于其出色的准确性和对数据细节的捕捉能力，常被用作Stacking集成中的一个基模型。
"""


def train_Random_Forest(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/random_forest_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        model.set_params(**loaded_params)
        model.fit(input_train, output_train)
        return model
    else:
        print("Training Random Forest model Using Bayes...")
        search_spaces = {
            'n_estimators': Integer(100, 200),
            'max_features': Categorical(['log2', 'sqrt']),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(4, 10),
            'min_samples_leaf': Integer(2, 10)
        }
        opt = BayesSearchCV(estimator=model,
                            search_spaces=search_spaces,
                            n_iter=50,
                            cv=3,
                            scoring='neg_mean_squared_error',
                            verbose=2,
                            random_state=42,
                            n_jobs=1)
        opt.fit(input_train, output_train)
        best_model = opt.best_estimator_

        predictions = best_model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Optimized Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        model_params = best_model.get_params()
        with open('weight_ckpt/random_forest_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

        return best_model


"""
    这里写的XGBoost模型不仅已经通过Boosting技术实现了集成学习，还通过粒子群优化进行了超参数的优化
    集成学习方法：XGBoost 就是一种基于Boosting技术的实现，实现了Gradient Boosting，
                它通过构建一系列的决策树来优化预测模型。这些决策树是顺序生成的，其中每一棵树都试图纠正前一棵树的预测错误。
                GBoost 由于其高效的性能和优秀的预测能力，常被用作Stacking集成中的一个基模型。
    模型本身已经是哪一种集成学习方法，或者模型需要用哪一种集成学习方法的原因可以询问GPT
"""


def train_XGBoost(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/xgboost_params.pkl', 'rb') as f:
            loaded_params = pickle.load(f)
        model = xgb.XGBRegressor(**loaded_params)
        model.fit(input_train, output_train)
        return model
    else:
        print("Training XGBoost model Using PSO...")

        def pso_objective(params):
            max_depth, learning_rate, alpha = int(params[0]), params[1], params[2]
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                colsample_bytree=0.4,
                learning_rate=learning_rate,
                max_depth=max_depth,
                alpha=alpha,
                n_estimators=200,
                random_state=42
            )
            model.fit(input_train, output_train)
            predictions = model.predict(input_test)
            mse = mean_squared_error(output_test, predictions)
            return mse

        lb = [1, 0.01, 1]
        ub = [10, 0.5, 100]
        best_params, _ = pso(pso_objective, lb, ub, swarmsize=50, maxiter=100, minstep=1e-8, minfunc=1e-8, debug=True)

        model_params = {
            'objective': 'reg:squarederror',
            'colsample_bytree': 0.4,
            'learning_rate': best_params[1],
            'max_depth': int(best_params[0]),
            'alpha': best_params[2],
            'n_estimators': 200,
            'random_state': 42
        }

        model = xgb.XGBRegressor(**model_params)
        model.fit(input_train, output_train)

        with open('weight_ckpt/xgboost_params.pkl', 'wb') as f:
            pickle.dump(model_params, f)

        predictions = model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Optimized Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        return model


"""
    决策树回归，用遗传算法调参
    Bagging：单一决策树往往容易过拟合，但可以通过Bagging技术，例如在Random Forest中使用，来提高其稳定性和准确性。
    作为基模型，决策树可以为Stacking提供深入的数据分割点。
"""


def train_Decision_Tree(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/decision_tree_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
        model = BaggingRegressor(base_estimator=DecisionTreeRegressor(**best_params), n_estimators=10, random_state=42)
        model.fit(input_train, output_train)
        predictions = model.predict(input_test)
        return model, output_test, predictions
    else:
        print("Training Decision Tree model Using GA...")

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attr_max_depth", np.random.randint, 1, 20)
        toolbox.register("attr_min_samples_split", np.random.randint, 2, 20)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_max_depth, toolbox.attr_min_samples_split), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evalModel(individual):
            model = DecisionTreeRegressor(max_depth=individual[0], min_samples_split=individual[1], random_state=42)
            model.fit(input_train, output_train)
            predictions = model.predict(input_test)
            mse = mean_squared_error(output_test, predictions)
            return (mse,)

        toolbox.register("evaluate", evalModel)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[1, 2], up=[20, 20], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=50)
        NGEN = 40
        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))

        best_ind = tools.selBest(population, 1)[0]
        best_params = {'max_depth': best_ind[0], 'min_samples_split': best_ind[1]}
        print("Best individual is %s, with MSE = %f" % (best_ind, best_ind.fitness.values[0]))

        model = BaggingRegressor(base_estimator=DecisionTreeRegressor(**best_params), n_estimators=10, random_state=42)
        model.fit(input_train, output_train)

        with open('weight_ckpt/decision_tree_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)

        predictions = model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Bagging Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        return model, output_test, predictions


"""
    LightGBM回归，是基于决策树的，使用贝叶斯优化调参
    Boosting：LightGBM 是一个基于Boosting技术的高效梯度提升框架，已经实现了Boosting，特别适合处理大规模数据。
    Stacking：由于其计算效率高和处理大数据的能力，LightGBM可以作为Stacking集成的强力基模型
"""


def train_LGBM(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/lightgbm_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        model.set_params(**loaded_params)
        model.fit(input_train, output_train)
        predictions = model.predict(input_test)
        return model, output_test, predictions
    else:
        print("Training LightGBM model Using Bayesian Optimization...")
        search_spaces = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 15),
            'learning_rate': Real(0.01, 0.3),
            'num_leaves': Integer(20, 150),
            'colsample_bytree': Real(0.1, 1.0)
        }

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces,
            n_iter=30,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=2,
            random_state=42,
            n_jobs=1
        )

        opt.fit(input_train, output_train)
        best_model = opt.best_estimator_

        predictions = best_model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Optimized Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        model_params = best_model.get_params()
        with open('weight_ckpt/lightgbm_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

        return best_model, output_test, predictions


"""
    支持向量机回归
    Bagging：虽然SVR通常不用于Bagging，但可以通过创建多个SVR模型，每个模型训练数据的子样本，来提高其泛化能力。
    Stacking：SVR可以作为一个回归任务中的基模型，其预测结果可以作为Stacking中的一个特征输入。
"""


def train_SVR(data, target_column='Energy_Consumption', pretrained=True):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

    # 特征缩放
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    bagging_model = BaggingRegressor(base_estimator=svr_model, n_estimators=10, random_state=42)
    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/svr_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)

        if 'estimator' in loaded_params:
            loaded_params['base_estimator'] = loaded_params.pop('estimator')

        bagging_model.set_params(**loaded_params)
        bagging_model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = bagging_model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return bagging_model, y_pred, y_test
    else:
        bagging_model.fit(X_train_scaled, y_train_scaled)
        model_params = bagging_model.get_params()
        with open('weight_ckpt/svr_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

    y_pred_scaled = bagging_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R²: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    return svr_model, y_pred, y_test


"""
    这就是将前面的所有模型作为基学习器，并会使用线性回归作为最终的学习器，进行Stacking集成学习的模型
    这里面用到的模型都要先预训练好，能够在权重文件夹中存在相应的权重文件，否则加载不了权重会报错
"""


def train_stacking_model(data, target_column='Energy_Consumption', pretrained=True):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

    # 加载预训练好的模型
    rf_model = train_Random_Forest(data, target_column, pretrained=True)
    xgb_model = train_XGBoost(data, target_column, pretrained=True)
    dt_model = train_Decision_Tree(data, target_column, pretrained=True)[0]
    lgbm_model = train_LGBM(data, target_column, pretrained=True)[0]
    svr_model = train_SVR(data, target_column, pretrained=True)[0]

    estimators = [
        ('random_forest', rf_model),
        ('xgboost', xgb_model),
        ('decision_tree', dt_model),
        ('lightgbm', lgbm_model),
        ('svr', svr_model)
    ]

    # Stacking Regressor
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/stacking_model_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        stacking_regressor.set_params(**loaded_params)
        stacking_regressor.fit(X_train, y_train)
        y_pred = stacking_regressor.predict(X_test)
        return stacking_regressor, y_pred, y_test
    else:
        # 训练Stacking模型（这里仅用于重新调整final_estimator）
        stacking_regressor.fit(X_train, y_train)
        model_params = stacking_regressor.get_params()
        with open('weight_ckpt/stacking_model_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

    y_pred = stacking_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Stacking Model R²: {r2}")
    print(f"Stacking Model Mean Squared Error: {mse}")
    print(f"Stacking Model Mean Absolute Error (MAE): {mae}")

    return stacking_regressor, y_pred, y_test


"""
    train_model这份函数的作用是训练模型，method是你想用哪种模型去拟合csv_path对应的训练集，pretrained是指是否使用预训练的模型参数
    如果你是第一次训练，就把pretrained这个参数设置为False，这时就会从头开始训练一个模型，并且把参数保存到weight_ckpt文件夹中，再返回这个训练好的模型即train_model返回的model;
    如果你之前有训练过，就可以把pretrained这个参数设置为True，这时就不会从头开始训练，而是直接从weight_ckpt文件夹中找到你训练好的权重，加载训练好的模型，直接返回model。
    只有训练的时候会有图返回
"""


def train_model(csv_path, method='Random_Forest', pretrained=False):
    print('=' * 25 + "Training Model Start..." + '=' * 25)
    df = pd.read_csv(csv_path)
    if method == 'Random_Forest':
        model = train_Random_Forest(df, target_column='Energy_Consumption', pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        rf_learning_curve(df, target_column='Energy_Consumption', filename='output_img/RF_learning_curve.png')

    elif method == 'Decision_Tree':
        model, output_test, pred = train_Decision_Tree(df, target_column='Energy_Consumption', pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_predictions_vs_actuals(pred, output_test, filename='output_img/Decision_Tree_predictions_vs_actuals.png')

    elif method == 'XGBoost':
        model = train_XGBoost(df, target_column='Energy_Consumption', pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_feature_importance(model, filename='output_img/XGBoost_feature_importance.png')

    elif method == 'LGBM':
        model, output_test, pred = train_LGBM(df, target_column='Energy_Consumption', pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_predictions_vs_actuals(pred, output_test, filename='output_img/LGBM_predictions_vs_actuals.png')

    elif method == 'SVR':
        model, y_pred, y_test = train_SVR(df, target_column='Energy_Consumption', pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_residuals(y_pred, y_test, filename='output_img/SVR_residuals.png')

    elif method == 'stacking_model':
        model, y_pred, y_test = train_stacking_model(df, target_column='Energy_Consumption', pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_predictions_vs_actuals(y_pred, y_test, filename='output_img/Stacking_predictions_vs_actuals.png')

    return model


"""
    predict这份函数的作用就是直接用来推理（predict）
    你需要给定一个预测的数据文件，里面的内容和和训练数据的csv（参考origin_data）结构相同，每个特征都要有，但是最后的预测数据Y这一列不需要，因为这是我们要预测的
"""


def predict(need_pred_csv_path, output_csv_path='temp_csv/pred_input_factors.csv', model=None):
    print('=' * 25 + "Energy Consumption Predict Start..." + '=' * 25)
    analysis_factors_pred(need_pred_csv_path, output_csv_name=output_csv_path,
                          features_path='weight_ckpt/significant_features.pkl')
    X = pd.read_csv(output_csv_path)
    pred_output = model.predict(X)
    print(pred_output)
    return pred_output


if __name__ == '__main__':
    # 注意读一下我写的英文内容，搞清楚怎么运行这份代码以及内容~
    # 集成学习中，Random Forest、决策树、SVR用了Bagging，XGBoost、LightGBM用了Boosting。然后它们都作为Stacking的基模型。
    # 仿照论文，我给出了三种评价指标R²，MSE，MAE

    # the dataset(csv) path you want to use it as train dataset and eval dataset
    train_dataset_path = 'temp_csv/output_factors.csv'
    model = train_model(train_dataset_path, method='stacking_model', pretrained=True)

    # the dataset(csv) path you want to use it as test dataset
    # you will only use testset's X data(The columns except Energy consumption) to predict result Y(Energy consumption)
    pred_path = 'temp_csv/pred_input_sample.csv'
    output = predict(pred_path, output_csv_path='temp_csv/pred_input_factors.csv', model=model)

"""
以下是我在训练的时候得到的结果：
    Random_Forest【R²: 0.968359836073141,Optimized Mean Squared Error(MSE): 5.144784083798369e-07,Mean Absolute Error(MAE): 0.00038028536404142874】
    Decision Tree【R²: 0.9611198728161638,Bagging Mean Squared Error: 6.322023488053221e-07,Mean Absolute Error (MAE): 0.00046211754324688477,Best individual is [19, 7], with MSE = 0.000001;】
    XGBoost【R²: 0.8800249913045917,Optimized Mean Squared Error: 1.9508290684478242e-06,Mean Absolute Error (MAE): 0.0008903522397127161】
    LGBM【R²: 0.9867383890941666,Optimized Mean Squared Error: 2.156377093101624e-07,Mean Absolute Error (MAE): 0.000290007131253444】
    SVR【R²: 0.9811158478565183,Mean Squared Error: 3.0706189009766096e-07,Mean Absolute Error (MAE): 0.00026251274825174166】
    Stacking Model【R²: 0.9874055942140146,Mean Squared Error: 2.0478875704443237e-07,Mean Absolute Error (MAE): 0.0002928205593187147】
"""
