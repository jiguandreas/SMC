import numpy as np
import pandas as pd
from pyswarm import pso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import pickle
import xgboost as xgb
import lightgbm as lgb


# 这份代码我用函数去写的，使用了多种模型和多种调优方法，但是正确的做法应该是写到一个类中的，这样可以把模型和调优方法分开，方便后续的扩展和维护
# data包括了特征列和目标列，target_column为目标列名，pretrained为True时加载模型已经训练好的参数并返回模型，False时重新训练

# 随机森林回归，使用贝叶斯优化调参
def train_Random_Forest(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)

    if pretrained:
        with open('./random_forest_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        model.set_params(**loaded_params)
        return model
    else:
        print("Training Random Forest model Using Bayes...")
        search_spaces = {
            'n_estimators': Integer(100, 200),
            'max_features': Categorical(['auto', 'sqrt']),
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
        mse = mean_squared_error(output_test, predictions)
        # Mean Squared Error: 1.0087914701055724e-06
        print(f"Optimized Mean Squared Error: {mse}")

        model_params = best_model.get_params()
        with open('./random_forest_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

        return best_model


# XGBoost回归，使用粒子群优化调参
def train_XGBoost(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    if pretrained:
        with open('xgboost_params.pkl', 'rb') as f:
            loaded_params = pickle.load(f)
        model = xgb.XGBRegressor(**loaded_params)
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

        with open('xgboost_params.pkl', 'wb') as f:
            pickle.dump(model_params, f)

        predictions = model.predict(input_test)
        mse = mean_squared_error(output_test, predictions)
        # Optimized Mean Squared Error: 3.1033549277412507e-06
        print(f"Optimized Mean Squared Error: {mse}")

        return model


# 决策树回归，用遗传算法调参
def train_Decision_Tree(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    if pretrained:
        with open('decision_tree_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
        model = DecisionTreeRegressor(**best_params)
        return model
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

        with open('decision_tree_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)

        model = DecisionTreeRegressor(max_depth=best_ind[0], min_samples_split=best_ind[1], random_state=42)
        model.fit(input_train, output_train)
        return model


# LightGBM回归，是基于决策树的，使用贝叶斯优化调参
def train_LGBM(data, target_column='Energy_Consumption', pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(random_state=42)

    if pretrained:
        with open('./lightgbm_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        model.set_params(**loaded_params)
        return model
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
        mse = mean_squared_error(output_test, predictions)
        # Optimized Mean Squared Error: 5.137845177879988e-07
        print(f"Optimized Mean Squared Error: {mse}")

        model_params = best_model.get_params()
        with open('./lightgbm_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

        return best_model


# 支持向量机回归
def train_SVR(data, target_column='Energy_Consumption'):
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
    svr_model.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = svr_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return svr_model


def regression_main(csv_path, method='Random_Forest', pretrained=False):
    df = pd.read_csv(csv_path)
    if method == 'Random_Forest':
        model = train_Random_Forest(df, target_column='Energy_Consumption', pretrained=pretrained)
    elif method == 'Decision_Tree':
        model = train_Decision_Tree(df, target_column='Energy_Consumption', pretrained=pretrained)
    elif method == 'XGBoost':
        model = train_XGBoost(df, target_column='Energy_Consumption', pretrained=pretrained)
    elif method == 'LGBM':
        model = train_LGBM(df, target_column='Energy_Consumption', pretrained=pretrained)
    elif method == 'SVR':
        model = train_SVR(df, target_column='Energy_Consumption')


if __name__ == '__main__':
    csv_path = './output_factors.csv'
    regression_main(csv_path, method='SVR', pretrained=True)
