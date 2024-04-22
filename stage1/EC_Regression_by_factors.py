from scipy.stats import randint

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def train_Random_Forest(data, target_column='Energy_Consumption'):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)

    # !!!need to concert the parameter in param_dist
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10)
    }
    rs_model = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                  n_iter=100, cv=5, scoring='neg_mean_squared_error',
                                  verbose=2, random_state=42, n_jobs=1)
    rs_model.fit(input_train, output_train)
    best_model = rs_model.best_estimator_

    predictions = best_model.predict(input_test)
    mse = mean_squared_error(output_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return best_model


if __name__ == '__main__':
    csv_path = './Electric_Vehicle_Trip_Energy_Consumption_Data.csv'
    df = pd.read_csv(csv_path)
    model = train_Random_Forest(df, target_column='Energy_Consumption')

