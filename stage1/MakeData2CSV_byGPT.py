import pandas as pd
import numpy as np

np.random.seed(42)


def define_feature(rows=10000, float_precision=4):
    common_factor = np.random.normal(0, 0.5, rows)

    feature_dict = {
        'Temperature': np.round(20 + 7 * common_factor + np.random.normal(0, 3, rows), float_precision),
        'Speed': np.round(75 + 15 * common_factor + np.random.uniform(-15, 15, rows), float_precision),
        'Load': np.round(300 + 100 * common_factor + np.random.uniform(-40, 40, rows), float_precision),
        'Ac_Use': np.random.binomial(1, 0.5 + 0.05 * common_factor),
        'Terrain_Slope': np.round(5 * common_factor + np.random.uniform(-5, 5, rows), float_precision),
        'Wind_Speed': np.round(25 + 10 * common_factor + np.random.normal(0, 8, rows), float_precision),
        'Driving_Mode': np.random.choice([0, 1, 2], rows),
        'Weather_Condition': np.random.choice([0, 1, 2], rows),
        'Vehicle_Age': np.round(5 + 5 * common_factor + np.random.uniform(-2, 2, rows), float_precision),
        'Tire_Type': np.random.choice([0, 1, 2], rows),
        'Night_Driving': np.random.binomial(1, 0.5 + 0.05 * common_factor),
        'Urban_Driving': np.random.binomial(1, 0.5 + 0.05 * common_factor),
        'Highway_Driving': np.random.binomial(1, 0.5 + 0.05 * common_factor),
        'Average_Passengers': np.round(3 + 2 * common_factor + np.random.uniform(-1, 1, rows), float_precision),
        'Cruise_Control_Use': np.random.binomial(1, 0.5 + 0.05 * common_factor),
        'Regenerative_Braking_System': np.random.binomial(1, 0.5 + 0.05 * common_factor),
        'External_Humidity': np.round(50 + 50 * common_factor + np.random.normal(0, 15, rows), float_precision),
        'Road_Condition': np.random.choice([0, 1, 2, 3], rows),
        'Heat_Pump_System': np.random.binomial(1, 0.5 + 0.05 * common_factor),
        'In_Car_Device_Count': np.round(2.5 + 2 * common_factor + np.random.normal(0, 1, rows), float_precision),
        'Charging_Frequency': np.round(3.5 + 3.5 * common_factor + np.random.normal(0, 1, rows), float_precision),
        'Months_Since_Last_Maintenance': np.round(6 + 6 * common_factor + np.random.uniform(-1, 1, rows),
                                                  float_precision),
    }

    feature_dict['Energy_Consumption'] = np.round(
        feature_dict['Temperature'] * 0.05 +
        feature_dict['Speed'] * 0.02 +
        feature_dict['Load'] * 0.01 +
        feature_dict['Ac_Use'] * 5 +
        feature_dict['Terrain_Slope'] * 0.1 +
        feature_dict['Wind_Speed'] * 0.03 +
        feature_dict['Driving_Mode'] * 2 +
        feature_dict['Weather_Condition'] * 1 +
        feature_dict['Vehicle_Age'] * 0.05 +
        feature_dict['Tire_Type'] * 0.5 +
        feature_dict['Night_Driving'] * 2 +
        feature_dict['Urban_Driving'] * 3 +
        feature_dict['Highway_Driving'] * 1.5 +
        feature_dict['Average_Passengers'] * 0.1 +
        feature_dict['Cruise_Control_Use'] * 1 +
        feature_dict['Regenerative_Braking_System'] * -2 +
        feature_dict['External_Humidity'] * 0.01 +
        feature_dict['Road_Condition'] * 0.5 +
        feature_dict['Heat_Pump_System'] * 3 +
        feature_dict['In_Car_Device_Count'] * 0.5 +
        feature_dict['Charging_Frequency'] * -0.5 +
        feature_dict['Months_Since_Last_Maintenance'] * 0.2 +
        np.random.normal(0, 1, rows) * 0.5,  # 添加随机噪声
        float_precision
    )

    return feature_dict


if __name__ == '__main__':
    # 生成数据
    df = pd.DataFrame(define_feature(10000))
    print("shape of data: ", df.shape)

    # 保存到CSV文件
    df.to_csv('Electric_Vehicle_Trip_Energy_Consumption_Data.csv', index=False)
