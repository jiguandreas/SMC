import pandas as pd
import numpy as np

# random seed
np.random.seed(42)


def define_feature(rows=10000, float_precision=4):
    feature_dict = {
        'Temperature': np.round(np.random.uniform(-5, 35, rows), float_precision),
        'Speed': np.round(np.random.uniform(30, 120, rows), float_precision),
        'Load': np.round(np.random.uniform(100, 500, rows), float_precision),
        'Ac_Use': np.random.choice([0, 1], rows),
        'Terrain_Slope': np.round(np.random.uniform(-10, 10, rows), float_precision),
        'Wind_Speed': np.round(np.random.uniform(0, 50, rows), float_precision),
        'Driving_Mode': np.random.choice([0, 1, 2], rows),
        'Weather_Condition': np.random.choice([0, 1, 2], rows),
        'Vehicle_Age': np.round(np.random.uniform(0, 10, rows), float_precision),
        'Tire_Type': np.random.choice([0, 1, 2], rows),
        'Night_Driving': np.random.choice([0, 1], rows),
        'Urban_Driving': np.random.choice([0, 1], rows),
        'Highway_Driving': np.random.choice([0, 1], rows),
        'Average_Passengers': np.round(np.random.uniform(1, 5, rows), float_precision),
        'Cruise_Control_Use': np.random.choice([0, 1], rows),
        'Regenerative_Braking_System': np.random.choice([0, 1], rows),
        'External_Humidity': np.round(np.random.uniform(0, 100, rows), float_precision),
        'Road_Condition': np.random.choice([0, 1, 2, 3], rows),
        'Heat_Pump_System': np.random.choice([0, 1], rows),
        'In_Car_Device_Count': np.random.randint(0, 5, rows),
        'Charging_Frequency': np.round(np.random.uniform(0, 7, rows), float_precision),
        'Months_Since_Last_Maintenance': np.round(np.random.uniform(0, 12, rows), float_precision),
    }

    feature_dict['Energy_Consumption'] = np.round(
        1.5 * feature_dict['Temperature'] * 0.05 +  # 温度的影响
        feature_dict['Speed'] * 0.02 +  # 车速的影响
        feature_dict['Load'] * 0.01 +  # 载重的影响
        feature_dict['Ac_Use'] * 5 +  # 空调使用的影响
        feature_dict['Terrain_Slope'] * 0.1 +  # 地形坡度的影响
        feature_dict['Wind_Speed'] * 0.03 +  # 风速的影响
        feature_dict['Driving_Mode'] * 2 +  # 驾驶模式的影响
        feature_dict['Weather_Condition'] * 1 +  # 天气条件的影响
        feature_dict['Vehicle_Age'] * 0.05 +  # 车龄的影响
        feature_dict['Tire_Type'] * 0.5 +  # 轮胎类型的影响
        feature_dict['Night_Driving'] * 2 +  # 夜间行驶的影响
        feature_dict['Urban_Driving'] * 3 +  # 城市行驶的影响
        feature_dict['Highway_Driving'] * 1.5 +  # 高速行驶的影响
        feature_dict['Average_Passengers'] * 0.1 +  # 平均乘客数量的影响
        feature_dict['Cruise_Control_Use'] * 1 +  # 定速巡航的使用影响
        feature_dict['Regenerative_Braking_System'] * -2 +  # 再生制动系统的影响
        feature_dict['External_Humidity'] * 0.01 +  # 外部湿度的影响
        feature_dict['Road_Condition'] * 0.5 +  # 路面状况的影响
        feature_dict['Heat_Pump_System'] * 3 +  # 热泵系统的影响
        feature_dict['In_Car_Device_Count'] * 0.5 +  # 车内设备使用数量的影响
        feature_dict['Charging_Frequency'] * -0.5 +  # 充电频率的影响
        feature_dict['Months_Since_Last_Maintenance'] * 0.2 +  # 维护频率的影响
        np.random.normal(0, 1, rows) * 0.5,  # 添加随机噪声
        float_precision
    )
    return feature_dict


if __name__ == '__main__':
    # 生成数据
    df = pd.DataFrame(define_feature(10000))
    print("shape of data: ", df.shape)

    # 保存成CSV
    df.to_csv('Electric_Vehicle_Trip_Energy_Consumption_Data.csv', index=False)
