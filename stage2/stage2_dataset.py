import numpy as np
import pandas as pd

np.random.seed(42)

num_samples = 10000

# 增加的新因素
vehicle_types = np.random.choice(['Sedan', 'SUV', 'Truck'], p=[0.5, 0.4, 0.1], size=num_samples)
vehicle_age = np.random.randint(1, 11, size=num_samples)  # 车龄1到10年
charging_station_type = np.random.choice(['Home', 'Public', 'Fast'], p=[0.6, 0.3, 0.1], size=num_samples)
usage_frequency = np.random.normal(loc=300, scale=100, size=num_samples)  # 每月平均行驶里程
environmental_humidity = np.random.uniform(30, 80, size=num_samples)  # 环境湿度30%到80%
vehicle_load = np.random.choice(['Light', 'Normal', 'Heavy'], p=[0.2, 0.6, 0.2], size=num_samples)

# 原有因素
charging_power = np.random.choice([3.7, 7.4, 11, 22, 50, 120, 150], p=[0.1, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05], size=num_samples)
cable_length = np.clip(np.random.normal(loc=charging_power/30, scale=0.5), 1, 10)
temperature = np.random.normal(loc=15, scale=10, size=num_samples)
temperature = np.clip(temperature, -20, 40)
battery_capacity = np.random.choice([40, 60, 75, 85, 100], p=[0.1, 0.25, 0.3, 0.25, 0.1], size=num_samples)
thermal_management = np.random.binomial(1, p=(battery_capacity/150 + charging_power/150)/2, size=num_samples)

# 计算充电损耗比例
loss_percentage = 10 + 0.1 * (150 - charging_power)  # 功率高损失低
loss_percentage += 0.5 * (cable_length - 1)  # 电缆越短损失越小
loss_percentage += 0.2 * np.abs(temperature - 20)  # 离理想温度远损失高
loss_percentage -= 3 * thermal_management  # 热管理系统减少损失
loss_percentage += 0.05 * vehicle_age  # 车辆年龄越大，损失越高

# 创建DataFrame
data_enhanced = pd.DataFrame({
    'Vehicle Type': vehicle_types,
    'Vehicle Age': vehicle_age,
    'Charging Station Type': charging_station_type,
    'Usage Frequency (monthly km)': usage_frequency,
    'Environmental Humidity (%)': environmental_humidity,
    'Vehicle Load': vehicle_load,
    'Charging Power (kW)': charging_power,
    'Cable Length (m)': cable_length,
    'Temperature (Celsius)': temperature,
    'Battery Capacity (kWh)': battery_capacity,
    'Thermal Management System (0=no, 1=yes)': thermal_management,
    'Loss Percentage (%)': loss_percentage
})

data_enhanced.head()
file_path = './Charging_Losses_Dataset.csv'
data_enhanced.to_csv(file_path, index=False)