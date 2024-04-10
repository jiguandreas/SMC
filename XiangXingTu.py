import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv("D:\数模\统计大赛\数据\Electric Vehicle Trip Energy Consumption Data.csv")

# 指定你需要检查的列
columns_to_check = ['Trip Energy Consumption', 'Trip Distance', 'Speed','Current','Total Voltage','Trip Time Length']

# 绘制箱线图
plt.figure(figsize=(10, 6))
data.boxplot(column=columns_to_check)
plt.title('Boxplot of Your Data')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
