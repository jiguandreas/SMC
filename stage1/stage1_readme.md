1.数据集包含以下变量：
- DayNum：日期编号，表示数据采集的日期。
- VehId：车辆编号，用于标识不同的车辆。
- Trip：行程，表示车辆的行驶路线或行程编号。
- Timestamp(ms)：时间戳（毫秒），记录数据采集的时间。
- Latitude[deg]：纬度（度），表示地点的纬度坐标。
- Longitude[deg]：经度（度），表示地点的经度坐标。
- Vehicle Speed[km/h]：车辆速度（公里/小时），记录车辆当前的行驶速度。
- OAT[DegC]：外部空气温度（摄氏度），表示车辆周围的环境温度。
- Air Conditioning Power[kW]：空调功率（千瓦），记录车辆空调系统的功率消耗。
- Heater Power[Watts]：加热器功率（瓦特），记录车辆加热系统的功率消耗。
- HV Battery Current[A]：高压电池电流（安培），表示车辆高压电池的电流输出。
- HV Battery SOC[%]：高压电池 SOC（百分比），表示车辆高压电池的电量百分比。
- HV Battery Voltage[V]：高压电池电压（伏特），表示车辆高压电池的电压。
- Elevation Raw[m]：原始海拔（米），记录车辆所在位置的海拔高度（原始数据）。
- Elevation Smoothed[m]：平滑海拔（米），记录车辆所在位置的海拔高度（经过平滑处理后的数据）。
- Gradient：坡度，表示车辆所在位置的坡度。
- Energy_Consumption：能耗，表示车辆的能量消耗情况。
- Matchted Latitude[deg]：匹配后的纬度（度），表示车辆实际位置与地图数据匹配后的纬度坐标。
- Matched Longitude[deg]：匹配后的经度（度），表示车辆实际位置与地图数据匹配后的经度坐标。
- Match Type：匹配类型，表示地图匹配的方式或类型。
- Class of Speed Limit：速度限制等级，表示道路上的速度限制等级。
- Speed Limit[km/h]：速度限制（公里/小时），记录道路的速度限制。
- Speed Limit with Direction[km/h]：带方向的速度限制（公里/小时），记录带有方向的道路速度限制。
- Intersection：交叉路口，记录车辆是否经过交叉路口（0代表无，1代表有）。
- Bus Stops：公交车站，记录车辆是否经过公交车站（0代表无，1代表有）。
- Focus Points：关注点，记录车辆所经过的特殊地点类型，如交叉路口、停车点、交通信号灯等（1代表交叉路口，2代表停车点，3代表交通信号灯）。

2.数据的缺失值、异常值以及标准化处理

3.进行主成分分析或因子分析对数据降维(保留信息要>=90%)，然后进行相关性分析，找出相关性较高的变量，进行变量筛选。

4.使用逻辑回归、决策树、随机森林、XGBoost等算法进行建模，对模型进行评估，选择最优模型。


for step1: MakeData2CSV_byGPT.py
for step2: csv_process.py
for step3: Analysis_factors.py
for step4: EC_Regression_by_factors.py
