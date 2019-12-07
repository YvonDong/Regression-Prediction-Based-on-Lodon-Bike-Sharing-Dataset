import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (8.0, 8.0)

#数据读取与信息查看
data = pd.read_csv('london_merged.csv')
data.info()
length = len(data.timestamp[:])
data_hour = np.zeros((length, 1), dtype=np.int64) #增加整数时刻列
data['hour'] = data.timestamp.apply(lambda x: x.split()[1])
for i in np.arange(length):
    hours = np.arange(24)
    for hour in hours:
        time = str(hour).rjust(2,'0') + ':00:00'
        if time == data.hour[i]:
            data_hour[i] = hour
            break
data['hour'] = data_hour
data = data.drop(['timestamp'], axis=1)

#去除异常点
fig = plt.figure()
plt.subplot(1, 2, 1)
sn.boxplot(data=data,y="cnt",orient="v")
plt.title("未除离群点", fontsize=16)
data = data[np.abs(data['cnt'] - data['cnt'].mean()) <= (3 * data['cnt'].std())]
plt.subplot(1, 2, 2)
sn.boxplot(data=data,y="cnt",orient="v")
plt.title("去除离群点", fontsize=16)
fig.show()
fig.savefig('outliers.jpg')
data.info()

#相关分性析
fig = plt.figure()
corr_mat = data[['t1', 't2', 'hum', 'wind_speed', 'weather_code',
                 'is_holiday', 'is_weekend', 'season', 'hour', 'cnt']].corr()
mask = np.array(corr_mat)
mask[np.tril_indices_from(mask)] = False
sn.heatmap(corr_mat, mask=mask, vmax=8, square=True, annot=True)
plt.title('相关性分析', fontsize=16)
plt.show()
fig.savefig('correalation.jpg')

#数据取值分布
fig = plt.figure()
sn.distplot(data['cnt'])
plt.title("cnt的统计分布", fontsize=16)
plt.show()
fig.savefig('bike-share.jpg')

#数据取值取对数后的分布
fig = plt.figure()
sn.distplot(np.log(data['cnt'] + 1))
plt.title("log(cnt+1)的统计分布", fontsize=16)
plt.show()
fig.savefig('log-bike-share.jpg')

#数据随时间及季节的变化
fig, ax = plt.subplots()
hourAggregated = pd.DataFrame(data.groupby(["hour","season"],sort=True)["cnt"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["cnt"],hue=hourAggregated["season"], data=hourAggregated, join=True, ax=ax)
ax.set(xlabel='小时', title='每天各小时使用量均值')
fig.show()
fig.savefig('hour-share.jpg')