import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (8.0, 8.0)

# 训练数据读取及预处理
train_data = pd.read_csv('train_data.csv')
train_length = len(train_data.timestamp[:])
train_hour = np.zeros((train_length, 1), dtype=np.int64)
train_data['hour'] = train_data.timestamp.apply(lambda x: x.split()[1])
for i in np.arange(train_length):
    hours = np.arange(24)
    for hour in hours:
        time = str(hour) + ':00'
        if time == train_data['hour'][i]:
            train_hour[i] = hour
            break
train_data['hour'] = train_hour[:]
train_data = train_data.drop(['timestamp'], axis=1)
train_data = train_data[np.abs(train_data['cnt'] - train_data['cnt'].mean())
                        <= (3 * train_data['cnt'].std())]

# 测试数据读取及预处理
test_data = pd.read_csv('test_data.csv')
test_length = len(test_data['timestamp'][:])
test_hour = np.zeros((test_length, 1), dtype=np.int64)
for i in np.arange(test_length):
    hours = np.arange(24)
    for hour in hours:
        time = str(hour) + ':00'
        if time in test_data.timestamp[i]:
            test_hour[i] = hour
            break
test_data['hour'] = test_hour
test_data = test_data.drop(['timestamp'], axis=1)
test_data = test_data[np.abs(test_data['cnt'] - test_data['cnt'].mean())
                      <= (3 * test_data['cnt'].std())]

X_train = train_data.drop(['cnt', 't2'], axis=1)
#X_train = train_data.drop(['cnt', 't2', 'is_holiday','is_weekend'], axis=1)
y_train = np.log(1 + train_data['cnt'])

X_test = test_data.drop(['cnt', 't2'], axis=1)
#X_test = test_data.drop(['cnt', 't2', 'is_holiday','is_weekend'], axis=1)

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#岭回归预测（即带L2正则化的线性回归）
RidgeModel = Ridge(alpha=1100)
RidgeModel.fit(X_train, y_train)
y_pred = pd.Series(np.exp(RidgeModel.predict(X_train)) - 1)
y_real = train_data['cnt']
y = pd.DataFrame({'real': y_real, 'prediction': y_pred})
fig = plt.figure()
plt.plot(y[['real', 'prediction']],alpha=0.5)
plt.legend(labels=['real', 'prediction'])
plt.title('岭回归训练结果', fontsize=16)
fig.show()
fig.savefig('Riger_Train.jpg')
print(rmsle(y_real, y_pred, False))
print(r2_score(np.log(y_real+1), np.log(y_pred+1)))

y_pred = pd.Series(np.exp(RidgeModel.predict(X_test)) - 1)
y_real = test_data['cnt']
y = pd.DataFrame({'real': y_real, 'prediction': y_pred})
fig = plt.figure()
plt.plot(y[['real', 'prediction']],alpha=0.5)
plt.legend(labels=['real', 'prediction'])
plt.title('岭回归预测结果', fontsize=16)
fig.show()
fig.savefig('Riger_Test.jpg')
print(rmsle(y_real, y_pred, False))
print(r2_score(np.log(y_real+1), np.log(y_pred+1)))


#随机森林回归预测
RandomForestModel = RandomForestRegressor(50)
RandomForestModel.fit(X_train, y_train)
y_pred = pd.Series(np.exp(RandomForestModel.predict(X_train)) - 1)
y_real = train_data['cnt']
y = pd.DataFrame({'real': y_real, 'prediction': y_pred})
fig = plt.figure()
plt.plot(y[['real', 'prediction']],alpha=0.5)
plt.legend(labels=['real', 'prediction'])
plt.title('随机森林回归训练结果', fontsize=16)
fig.show()
fig.savefig('RandomForest_Train.jpg')
print(rmsle(y_real, y_pred, False))
print(r2_score(np.log(y_real+1), np.log(y_pred+1)))

y_pred = pd.Series(np.exp(RandomForestModel.predict(X_test)) - 1)
y_real = test_data['cnt']
y = pd.DataFrame({'real': y_real, 'prediction': y_pred})
fig = plt.figure()
plt.plot(y[['real', 'prediction']],alpha=0.5)
plt.legend(labels=['real', 'prediction'])
plt.title('随机森林回归预测结果', fontsize=16)
fig.show()
fig.savefig('RandomForest_Test.jpg')
print(rmsle(y_real, y_pred, False))
print(r2_score(np.log(y_real+1), np.log(y_pred+1)))
