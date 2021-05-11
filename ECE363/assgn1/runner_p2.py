from Templates import *

x_train,y_train, x_test , y_test = DatasetLoader(2).loadDataset()
m1 = LogisticRegressor( lam = None, reg = None, epochs = 100 )
m2 = LogisticRegressor(lam = 0.5,reg= 'l1', epochs=100)
m3 = LogisticRegressor(lam = 0.5, reg = 'l2', epochs= 100)
# m1.fit(x_train=x_train, y_train=y_train)
# m2.fit(x_train=x_train, y_train=y_train)
# m3.fit(x_train=x_train, y_train=y_train)

# plt.plot(range(m1.params['epochs']),m1.params['trainAccuracy'])
# plt.plot(range(m1.params['epochs']),m2.params['trainAccuracy'])
# plt.plot(range(m1.params['epochs']),m3.params['trainAccuracy'])
# plt.show()
