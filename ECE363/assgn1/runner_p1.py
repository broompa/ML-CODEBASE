from Templates import *

# Part 1.1
x,y = DatasetLoader(0).loadDataset()
cv = CrossValidate(x,y)
cv.k_fold(folds=5)

# Part 1.2
cv.newVal2Test(cv.leastValRMSE())

mParams_1 = {"lam":[0.01, 0.1, 0.5, 1], "lr":[1e-5 , 1e-4 ], "reg": ['\"l1\"'],'epochs':[500]} # l1
cv.GridSearch(LinearRegressorGradient,mParams_1,folds=5)
config_l1b = cv.getBestConfig()

mParams_2 = {"lam":[0.01, 0.1, 0.5, 1], "lr":[1e-6 , 1e-4 ], "reg": ['\"l2\"'],'epochs':[100]} # l2
cv.GridSearch(LinearRegressorGradient,mParams_2,folds=5)
config_l2b = cv.getBestConfig()

m1 = LinearRegressorGradient(**eval(config_l1b))
m1.fit(x_train = cv.params['x_new'],y_train=cv.params['y_new'])
m2 = LinearRegressorGradient(**eval(config_l2b))
m2.fit(x_train=cv.params['x_new'],y_train=cv.params['y_new'])

print("L1 RMSE Loss(test set): %f"%(m1.rmseLoss(y_val=cv.params['y_ntest'],x_val=cv.params['x_ntest'])))
print(config_l1b)
print("L2 RMSE Loss(test set): %f"%(m2.rmseLoss(y_val=cv.params['y_ntest'],x_val=cv.params['x_ntest'])))
print(config_l2b)

plt.subplot(2,2,1)
plt.plot(range(cv.params['epochs']), cv.params['trainRMSE_G'],label = "Mean RMSE Train")
plt.legend()
plt.subplot(2,2,2)
plt.plot(range(cv.params['epochs']),  cv.params['valRMSE_G'],label = "Mean RMSE Val")
plt.legend()
plt.subplot(2,2,3)
plt.plot(range(m1.params['epochs']), m1.params['trainRMSE'],label = 'l1 reg')
plt.legend()
plt.subplot(2,2,4)
plt.plot(range(m2.params['epochs']), m2.params['trainRMSE'],label = 'l2 reg')
plt.legend()
plt.show()

# Part 1.3
x,y = DatasetLoader(1).loadDataset()

m1 = LinearRegressorGradient(lam = None , reg = None, epochs=1000)
m1.fit(x_train = x,y_train =y)

m2 = LinearRegressorGradient(lam = 0.5, reg = 'l1', epochs= 1000)
m3 = LinearRegressorGradient(lam = 0.5, reg = 'l2', epochs= 1000)
m2.fit(x_train = x,y_train =y )
m3.fit(x_train = x,y_train =y)
m4 = LinearRegressorNormal()
m4.fit(x_train=x,y_train=y)

y1 = m1.predict(x)
y2 = m2.predict(x)
y3 = m3.predict(x)
y4 = m4.predict(x)


plt.scatter(x,y,c='r',marker='.')
plt.plot(x,y1,'b--',label = 'Reg None')
plt.plot(x,y2,'g--',label = 'Reg l1')
plt.plot(x,y3,'m--',label = 'Reg l2')
plt.plot(x,y4,'y--',label = 'Normal Eq')
plt.legend()
plt.show()



