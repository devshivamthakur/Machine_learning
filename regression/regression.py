from matplotlib import colors
import numpy as np
import random
from numpy.core.fromnumeric import ptp
from sklearn.model_selection import train_test_split
from student_reg import studentReg
import matplotlib.pyplot as plt
ages_d=[19,20,22,23,25,26]  #ages data
networth_d=[1000,1200,1350,1400,1500,2500]  # net worth data
ages=np.reshape(np.array(ages_d),(len(ages_d),1))  #convert data into proper 2d array
networth=np.reshape(np.array(networth_d),(len(networth_d),1)) #convert data into proper 2d array

ages_train,ages_test,networth_train,networth_test =train_test_split(ages,networth)
# print("ages_train",ages_train)
# print("ages_test",ages_test)
# print("networth_train",networth_train)
# print("networth_test",networth_test)

reg=studentReg(ages_train,networth_train)


print("Coefficient ",reg.coef_)
print("Slop ",reg.intercept_)

print("predict ",reg.predict([[27]]))

plt.scatter(ages_train,networth_train,color='b' ,label="train data")
plt.scatter(ages_test,networth_test,color='r' ,label="test data")
plt.scatter(ages_test,reg.predict(ages_test),color='black')
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("Net Worths")
plt.show()


