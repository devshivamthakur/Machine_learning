
from sklearn.linear_model import LinearRegression

def studentReg(ages_train, networth_train):
    reg=LinearRegression()  #create and train a regression
    reg.fit(ages_train,networth_train)    # fit data into regression
    return reg
