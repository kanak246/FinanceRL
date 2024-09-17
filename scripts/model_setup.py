from sklearn.linear_model import SGDRegressor

#Creating a Random Forest Classifier Model 

def create_model(): 

    #Default parameters --> need to change/modify 
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    return model 
