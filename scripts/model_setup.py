from sklearn.ensemble import RandomForestRegressor

#Creating a Random Forest Classifier Model 

def create_model(): 

    #Default parameters --> need to change/modify 
    model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    return model 