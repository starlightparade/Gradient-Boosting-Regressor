import numpy as np
from DecisionTreeRegressor import MyDecisionTreeRegressor
import os
import json
import operator


class MyGradientBoostingRegressor():
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param learning_rate: type:float
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        int (default=100)
        :param n_estimators: type: integer
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        :param max_depth: type: integer
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node

        estimators: the regression estimators
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = np.empty((self.n_estimators,), dtype=np.object)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.f0 = 0

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.estimators in this function
        '''
        N = X.shape[0]
        self.f_0 = np.average(y)
        # print("f0 = ",self.f_0)
 
        n_estimators = self.n_estimators
        lr = self.learning_rate
        
        for m in range(n_estimators):
            if m == 0:
                y_predict = np.repeat(self.f_0, len(y))
            else:
                y_predict = y_predict_new
                
            residual = y - y_predict
            regressor_m = MyDecisionTreeRegressor(min_samples_split=self.min_samples_split,max_depth=self.max_depth)
            regressor_m.fit(X, residual)
            y_predict_new = y_predict + lr * regressor_m.predict(X)

            self.estimators[m] = regressor_m

        return self.estimators

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_predict = self.f_0
        for m in range(self.n_estimators): 
            y_predict += self.learning_rate * self.estimators[m].predict(X)

        return y_predict

    def get_model_string(self):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})

        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            n_estimators = 10 + j * 10
            gbr = MyGradientBoostingRegressor(n_estimators=n_estimators, max_depth=5, min_samples_split=2)
            gbr.fit(x_train, y_train)
            model_string = gbr.get_model_string()

            with open("Test_data" + os.sep + "gradient_boosting_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)

            print(operator.eq(model_string, test_model_string))

            y_pred = gbr.predict(x_train)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_gradient_boosting_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)
