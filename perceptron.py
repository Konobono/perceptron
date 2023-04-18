import pandas as pd
import numpy as np
import csv

data = pd.read_csv("data_banknote_authentication.csv", sep=",", decimal=".")
# data.drop(data.columns[2], axis=1, inplace=True)
data = data.replace('?', np.NaN)
data = data.apply(pd.to_numeric)
# data = data.sample(frac=1, random_state=42)
data = data.reindex(np.random.permutation(data.index))
# print(data)


X = data.iloc[:, :-1].values
X1 = np.array(X)
y = data.iloc[:, -1].values
y1 = np.array(y)

n_samples = len(X)
n_train = int(0.6 * n_samples)
n_val = int(0.2 * n_samples)
n_test = n_samples - n_train - n_val


# zainicjowac wagi -> obliczyc wagi*input(x) W CALYM WIERSZU ->
# za pomoca funkcji signum wyznaczyc klase 0 albo 1 jesli >1 lub <1
# -> jesli predykcja = prawdziwa klasa to idziemy do x+1, jezeli nie edytujemy wagi i bias


def weights_matrix(data_x):
    weights = np.random.uniform(low=-1.0, high=1.0, size=data_x.shape)
    weights_array = np.array(weights)
    # print(weights_array)
    return weights_array


def sum_matrix(data_x):
    suma = np.multiply(weights_matrix(data_x), data_x)
    sum_wx = np.array(suma)
    result = []

    for row in sum_wx:
        row_sum = sum(row)
        result.append(row_sum)
    return result


def first_prediction(data_x):

    y_pred = np.sign(sum_matrix(data_x))
    a_2d = y_pred.reshape((-1, 1))
    f = open('trans_matrix.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(a_2d)
    f.close()
    y_pred_t = pd.read_csv("trans_matrix.csv", sep=",", decimal=".")
    y_pred_t = y_pred_t.apply(pd.to_numeric)
    return y_pred_t


def w_correction(data_x):
    matrix = first_prediction(data_x)
    for i in range(len(matrix)):
        if matrix[i] == y[i]:
            continue
        else:
            pass


print(first_prediction(X1))



