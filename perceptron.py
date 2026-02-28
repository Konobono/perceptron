import pandas as pd
import numpy as np

data = pd.read_csv("data_banknote_authentication.csv", sep=",", decimal=".")
data = data.replace('?', np.NaN)
data = data.apply(pd.to_numeric)
data = data.reindex(np.random.permutation(data.index))

X = data.iloc[:, :-1].values
X = np.array(X)
X = np.c_[np.ones((len(X), 1)), X]
y = data.iloc[:, -1].values
y = np.array(y)

n_samples = len(X)
n_train = int(0.8 * n_samples)
n_test = n_samples - n_train

X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]


def activation(matrix):
    if matrix >= 0:
        return 1
    else:
        return 0


def guess(data_x, weights):
    suma = np.dot(weights, data_x)
    y_p = activation(suma)
    return y_p


def train(lr, epoch, weights):
    for j in range(epoch):
        for x_i, target in zip(X_train, y_train):
            delta_w = lr * (target - guess(weights, x_i))  # lr * error i potem *input
            for i in range(len(x_i)):
                weights[i] += delta_w * x_i[i]

    print('Weights: \n', weights)


def test(weights, data_test):
    pred_labels = []

    for i in range(len(y_test)):
        y_pred = guess(weights, data_test[i])
        pred_labels.append(y_pred)

    print("\nTest data: \n", X_test[:5])
    print("\nPredicted labels: \n", pred_labels[:5])

    return pred_labels


def confusion_matrix(true_label, pred_label):

    num_classes = len(np.unique(true_label))
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    con_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(true_label)):
        true_label_check = true_label[i]
        pred_label_check = pred_label[i]
        con_matrix[true_label_check - 1][pred_label_check - 1] += 1
    print("\nConfusion matrix: \n", con_matrix)

    for i in range(len(true_label)):
        for j in range(num_classes):
            if (true_label[i] == j) & (pred_label[i] == j):
                tp[j] += 1
            if (true_label[i] != j) & (pred_label[i] == j):
                fp[j] += 1
            if (true_label[i] != j) & (pred_label[i] != j):
                tn[j] += 1
            if (true_label[i] == j) & (pred_label[i] != j):
                fn[j] += 1

    sensitivity = np.zeros(num_classes, dtype=np.float64)
    specificity = np.zeros(num_classes, dtype=np.float64)
    precision = np.zeros(num_classes, dtype=np.float64)
    accuracy = np.zeros(num_classes, dtype=np.float64)

    for i in range(num_classes):
        sensitivity[i] = tp[i] / (tp[i] + fn[i])
        specificity[i] = tn[i] / (fp[i] + tn[i])
        precision[i] = tp[i] / (tp[i] + fp[i])
        accuracy[i] = (tp[i] + tn[i]) / len(true_label)

    for k in range(num_classes):
        print("\nTP instances for class", k, "=", tp[k])
        print("TN instances for class", k, "=", tn[k])
        print("FP instances for class", k, "=", fp[k])
        print("FN instances for class", k, "=", fn[k])

    for p in range(num_classes):
        print("\nSensitivity for class", p, "=", sensitivity[p]*100, "%")
        print("Specificity for class", p, "=", specificity[p]*100, "%")
        print("Precision for class", p, "=", precision[p]*100, "%")
        print("Accuracy for class", p, "=", accuracy[p]*100, "%")


# w_i = np.random.uniform(low=0, high=1.0, size=5)
w_i = np.zeros(5)
train(lr=0.1, epoch=10, weights=w_i)
predicted_label = test(w_i, X_test)
confusion_matrix(y_test, predicted_label)
