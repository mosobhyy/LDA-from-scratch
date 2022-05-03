import numpy as np


class LDA_SCRATCH:
    def __init__(self):
        self.z = []
        self.w = []
        self.b = []

    def fit(self, X, y):

        def sum_square_error(z):
            """##### **Formulate W**"""
            w = np.zeros((z.shape[1], 1))

            """##### **Formulate b**"""
            b = np.ones((z.shape[0], 1))

            """##### **Get W values (W = inverse(Z.T * Z) * Z.T * b)**"""
            z, w, b = np.matrix(z), np.matrix(w), np.matrix(b)
            z_transpose = np.transpose(z)

            """##### **inverse(Z.T * Z)**"""
            term1 = np.dot(z_transpose, z)
            term1 = np.linalg.inv(term1)

            """##### **Z.T * b**"""
            term2 = np.dot(z_transpose, b)

            """##### **W**"""
            w = np.dot(term1, term2)

            return z, w, b

        data = X.copy()
        z = X.copy()
        target = y.name
        data[target] = y.values

        """##### **Adding a new column called "ones" to satisfy "straight-line equation"**"""
        z.insert(z.shape[1], 'Ones', 1)

        """##### **Reformulate by multiplying all features of other classes by -1**"""
        labels = data[target].unique()
        labels.sort()

        """##### **Coefficients of line equation (Weights) between 1 Class and the two others)**"""
        for label in labels:
            temp = z.copy()
            temp *= -1
            temp[data[target] == label] = temp[data[target] == label] * -1

            result = sum_square_error(temp)
            self.z.append(result[0])
            self.w.append(result[1])
            self.b.append(result[2])

    def predict(self, x_test):
        """##### **Augment x_test with columns of ones**"""
        ones = np.ones((x_test.shape[0]))
        x_test = np.column_stack((x_test, ones))

        """##### **Result as shape of (number of samples * number of classes)**"""
        result = np.zeros((x_test.shape[0], len(self.w)), dtype=int)

        """##### **Augment x_test with columns of ones**"""
        for i, weights in enumerate(self.w):
            eq = np.dot(x_test, weights)
            eq = np.where(eq > 0, 1, 0)
            result[:, i] = eq.flatten()

        return result

    def score(self, y_test, y_pred):
        stats = []
        error = 0

        # count of classes predicted for the sample
        count_classes_predicted = np.count_nonzero(y_pred, axis=1)

        # predicted class for the sample
        classes_predicted = np.argmax(y_pred, axis=1)

        for i in range(len(y_test)):
            if count_classes_predicted[i] == 0:
                stats.append('New classification')

            elif count_classes_predicted[i] == 1:
                if classes_predicted[i] == y_test.values[i]:
                    stats.append('Classified Correctly')
                else:
                    stats.append('Classified Wrong')
                    error += 1

            elif count_classes_predicted[i] == 2:
                stats.append('Undetermined Class')

        total = len(y_test)
        accuracy = (1 - error / total) * 100

        return accuracy, stats
