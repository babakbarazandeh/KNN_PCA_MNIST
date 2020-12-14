import gzip
import numpy as np
from sklearn.decomposition import PCA
import sys
import struct
from array import array
class KNN():
    def __init__(self, K,D,N_Testing, N_Training, PATH):
        self.K = K
        self.D = D
        self.N = N_Testing
        self.Total = N_Training + N_Testing
        self.path_data = PATH + "/train-images-idx3-ubyte"
        self.path_data_label = PATH + "/train-labels-idx1-ubyte"
        self.path = PATH
        self.P_l = []
        self.txt = ""

    def convert_data(self):

        with open(self.path_data_label, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = array("B", file.read())
        with open(self.path_data, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return np.array(images)[0:self.Total, :, :], np.array(labels)[0:self.Total]

    def label(self,Z):
        T = [0.0] * 10

        for i in range(len(Z)):
            if Z[i][0] <= 0.0001:
                return Z[i][1]

            T[Z[i][1]] += 1 / (Z[i][0])

        return np.argmax(T)
    def data_process(self):

        data, label = self.convert_data()
        data_test = data[0:self.N, :, :].reshape((-1, 28 * 28))
        self.label_test = label[0:self.N]

        data_train = data[self.N:, :, :].reshape((-1, 28 * 28))
        self.label_train = label[self.N:]

        pca = PCA(n_components=self.D, svd_solver='full')
        pca.fit(data_train)
        self.Y_test = pca.transform(data_test)
        self.Y_train = pca.transform(data_train)

    def dist(self,x1, x2):
        return np.linalg.norm(x1 - x2)

    def top_k(self,data_tr, data_tr_label, data_test, k):
        disti = []
        for i in range(data_tr.shape[0]):
            temp = self.dist(data_tr[i, :], data_test)
            disti.append((temp, data_tr_label[i]))

        disti.sort(key=lambda tup: tup[0])
        return disti[0:k]
    def predict(self):
        for i in range(self.Y_test.shape[0]):
            Z = self.top_k(self.Y_train, self.label_train, self.Y_test[i, :], self.K)
            l = self.label(Z)
            self.P_l.append(l)
            self.txt += str(l) + " " + str(self.label_test[i]) + "\n"
    def main(self):
        self.data_process()
        self.predict()
        f = open("result.txt","w")
        f.write(self.txt)


O = KNN(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]), int(sys.argv[4]),sys.argv[5])
O.main()
