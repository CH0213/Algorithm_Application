import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import MinMaxScaler


#####[데이터 전처리]########################################
file = open('./seoul_student.txt', 'r', encoding='UTF8')
data = []
line = file.readline()

while True:
    line = file.readline()
    if not line: break
    line = line.split('\t')
    for i in line:
        i.rstrip()
    data.append(line[:])

file.close()
data = np.array(data, dtype=float)
###########################################################


#####[시각화 함수]##########################################
def draw_graph(data):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], cmap='rainbow')
    plt.show()

def draw_graph2(data):
    plt.figure()
    plt.scatter(data, [0] * len(data), cmap='rainbow')
    plt.show()
###########################################################


#####[Goal1]###############################################
scaler = MinMaxScaler()
data = scaler.fit_transform(data[:])
draw_graph(data)
###########################################################


#####[Goal2]###############################################
sklearn_pca = sklearnPCA(n_components=1)
sklearn_transf = sklearn_pca.fit_transform(data)

draw_graph2(sklearn_transf)
###########################################################


#####[Goal3]###############################################

# mean 함수 구현
def mean(data):
    temp = np.zeros((1, len(data[0])))
    num = 0
    for i in data:
        temp += i
        num += 1
    return temp/num


# Covariance Matrix 함수 구현
def cov(data):
    return data.T.dot(data)/len(data.T[0])


# mean을 사용하여 데이터 정규화
data = data-mean(data)


# Eigenvectors, Eigenvalues 구하기
eig_val_cov, eig_vec_cov = np.linalg.eig(cov(data))


# 차원 축소
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_vec_cov))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
projection_mat = np.array([eig_pairs[0][1]])


# 선형변환을 통해 새로운 데이터 셋 구하기
transfromed = projection_mat.dot(data.T)


# 새로운 데이터 셋 시각화
draw_graph2(transfromed.T)

###########################################################
