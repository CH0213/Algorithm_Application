import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


#####[데이터 인덱싱]########################################
file = open('seoul_tax.txt', 'r', encoding='UTF8')
matrix = []
line = file.readline()

while True:
    line = file.readline()
    if not line: break
    line = line.split('\t')
    for i in line:
        i.rstrip()
    matrix.append(line[1:])

file.close()
###########################################################


#####[함수 정의]############################################
def Cosine(list):
    return np.ones((25, 25)) - cosine_similarity(matrix)

def Manhattan(list):
    temp = np.zeros((25,25))
    np.array(list, dtype=float)

    for i in range(25):
        for j in range(25):
            array1 = np.array(list[i], dtype=float)
            array2 = np.array(list[j], dtype=float)
            array_sub = array1-array2
            temp[i,j] = np.sum(np.abs(array_sub))
    return temp

def Euclidean(list):
    temp = np.zeros((25,25))
    np.array(list, dtype=float)

    for i in range(25):
        for j in range(25):
            array1 = np.array(list[i], dtype=float)
            array2 = np.array(list[j], dtype=float)
            array_sub = array1 - array2

            temp[i,j] = np.sqrt(np.sum(np.square(array_sub)))

    return temp
###########################################################


#####[시각화]##############################################
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.title("Cosine")
plt.pcolor(Cosine(matrix))
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Manhattan")
plt.pcolor(Manhattan(matrix))
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Euclidean")
plt.pcolor(Euclidean(matrix))
plt.colorbar()

plt.show()
###########################################################


#####[정규화 후 시각화]#####################################
matrix_n = scaler.fit_transform(matrix[:])


plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.title("Cosine(Normalization)")
matrix_cosine = np.ones((25,25)) - cosine_similarity(matrix_n)
plt.pcolor(matrix_cosine)
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Manhattan(Normalization)")
plt.pcolor(Manhattan(matrix_n))
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Euclidean(Normalization)")
plt.pcolor(Euclidean(matrix_n))
plt.colorbar()

plt.show()
###########################################################