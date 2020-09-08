import numpy as np
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


#####[데이터 전처리]########################################
file = open('covid-19.txt', 'r', encoding='UTF8')
data = []
line = file.readline()

while True:
    line = file.readline()
    if not line: break
    line = line.split('\t')
    for i in line:
        i.rstrip()
    data.append(line[5:7])

file.close()
data = np.array(data, dtype=float)
data = scaler.fit_transform(data[:])
###########################################################


#####[시각화 함수]##########################################
def draw_graph(data, labels):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    plt.show()
###########################################################


#####[유클리디안 거리함수]##################################
def Euclidean(list1, list2):

    array1 = np.array(list1, dtype=float)
    array2 = np.array(list2, dtype=float)
    array_sub = array1 - array2

    return np.sqrt(np.sum(np.square(array_sub)))
###########################################################


#####[Goal1]###############################################
clustering = DBSCAN(eps=0.1, min_samples=2).fit(data)
clustering2 = AgglomerativeClustering(n_clusters=8, affinity='Euclidean', linkage='complete').fit(data)

draw_graph(data, clustering.labels_)
draw_graph(data, clustering2.labels_)
###########################################################


#####[Goal2]###############################################
class KMeans:
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.cluster = OrderedDict()

    def init_center(self):
        index = random.randint(0, self.n)
        index_list = []
        for i in range(self.n):
            while index in index_list:
                index = random.randint(0, self.n)
            index_list.append(index)
            self.cluster[i] = {'center': self.data[index], 'data': []}


    def clustering(self, cluster):
        for i in self.data:
            a = 2
            num = 0
            for j in range(self.n):
                if(a > Euclidean(i, self.cluster[j]['center'])):
                    a = Euclidean(i, self.cluster[j]['center'])
                    num = j

            self.cluster[num]['data'].append(i)

        return self.cluster


    def update_center(self):
        for i in range(self.n):
            temp = np.zeros(2)
            num = 0
            for j in self.cluster[i]['data']:
                temp += j
                num += 1

            self.cluster[i]['center'] = temp / num


    def update(self):

        while True:
            ##기존 center 저장#######################
            temp_1 = np.zeros((self.n, 2))
            for i in range(self.n):
                temp_1[i] = self.cluster[i]['center']
            #########################################

            ##센터를 업데이트 해준 후 기존 클러스터의 'data'를 비워주는 작업.(비워주지 않는다면 계속 추가되므로) ##
            self.update_center()
            for i in range(self.n):
                self.cluster[i] = {'center': self.cluster[i]['center'], 'data': []}
            ###################################################################################################

            ##클러스터링 후 센터 저장#################
            self.clustering(self.cluster)

            temp_2 = np.zeros((self.n, 2))
            for i in range(self.n):
                temp_2[i] = self.cluster[i]['center']
            #########################################

            ##클러스터링 전 후 center비교#############
            if np.array_equal(temp_1, temp_2):
                break
            #########################################


    def fit(self):
        self.init_center()
        self.cluster = self.clustering(self.cluster)
        self.update()

        result, labels = self.get_result(self.cluster)
        draw_graph(result, labels)

    def get_result(self, cluster):
        result = []
        labels = []
        for key, value in cluster.items():
            for item in value['data']:
                labels.append(key)
                result.append(item)

        return np.array(result), labels

K = KMeans(data, 8)
K.fit()
###########################################################

