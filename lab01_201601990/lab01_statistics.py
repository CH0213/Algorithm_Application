import numpy as np
import matplotlib.pyplot as plt

array1 = np.zeros(101)
array2 = np.zeros(101)
array3 = np.zeros(101)

file = open('seoul.txt', 'r', encoding='UTF8')


line = file.readline()
num = line.replace("세", "").split()
age = np.array(num[2:103], dtype=int)

i = 0
while True:
    i += 1
    line = file.readline()
    if not line: break
    num = line.split()
    arrayX = np.array(num[2:], dtype=int)
    if (i % 3 == 1):
        array1 += arrayX
    if (i % 3 == 2):
        array2 += arrayX
    if (i % 3 == 0):
        array3 += arrayX

file.close()
print("계 :", end = ' ')
for i in array1:
    print(int(i), end = '  ')

print("\n계 총합: ", int(array1.sum()), end = ' ')
print("\n계 평균: ", int(array1.mean()),end = ' ')
print("\n계 분산: ", int(array1.var()))

print("\n남자 :", end = ' ')
for i in array2:
    print(int(i), end = '  ')

print("\n남자 총합: ", int(array2.sum()), end = ' ')
print("\n남자 평균: ", int(array2.mean()),end = ' ')
print("\n남자 분산: ", int(array2.var()))

print("\n여자 :", end = ' ')
for i in array3:
    print(int(i), end = '  ')

print("\n여자 총합: ", int(array3.sum()), end = ' ')
print("\n여자 평균: ", int(array3.mean()),end = ' ')
print("\n여자 분산: ", int(array3.var()))

plt.figure()
plt.xlabel('age')
plt.ylabel('population')
plt.title('total')
plt.bar(age, array1)
plt.show()

plt.figure()
plt.xlabel('age')
plt.ylabel('population')
plt.title('man')
plt.bar(age, array2)
plt.show()

plt.figure()
plt.xlabel('age')
plt.ylabel('population')
plt.title('woman')
plt.bar(age, array3)
plt.show()