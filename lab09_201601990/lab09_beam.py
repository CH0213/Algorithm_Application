import numpy as np

def Greedy_Search():

    score = 10
    list = []

    for i in range(10):
        np.random.seed(int(score))
        temp = np.random.rand(5)
        score *= (10 * temp.max())
        list.append(temp.argmax())

    return [list, score]


def Beam_Search(k):

    score = 10
    sequences = [[list(), score]]
    for i in range(10):
        temp = list()
        for j in range(len(sequences)):
            seq, score = sequences[j]
            np.random.seed(int(sequences[j][1]))
            row = np.random.rand(5)

            for l in range(len(row)):
                temp_2 = [seq + [l], score * 10 * (row[l])]
                temp.append(temp_2)

        ordered = sorted(temp, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:k]

    return sequences



if __name__ == '__main__':

    print("Greedy Search")
    print(Greedy_Search())

    print("Beam Search")
    result = Beam_Search(3)
    for i in result:
        print(i)
