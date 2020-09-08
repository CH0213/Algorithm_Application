import pydot as pydot
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

## <Goal 1 : Data 전처리> #################################
Train = pd.read_csv('./data/train.csv')
Test = pd.read_csv('./data/test.csv')

def Data_preprocessing(data):

    ## String => 숫자
    data['Sex'] = pd.get_dummies(data['Sex'])


    ## NaN값을 최빈값으로 대체 후 Lanbel Encoding
    most_freq = data['Embarked'].value_counts(dropna=True).idxmax()
    data['Embarked'].fillna(most_freq, inplace=True)
    label = LabelEncoder()
    label.fit(['C', 'Q', 'S'])
    data["Embarked"] = label.transform(data["Embarked"])

    ## NaN값을 평균 값으로 대체
    mean_age = np.mean(data['Age'])
    data['Age'].fillna(mean_age,inplace = True)
    mean_fare = np.mean(data['Fare'])
    data['Fare'].fillna(mean_fare,inplace = True)

    ## 삭제
    del data['Name']
    del data['Ticket']
    del data['Cabin']

    return data


Train = Data_preprocessing(Train)
Test = Data_preprocessing(Test)
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

Train_X = Train[features]
Train_Y = Train['Survived']
Test_X = Test[features]
###########################################################




## <Goal 2 : Decision Tree 활용 및 시각화> #################

## 시각화 코드
def Visualization(dt):
    export_graphviz(dt, out_file="dt.dot", class_names=['No', 'Yes'],
                    feature_names=features, impurity=False, filled=True)
    (graph, ) = pydot.graph_from_dot_file('dt.dot', encoding='utf8')
    graph.write_png('dt.png')


## 의사 결정 트리 생성, train_X, train_Y로 학습
dt = DecisionTreeClassifier()
dt.fit(Train_X, Train_Y)

## 의사 결정 트리를 시각화
Visualization(dt)

## 학습한 의사 결정 트리로 test_X를 예측
predict = dt.predict(Test_X)

## gender_submission의 형태로 csv파일을 생성
submit = pd.read_csv('./data/gender_submission.csv', index_col='PassengerId')
submit["Survived"] = predict
submit = pd.DataFrame(submit)
submit.to_csv("Prediction_submit.csv")


###########################################################
