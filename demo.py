from sklearn import tree

# [height (cm), weight (kg), shoe size (EU)]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Gender
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# initializes decision tree
clf = tree.DecisionTreeClassifier()

# Train dataset
clf = clf.fit(X, Y)

# Given new (height, weight, shoe size) predict gender
prediction = clf.predict([[190, 70, 43]])

if __name__ == '__main__':
    print(prediction)
