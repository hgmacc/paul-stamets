import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sbn
from scipy.stats import mannwhitneyu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

mushrooms = pd.read_csv("data/mushroom.csv")
label_encoder = LabelEncoder()
mushrooms_encoded = mushrooms.apply(label_encoder.fit_transform)
y = mushrooms_encoded["0"]
X = mushrooms_encoded.drop("0", axis=1)
print(X[:5])
print(y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
knn.predict(X_test)

print(knn.score(X_test, y_test))

'''

arr = [
    4.0000000000000036,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.20000000000000018,
    0.20000000000000018,
    0.20000000000000018,
    0.20000000000000018,
    0.20000000000000018,
    0.20000000000000018,
    0.6000000000000005,
    0.40000000000000036,
    0.8000000000000007,
    0.6000000000000005,
    1.0000000000000009,
    0.8000000000000007,
    1.4000000000000012,
    1.200000000000001,
]
outfile = Path(__file__).resolve().parent / "errors.npy"
np.save(outfile, arr, allow_pickle=False)



lis = [[1, 4, 7], [3, 6, 9], [2, 59, 8]]
lis = sorted(lis, key=lambda x: x[0])
print(lis)

features = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises?",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]
a = []
for i, l in zip(features, auc_val):
    r = [i, l]
    a.append(r)
a = sorted(a, key=lambda h: h[1])
for g in a:
    print(g)


features = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises?",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

# start:stop:step
# df = DataFrame(shroom_df, columns=list("abcde"))
# RangeIndex: 8124 entries, 0 to 8123

a = [1, 2, 3, 4]

for i in a:
    print(i)

def plot_knn(knn_error, graph):
    """ A plot of testing error rate (as a percentage on the y axis) vs. K (x axis). """
    df = pd.DataFrame({"%_Error": knn_error, "K_Values": list(range(1, 21))})
    sbn.scatterplot(data=df, y="%_Error", x="K_Values", s=100, color=".2", marker="+")
    plt.title(graph)
    plt.show()



def plot_knn(knn_error, graph):
    """ A plot of testing error rate (as a percentage on the y axis) vs. K (x axis). """
    df = pd.DataFrame({"%_Error": knn_error, "K_Values": list(range(1, 21))})
    sbn.scatterplot(data=df, y="%_Error", x="K_Values", s=100, color=".2", marker="+")
    plt.title(graph)
    plt.show()


shroom_df = pd.read_csv("data/mushroom.csv", usecols=list(range(1, 20)), header=None)
shroom_df = shroom_df.apply(LabelEncoder().fit_transform)
tr_data = shroom_df[:4062].to_numpy()  # first half of csv file
test_data = shroom_df[4062:]  # second half

for i in range(3):
    print(tr_data[i])

shroom_label = pd.read_csv("data/mushroom.csv", usecols=[0], header=None)
shroom_label = shroom_label.apply(LabelEncoder().fit_transform)

tr_label = shroom_label[:4062]  # first half of labels
test_label = shroom_label[4062:]  # second half of labels
'''
