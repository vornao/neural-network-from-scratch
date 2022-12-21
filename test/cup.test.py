import pandas as pd


# import cup tr from data folder, but remove lines starting with # and remove startring and ending spaces
tr = pd.read_csv("data/cup.train.csv", comment="#")

#change figsize seaborn heatmap
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
#plot heatmap
sns.heatmap(tr.corr(), annot=True, fmt=".2f")

# make multiple kde plots for attributes from a1 to a8
import matplotlib.pyplot as plt
for i in range(1,9):
    sns.kdeplot(tr["a"+str(i)], shade=True)
    plt.show()

# change figure size
plt.figure(figsize=(20,10))

# change plot size matplotlib
plt.rcParams["figure.figsize"] = (20,10)



# perform minmax scaling on each column
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
tr = pd.DataFrame(scaler.fit_transform(tr), columns=tr.columns)


# split in training and validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(tr.drop(["ty", 'tx'], axis=1), tr[['tx','ty']], test_size=0.2, random_state=42)



# build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
    ])





sns.kdeplot(tr['a1'], shade=True)
sns.kdeplot(tr['a8'], shade=True)
