import torch
import numpy as np
import pandas as pd
import torch.optim
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv("SpotifyFeatures.csv")
data = data.sample(frac=1).reset_index(drop=True)
unused_col = ['artist_name', 'track_name', 'track_id', 'key', 'time_signature',
              'popularity', 'loudness', 'tempo']
df = data.drop(columns=unused_col).reset_index(drop=True)
# numerical representations of a few variables
# df['time_signature'].unique().tolist()
df['mode'].unique().tolist()
# df['key'].unique().tolist()
mode_dict = {'Major': 1, 'Minor': 0}
# key_dict = {'C': 1, 'C#': 2, 'D': 3, 'D#': 4, 'E': 5, 'F': 6,
#            'F#': 7, 'G': 9, 'G#': 10, 'A': 11, 'A#': 12, 'B': 12}

# df['time_signature'] = df['time_signature'].apply(lambda x: int(x[0]))
df['mode'].replace(mode_dict, inplace=True)
# df['key'] = df['key'].replace(key_dict).astype(int)
df['duration_ms'] = np.log10(df['duration_ms'])
# df['popularity'] = df['popularity'] / 100
# df['loudness'] = (df['loudness'] + 25) / 25
# df['tempo'] = df['tempo'] / 240

features_pandas = df.drop(columns=['genre'])

features = torch.tensor(features_pandas.values, dtype=torch.float)
targets_pandas = df['genre']
# targets_pandas = y_1[0:len(y_1)//2]
targets = torch.tensor((pd.get_dummies(targets_pandas)).values,
                       dtype=torch.float)

features_train = features[:features.shape[0] * 4 // 5]
features_test = features[features.shape[0] * 4 // 5:]

targets_train = targets[:targets.shape[0] * 4 // 5]
targets_test = targets[targets.shape[0] * 4 // 5:]

train_ds = TensorDataset(features_train, targets_train)
batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

test_ds = TensorDataset(features_test, targets_test)
batch_size_test = 10
test_dl = DataLoader(test_ds, batch_size, shuffle=True)

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(9, 15),
    torch.nn.ReLU(),
    torch.nn.Linear(15, 18),
    torch.nn.ReLU(),
    torch.nn.Linear(18, 26),
    torch.nn.Softmax()
)

lr_rate_1 = .5
lr_rate_2 = 0.001


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 100 == 0:
            loss, current = loss.item(), i * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
        # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += loss_fn(outputs, y).item()
            _, y_labels = torch.max(y.data, 1)
            correct += (predicted == y_labels).sum().item()

    total /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss:{total:>8f} \n")


optimizer_1 = torch.optim.SGD(model.parameters(), lr=lr_rate_1)
optimizer_2 = torch.optim.SGD(model.parameters(), lr=lr_rate_2)
criterion = torch.nn.MSELoss()
epochs_1 = 20
epochs_2 = 30

for t in range(epochs_1):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dl, model, criterion, optimizer_1)
for t in range(epochs_2):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dl, model, criterion, optimizer_2)
test_loop(test_dl, model, criterion)
print("Done!")

# test case
# print(model(song))
# song1 = torch.tensor([12, 1, 0.7, 490, 0.6, 0.5, 0.4964,
#                      -14.287, 1, 0.2547, 95.001, 0.986])
