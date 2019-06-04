# # Bonus task - LSTM implementation

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

from utils import read_datasets, get_score, plot_anomalies
import numpy as np


# Read datasets

scaled_df1, train_y1, scaled_df2, train_y2, scaled_test_df, y = read_datasets()


# Set network parameters
# torch.cuda.set_device(0)

input_size = scaled_test_df.shape[1]

# Data params
batch_size = scaled_df1.shape[0]

# LSTM reads in one timestep at a time.
lstm_input_size = 1

# size of hidden layers
h1 = 32
output_dim = 1
num_layers = 3
learning_rate = 1e-3
num_epochs = 2
dtype = torch.float


# ## Convert datasets to Tensors

# make training and test sets in torch
X_train = torch.from_numpy(scaled_df1.to_numpy()).type(torch.Tensor)
X_val = torch.from_numpy(scaled_df2.to_numpy()).type(torch.Tensor)
X_test = torch.from_numpy(scaled_test_df.to_numpy()).type(torch.Tensor)
y_train = torch.from_numpy(train_y1.to_numpy()).type(torch.Tensor).view(-1)
y_val = torch.from_numpy(train_y2.to_numpy()).type(torch.Tensor).view(-1)
y_test = torch.from_numpy(y.to_numpy()).type(torch.Tensor).view(-1)

X_train = X_train.view(scaled_df1.shape[1], scaled_df1.shape[0])
X_val = X_val.view(scaled_df2.shape[1], scaled_df2.shape[0])
X_test = X_test.view(scaled_test_df.shape[1], scaled_test_df.shape[0])


# ## Build LSTM model

# Define model
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self, x):
        return (torch.zeros(self.num_layers, x.shape[0], self.hidden_dim),
                torch.zeros(self.num_layers, x.shape[0], self.hidden_dim))
    
    # forward pass through LSTM layer
    def forward(self, x):
        self.hidden = self.init_hidden(x)

        lstm_out, self.hidden = self.lstm(x.view(x.shape[0], x.shape[1], -1))

        y_pred = self.linear(lstm_out[-1].view(x.shape[1], -1))
        return y_pred.view(-1)


# ## Create LSTM



model = LSTM(lstm_input_size, h1, batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(reduction='sum')

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train model

hist = np.zeros(num_epochs)
prediction_list = []
prediction_list_val = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)

    hist[epoch] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()
    running_loss_train = loss.item()
    print('Epoch {} Train Loss:{}'.format(epoch + 1, running_loss_train))


# Plot preds and performance
# base on https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098

valid_x_predictions = model(X_val)
mse = np.mean(np.power(X_val.detach().numpy() - valid_x_predictions.detach().numpy(), 2), axis=0)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': y_val == 1})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
# plt.show()
plt.savefig('../plots/lstm/pre_rec.png', bbox_inches='tight')


# ## Plot reconstruction error for the two classes


# based on https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
threshold = 1


test_x_predictions = model(X_test)
mse = np.mean(np.power(X_test.detach().numpy() - test_x_predictions.detach().numpy(), 2), axis=0)
error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y})
error_df_test = error_df_test.reset_index()

groups = error_df_test.groupby('True_class')

fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Attack" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
# plt.show()
plt.savefig('../plots/lstm/diff_classes.png', bbox_inches='tight')


# Scores


predicted_anomalies = np.where(error_df.Reconstruction_error.values > threshold)[0]
true_anomalies = np.where(error_df_test.True_class.values > 0)[0]
[tp, fp, fn, tn, tpr, tnr, Sttd, Scm, S] = get_score(predicted_anomalies, true_anomalies, y=y)
print("TP: {0}, FP: {1}, TPR: {2}, TNR: {3}".format(tp, fp, tpr, tnr))
print("Sttd: {0}, Scm: {1}, S: {2}".format(Sttd, Scm, S))
print("TN: {0}, FN: {1}".format(tn, fn))


