import pickle
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, TomekLinks, RandomUnderSampler
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from sklearn.preprocessing import MinMaxScaler
from mlxtend.preprocessing import shuffle_arrays_unison
#from mlxtend.preprocessing import shuffle_arrays_unison
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import torch
from torch.utils import data as torchData
import random
import sys
import io
import os
import os.path as osp
from torchsummary import summary
import time
import copy
from tqdm.auto import tqdm
from utils import *
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

"""
This is a heavily modified file by Pav Patra taken from the following article: 
https://towardsdatascience.com/shortest-path-distance-with-deep-learning-311e19d97569 which takes 
concepts from the paper node2vec: Scalable Feature Learning for Networks by Grover, Aditya and Leskovec, 
Jure (https://doi.org/10.1145/2939672.2939754). The original file contained several issues and 
compatibility issues with my own system.
"""

"""
This file is created by Pav Patra

Execute this file in a virtual environment containing all packages for the libraries above. 

This file trains a best case node2vec shortestst path finding model.
The poor resulting MSE results are output at the end of execution

"""


# This file trains and evalueates the model

# get current working directory
cwd = os.getcwd()

# get train, test, cv file paths
trainPath = osp.join(cwd, "outputs", "train_xy_no_sampling_stdScale.pk")
cvPath = osp.join(cwd, "outputs", "val_xy_no_sampling_stdScale.pk")
testPath = osp.join(cwd, "outputs", "test_xy_no_sampling_stdScale.pk")
trainCombineSample = osp.join(cwd, "outputs", "train_xy_combine_sampling.pk")
print(trainPath)
print(cvPath)
print(testPath)

# Read saved files into numpy arrays
x_train, y_train = pickle.load(open(trainPath, 'rb'))
x_cv, y_cv = pickle.load(open(cvPath, 'rb'))
x_test, y_test = pickle.load(open(testPath, 'rb'))
print('shapes of train, validation, test data ', x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape) 
values, counts = np.unique(y_train, return_counts=True)
num_features = x_train.shape[1]
print('Frequency of distance values before sampling ', values, counts)


# as data is imbalanced, oversample the minority targets and undersample the majority type samples
# the fraction for over/under sampling was chosen
seedRandom = 9999

# trial values for underSampling to work

x = int(counts[2] * 0.4)
y = int(0.4 * x)

undersample_dict = {3:y, 3:x}  # change for ValueError in under-sampling methods line in x_train, y_train = under_sampler.fit_resample(x_train, y_train.astype(int))
print(undersample_dict)
under_sampler = RandomUnderSampler(sampling_strategy=undersample_dict, random_state=seedRandom)  # n_jobs = 15
x_train, y_train = under_sampler.fit_resample(x_train, y_train.astype(int))
print('Frequency of distance values after undersampling', np.unique(y_train, return_counts=True))

# trial values for overSampling to work

minority_samples = int(7.3 * x)  #changed for x_train, y_train = over_sampler.fit_resample(x_train, y_train.astype(int)) to work
oversample_dict = {1:minority_samples, 4:minority_samples, 5:minority_samples, 6:minority_samples, 7:minority_samples}
over_sampler = RandomOverSampler(sampling_strategy=oversample_dict, random_state=seedRandom)  # ,n_jobs=15, k_neighbors= 5
x_train, y_train = over_sampler.fit_resample(x_train, y_train.astype(int))
print('Frequency of distance values after oversampling ', np.unique(y_train, return_counts=True))

pickle.dump((x_train, y_train), open(trainCombineSample, 'wb'))

print(x_train.shape, y_train.shape)

# shuffle data after over/under sampling
x_train, y_train = shuffle_arrays_unison(arrays=[x_train, y_train], random_seed=np.random.seed(999))


print(x_train.shape, y_train.shape)

# Create a baseline for this dataset by training a Linear Regression model
baseline_model = LinearRegression(fit_intercept=True, n_jobs=-1).fit(x_train, y_train)

y_pred = baseline_model.predict(x_test)
y_class = np.round(y_pred)
baseline_acc = accuracy_score(y_test, y_class)*100
baseline_mse = mean_squared_error(y_test, y_pred)
baseline_mae = mean_absolute_error(y_test, y_pred)
print("Baseline: Accuracy={}%, MSE={}, MAE={}".format(round(baseline_acc, 2), round(baseline_mse,2), round(baseline_mae,2)))

# Create pytorch data loaders for train/val/test datasets

params = {'batch_size': 1000, 'input_size': num_features, 'hidden_units_1': 200, 'hidden_units_2': 100, 'hidden_units_3': 50, 'do_1': 0.2, 'do_2': 0.1, 'do_3': 0.05, 'output_size': 1, 'lr': 0.001, 'min_lr': 1e-5, 'max_lr': 1e-3, 'epochs': 500, 'lr_sched': 'clr', 'lr_sched_mode': 'triangular', 'gamma': 0.95}

print(params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

trainset = torchData.TensorDataset(torch.as_tensor(x_train, dtype=torch.float, device=device), torch.as_tensor(y_train, dtype=torch.float, device=device))
train_dl = torchData.DataLoader(trainset, batch_size=params['batch_size'], drop_last=True)

val_dl = torchData.DataLoader(torchData.TensorDataset(torch.as_tensor(x_cv, dtype=torch.float, device=device), torch.as_tensor(y_cv, dtype=torch.float, device=device)), batch_size=params['batch_size'], drop_last=True)

test_dl = torchData.DataLoader(torchData.TensorDataset(torch.as_tensor(x_test, dtype=torch.float, device=device), torch.as_tensor(y_test, dtype=torch.float, device=device)), batch_size=params['batch_size'], drop_last=True)

# Check for batches with all same type of samples (same distance samples)

print('value counts in whole data', np.unique(y_train, return_counts=True))
count=0
for i, data in enumerate(train_dl, 0):
    input, target = data[0], data[1]
    t = torch.unique(target, return_counts=True)[1]
    if( t==params['batch_size']).any().item():
        count += 1
print('{} ({}%) batches have all same targets'.format(count, np.round(count/len(train_dl)*100, 2) ))


# Initialise model, loss, learning rate schedulers etc

torch.manual_seed(9999)
def getModel():
    """
    Creates a PyTorch model. Change the 'params' dict above to
    modify the neural net configuration 
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(params['input_size'], params['hidden_units_1']),
        torch.nn.BatchNorm1d(params['hidden_units_1']),
        torch.nn.ReLU(),
        torch.nn.Linear(params['hidden_units_1'], params['hidden_units_2']),
        torch.nn.BatchNorm1d(params['hidden_units_2']),
        torch.nn.ReLU(),
        torch.nn.Linear(params['hidden_units_2'], params['hidden_units_3']),
        torch.nn.BatchNorm1d(params['hidden_units_3']),
        torch.nn.ReLU(),
        torch.nn.Linear(params['hidden_units_3'], params['output_size']),
        torch.nn.ReLU(),
        # torch.nn.Softplus(),
    )
    model.to(device)
    return model

def poissonLoss(y_pred, y_true):
    """
    Custom loss function for Poisson model.
    Equivalent Keras implementation for reference:
    K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)
    For output of shape (2,3) it return (2,) vector. 
    """
    y_pred = torch.squeeze(y_pred)
    loss = torch.mean(y_pred - y_true * torch.log(y_pred+1e-7))
    return loss


model = getModel()

print('model loaded into device=', next(model.parameters()).device)

# capture model summary as string
oldStdout = sys.stdout
sys.stdout = buffer = io.StringIO()

summary(model, input_size=(params['input_size'], ))

sys.stdout = oldStdout
modelSummary = buffer.getvalue()
print('model-summary\n', modelSummary)
# this model-summary can be written to the tensorboard

lr_reduce_patience = 2.0
lr_reduce_factor = 0.1

loss_fn = poissonLoss
optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

if params['lr_sched'] == 'reduce_lr_plateau':
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor, patience=lr_reduce_patience, verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=1e-9, eps=1e-08)
elif params['lr_sched'] == 'clr':
         lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, params['min_lr'], params['max_lr'], step_size_up=8*len(train_dl), step_size_down=None, mode=params['lr_sched_mode'], last_epoch=-1, gamma=params['gamma'])

print('lr scheduler type:', lr_sched)
for param_group in optimizer.param_groups:
    print(param_group['lr'])


def evaluate(model, dl):
    """
    This function is used to evaluate the model with validation.
    args: model and data loader
    returns: loss
    """
    model.eval()
    final_loss = 0.0
    count = 0
    with torch.no_grad():
        for data_cv in dl:
            inputs, dist_true = data_cv[0], data_cv[1]
            count += len(inputs)
            outputs = model(inputs)
            loss = loss_fn(outputs, dist_true)
            final_loss += loss.item()
    return final_loss/len(dl)

def save_checkpoint(state, state_save_path):
    if not os.path.exists("/".join(state_save_path.split('/')[:-1])):
        os.makedirs("/".join(state_save_path.split('/')[:-1]))
    torch.save(state, state_save_path)


# train the model and record results in tensorboard

last_loss = 0.0
min_val_loss = np.inf
patience_counter = 0
early_stop_patience = 50
best_model = None
train_losses = []
val_losses = []

output_path = osp.join(cwd, "outputs")
tb_path = output_path+'/logs/runs'
#run_path = tb_path+'/run47_smallerNN_noDO'
run_path = tb_path+'/run2Weighted_GitHubOriginal_noDO'
checkpoint_path = run_path+'/checkpoints'
resume_training = False
start_epoch = 0
iter_count = 0

if os.path.exists(run_path):
     print("this experiment already exists!")
else:


    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(log_dir=run_path, comment='', purge_step=None, max_queue=1, flush_secs=30, filename_suffix='')
    writer.add_graph(model, input_to_model=torch.zeros(params['input_size']).view(1,-1).cuda(), verbose=False)  # not useful

    # resume training on a saved model
    if resume_training:
        # change this
        prev_checkpoint_path = osp.join(cwd, "outputs", "logs", "runs", "run42_clr_g0.95", "checkpoints")
        suffix = '1592579305.7273214'  # change this
        model.load_state_dict(torch.load(prev_checkpoint_path+'/model_'+suffix+'.pt'))
        optimizer.load_state_dict(torch.load(prev_checkpoint_path+'/optim_'+suffix+'.pt'))
        lr_sched.load_state_dict(torch.load(prev_checkpoint_path+'/sched_'+suffix+'.pt'))
        state = torch.load(prev_checkpoint_path+'/state_'+suffix+'.pt') 
        start_epoch = state['epoch']
        writer.add_text('loaded saved model:', str(params))
        print('loaded saved model', params)

    writer.add_text('run_change', 'Smaller 3 hidden layer NN, no DO' + str(params))

    torch.backends.cudnn.benchmark = True
    print('total epochs=', len(range(start_epoch, start_epoch+params['epochs'])))

    with torch.autograd.detect_anomaly():  # use this to detect bugs while training
        for param_group in optimizer.param_groups:
            print('lr-check', param_group['lr'])
        for epoch in range(start_epoch, start_epoch+params['epochs']):  # loop over the dataset multiple times
            running_loss = 0.0
            stime = time.time()

            for i, data in enumerate(train_dl, 0):
                iter_count += 1
                # get the inputs; data is a list of [inputs, dist_true]
                model.train()
                inputs, dist_true = data[0], data[1]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_fn(outputs, dist_true)

                loss.backward(retain_graph=True)     # issue here
                #optimizer.step()

                running_loss += loss.item()
                last_loss = loss.item()

                loss.backward()      # issue here
                optimizer.step()

                running_loss += loss.item()
                last_loss = loss.item()

                for param_group in optimizer.param_groups:
                    curr_lr = param_group['lr']
                writer.add_scalar('monitor/lr-iter', curr_lr, iter_count-1) 

                if not isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_sched.step()

            val_loss = evaluate(model, val_dl)  
            if isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_sched.step(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
                best_model = copy.deepcopy(model)
                print(epoch,"> Best val_loss model saved:", round(val_loss, 4))
            else:
                patience_counter += 1
            train_loss = running_loss/len(train_dl)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
            writer.add_scalar('monitor/lr-epoch', curr_lr, epoch)
            if patience_counter > early_stop_patience:
                print("Early stopping at epoch {}. current val_loss {}".format(epoch, val_loss))
                break

            if epoch % 10 == 0:
                torch.save(best_model.state_dict(), checkpoint_path+'/model_cp.pt')
                torch.save(optimizer.state_dict(), checkpoint_path+'/optim_cp.pt')
                torch.save(lr_sched.state_dict(), checkpoint_path+'/sched_cp.pt')
                writer.add_text('checkpoint saved', 'at epoch='+str(epoch))
                print("epoch:{} -> train_loss={},val_loss={} - {}".format(epoch, round(train_loss, 5),round(val_loss, 5), time.time()-stime))

print('Finished Training')


best_model_path = checkpoint_path+'/model_cp.pt'
opt_save_path = checkpoint_path+'/optim_cp.pt'
sched_save_path = checkpoint_path+'/sched_cp.pt'


# test the model with test data

# use data in this function to identify how to query this model
# not use of dl (test_dl) which contains x_test and y_test
# in formDistMap.py, x_test and y_test have different shapes from the same emd_dist_pair of the same graph
# test_dl = torchData.DataLoader(torchData.TensorDataset(torch.as_tensor(x_test, dtype=torch.float, device=device), torch.as_tensor(y_test, dtype=torch.float, device=device)), batch_size=params['batch_size'], drop_last=True)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=seedRandom, shuffle=True)
# y_hat contains outputs
def test(model, dl):
    model.eval()
    final_loss = 0.0
    count = 0
    y_hat = []
    with torch.no_grad():
        
        for data_cv in dl:
            inputs, dist_true = data_cv[0], data_cv[1]
            #print(f"True Distances length: {len(dist_true)}")
            count += len(inputs)
            outputs = model(inputs)
            #print(f"Outputs Length: {len(outputs)}")
            y_hat.extend(outputs.tolist())
            loss = loss_fn(outputs, dist_true)
            final_loss += loss.item()
        print(f"Length of data file: {len(dl)}")
    return final_loss/len(dl), y_hat

model.load_state_dict(torch.load(best_model_path))
test_loss, y_hat = test(model, test_dl)
print(test_loss)
#writer.add_text('test-loss', str(test_loss))
try:
    if MinMaxScaler:
        y_hat = MinMaxScaler.inverse_transform(y_hat)
        y_test = MinMaxScaler.inverse_transform(y_test)
except:
    pass

print(y_hat[50:60], y_test[50:60])

#writer.add_text('Accuracy=', str(accuracy_score(y_test[:len(y_hat)], np.round(y_hat))))
print(str(accuracy_score(y_test[:len(y_hat)], np.round(y_hat))))

# show distance value wise precision (bar chart)
# predicted values are less than real test samples because last samples from test are dropped to maintain same batch size (drop_last=True)

y_hatNew = np.array(y_hat).squeeze()
y_testNew = y_test[:len(y_hat)]
print(f"Number of test values = {len(y_test)}, number of output values = {len(y_hat)}")

# need to finds what the indices of y_test and y_hat refer to
# then find how this model can be used for entirely different graphs
print("First 20 y_hat (model) values vs y_test (actual) values:")
for i in range(0, 20):
    print(f"y_hat[i] = {y_hat[i]}, y_test[i] = {y_test[i]}")

print("Last 20 y_hat (model) values:")
for i in range(len(y_hat)-20, len(y_hat)):
    print(f"y_hat[i] = {y_hat[i]}")

print("Last 20 y_test (model) values:")
for i in range(len(y_test)-20, len(y_test)):
    print(f"y_test[i] = {y_test[i]}")

dist_accuracies = []
dist_counts = []
#rangLength = len(y_testNew)
rangLength = 26
for i in range(1, rangLength):
    mask = y_testNew==i
    dist_values = y_testNew[mask]
    #print("Distance Values:")
    #print(dist_values)
    dist_preds = np.round(y_hatNew[mask])
    #print("Distance Predictions:")
    #print(dist_preds)
    dist_accuracies.append(np.sum(dist_values == dist_preds)*100/len(dist_values))
    dist_counts.append(len(dist_values))



fig = plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.bar(range(1,rangLength), dist_accuracies)
for index, value in enumerate(dist_accuracies):
    plt.text(index+1, value, str(np.round(value, 2))+'%', ha='center', va='bottom', rotation=90)
plt.title('distance-wise accuracy')
plt.xlabel('distance values')
plt.ylabel('accuracy')
plt.subplot(2, 1, 2)
plt.bar(range(1, rangLength), dist_counts)
for index, value in enumerate(dist_counts):
    plt.text(index+1, value, str(value), ha='center', va='bottom', rotation=90)
plt.title('distance-wise count')
plt.xlabel('distance values')
plt.ylabel('counts')
fig.tight_layout(pad=3.0)
plt.show()
#writer.add_figure('test/results', fig)
#writer.add_text('class avg accuracy', str(np.mean(dist_accuracies)))
print('class avh accuracy', np.mean(dist_accuracies))

#writer.add_text('MSE', str(np.mean((np.array(y_hat).squeeze()-y_test[:len(y_hat)])**2)))
print('MSE', np.mean((np.array(y_hat).squeeze()-y_test[:len(y_hat)])**2))

#writer.add_text('MAE', str(np.mean(np.abs(np.array(y_hat).squeeze() - y_test[:len(y_hat)]))))
print('MAE', np.mean(np.abs(np.array(y_hat).squeeze() - y_test[:len(y_hat)])))