import time as TT
import tensorflow.keras.backend as K
from deepxde.data.sampler import BatchSampler
from deepxde.data.data import Data
import sys
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from deepxde.backend import tf
# tf.config.optimizer.set_jit(False)
import os
import deepxde as dde
print(dde.__version__)
dde.config.disable_xla_jit()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


m = 101  # Number of points in the input load amplitude
N_component = 2
HIDDEN = 32
batch_size = 64
fraction_train = 0.8
N_epoch = 300000
N_data = 4000
# N_data = 9900
use_existing_index = False
# skip_ = 1
# N_output_frame = 40 // skip_
N_output_frame = 1
sub = '_minmax_HD32_vec2_selectRELU_4000'


print('\n\nModel parameters:')
print(sub)
print('batch_size  ', batch_size)
print('HIDDEN  ', HIDDEN)
print('N_output_frame  ', N_output_frame)
print('N_component  ', N_component)
print('m  ', m)
print('fraction_train  ', fraction_train)
print('\n\n\n')


seed = 123
tf.keras.backend.clear_session()
try:
    tf.keras.utils.set_random_seed(seed)
except:
    pass
dde.config.set_default_float("float64")


#######################################################################################################################
# Define model
# /u/junyanhe/.local/lib/python3.9/site-packages/deepxde

class DeepONetCartesianProd(dde.maps.NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(
                activation["trunk"])
        else:
            activation_branch = self.activation_trunk = dde.maps.activations.get(
                activation)

        # User-defined network
        self.branch = layer_sizes_branch[1]
        self.trunk = layer_sizes_trunk[0]
        # self.b = tf.Variable(tf.zeros(1),dtype=np.float64)
        self.b = tf.Variable(tf.zeros(1, dtype=dde.config.real(tf)))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        # Branch net to encode the input function
        x_func = self.branch(x_func)  # [ bs , HD , N_TS ]
        # N_TS is number of time steps
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))  # [ N_pts , HD ]
        # Dot product, batch_size, N_TS, N_pts, c(number of vector components)
        x = tf.einsum("bht,nhc->btnc", x_func, x_loc)
        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        # Selectively apply relu
        x_final = tf.stack([x[:, :, :, 0], tf.nn.relu(x[:, :, :, 1])], axis=-1)
        return x_final
        # return tf.nn.relu( x )


branch = tf.keras.models.Sequential([
    tf.keras.layers.GRU(units=256, batch_input_shape=(batch_size, m, 1),
                        activation='tanh', return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
    tf.keras.layers.GRU(units=128, activation='tanh',
                        return_sequences=False, dropout=0.00, recurrent_dropout=0.00),
    tf.keras.layers.RepeatVector(HIDDEN),
    tf.keras.layers.GRU(units=128, activation='tanh',
                        return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
    tf.keras.layers.GRU(units=256, activation='tanh',
                        return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(N_output_frame))])
branch.summary()

my_act = "relu"
trunk = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2,)),
    tf.keras.layers.Dense(101, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(HIDDEN * N_component,
                          activation=my_act, kernel_initializer='GlorotNormal'),
    tf.keras.layers.Reshape([HIDDEN, N_component]),
])
trunk.summary()


net = DeepONetCartesianProd(
    [m, branch], [trunk], my_act, "Glorot normal")


#######################################################################################################################
# Load data
# base = '/projects/bblv/skoric/DEEPXDE_TEST_VECTOR_DOG_BONE_PLASTIC/Data'
base = "../data/dogbone"
Train_and_test_Amp = np.load(
    base+'/Amp.npz')['a'].astype(np.float64)[:N_data]  # (9903,101)
# Stress = np.load(base+'/Stress.npz')['a'].astype(np.float64)[:N_data,::skip_,:] # (9903, 40, 3060)
# PEEQ = np.load(base+'/PEEQ.npz')['a'].astype(np.float64)[:N_data,::skip_,:] # (9903, 40, 3060)

Stress = np.load(
    base+'/Stress.npz')['a'].astype(np.float64)[:N_data, -1, :]  # (9903,3060)
Stress = Stress.reshape(Stress.shape[0], 1, Stress.shape[1])  # (9903, 1, 3060)
print("Stress.shape = ", Stress.shape)

# (9903,3060)
PEEQ = np.load(base+'/PEEQ.npz')['a'].astype(np.float64)[:N_data, -1, :]
PEEQ = PEEQ.reshape(PEEQ.shape[0], 1, PEEQ.shape[1])  # (9903, 1, 3060)
print("PEEQ.shape = ", PEEQ.shape)

xy_train_testing = np.load(
    base+'/Coords.npy').astype(np.float64)[:N_data]  # (3060, 2)


#########################################################################################
# Cap and scale, stress
smax = 260.
flag = Stress > smax
print('Capped ', np.sum(flag) / float(len(flag.flatten()))
      * 100, ' percent stress data points')
Stress[flag] = smax

Sshape = Stress.shape
Stress = Stress.reshape([Sshape[0]*Sshape[1], Sshape[2]])
# Scale
scaler = MinMaxScaler()
scaler.fit(Stress)
Stress = scaler.transform(Stress).reshape(Sshape)

#########################################################################################
# Scale, PEEQ
smax = 0.6 * np.max(PEEQ)
flag = PEEQ > smax
print('Capped ', np.sum(flag) / float(len(flag.flatten()))
      * 100, ' percent PEEQ data points')
PEEQ[flag] = smax

PEEQ = PEEQ.reshape([Sshape[0]*Sshape[1], Sshape[2]])

# Scale
scaler2 = MinMaxScaler()
scaler2.fit(PEEQ)
PEEQ = scaler2.transform(PEEQ).reshape(Sshape)


##########################################################################################
# Combine data
s_data = np.stack([Stress, PEEQ], axis=-1)


N_valid_case = len(Stress)
N_train = int(N_valid_case * fraction_train)

"""

if use_existing_index:
    train_case = np.load('TrainIndex.npy')
else:
    train_case = np.random.choice( N_valid_case , N_train , replace=False )
    np.save('TrainIndex.npy',train_case)

test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )
"""
print('Training with ', N_train, ' points')


u0_train = Train_and_test_Amp[:N_train]
u0_testing = Train_and_test_Amp[N_train:]
s_train = s_data[:N_train]
s_testing = s_data[N_train:]


###################################################################################
"""
for i in range( N_component ):
    s0_plot = s_train[:,:,:,i].flatten()
    s1_plot = s_testing[:,:,:,i].flatten()
    plt.hist( s0_plot , bins=50 , color='r' , alpha=0.6 , density=True )
    plt.hist( s1_plot , bins=50 , color='b' , alpha=0.6 , density=True )
    plt.legend(['Training' , 'Testing'])
    plt.savefig('train_test_dist_comp'+str(i)+'.pdf')
    plt.close()
###################################################################################
"""


print('u0_train.shape = ', u0_train.shape)
print('u0_testing.shape = ', u0_testing.shape)
print('s_train.shape = ', s_train.shape)
print('s_testing.shape = ', s_testing.shape)
print('xy_train_testing.shape', xy_train_testing.shape)

x_train = (u0_train, xy_train_testing)
y_train = s_train
x_test = (u0_testing, xy_train_testing)
y_test = s_testing


class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y


data = TripleCartesianProd(x_train, y_train, x_test, y_test)


#######################################################################################################################
# Loss
def MSE(y_true, y_pred):
    tmp = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
    return tf.math.reduce_mean(tmp)

# Metrics


def err(y_train, y_pred):
    ax = -1
    return np.linalg.norm(y_train - y_pred, axis=ax) / (np.linalg.norm(y_train, axis=ax) + 1e-8)


def L2_S(y_train, y_pred):
    my_shape = y_train.shape[:-1]
    y_train_original = scaler.inverse_transform(y_train[:, :, :, 0].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    y_pred_original = scaler.inverse_transform(y_pred[:, :, :, 0].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    return np.mean(err(y_train_original, y_pred_original).flatten())


def ABS_S(y_train, y_pred):
    my_shape = y_train.shape[:-1]
    y_train_original = scaler.inverse_transform(y_train[:, :, :, 0].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    y_pred_original = scaler.inverse_transform(y_pred[:, :, :, 0].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    return np.mean(np.abs(y_train_original - y_pred_original).flatten())


def L2_EP(y_train, y_pred):
    my_shape = y_train.shape[:-1]
    y_train_original = scaler2.inverse_transform(y_train[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    y_pred_original = scaler2.inverse_transform(y_pred[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    return np.mean(err(y_train_original, y_pred_original).flatten())


def ABS_EP(y_train, y_pred):
    my_shape = y_train.shape[:-1]
    y_train_original = scaler2.inverse_transform(y_train[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    y_pred_original = scaler2.inverse_transform(y_pred[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    return np.mean(np.abs(y_train_original - y_pred_original).flatten())


# Build model
model = dde.Model(data, net)
# Stage 1 training
model.compile(
    "adam",
    lr=5e-4,
    decay=("inverse time", 1, 1e-4),
    loss=MSE,
    metrics=[L2_S, ABS_S, L2_EP, ABS_EP],
)
losshistory1, train_state1 = model.train(
    iterations=N_epoch, batch_size=batch_size, model_save_path="./mdls/TrainedModel"+sub)
np.save('losshistory'+sub+'.npy', losshistory1)


st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ', duration, ' s')
print('Prediction speed = ', duration / float(len(y_pred)), ' s/case')

my_shape = y_test.shape[:-1]
# Stress
y_test_original1 = scaler.inverse_transform(y_test[:, :, :, 0].reshape(
    [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
y_pred_original1 = scaler.inverse_transform(y_pred[:, :, :, 0].reshape(
    [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)

# PEEQ
y_test_original2 = scaler2.inverse_transform(y_test[:, :, :, 1].reshape(
    [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
y_pred_original2 = scaler2.inverse_transform(y_pred[:, :, :, 1].reshape(
    [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)

np.savez_compressed('TestData'+sub+'.npz', a=y_test_original2, b=y_test_original1,
                    c=y_pred_original2, d=y_pred_original1, f=xy_train_testing, e=u0_testing)


error_s = err(y_test_original1, y_pred_original1)
# error_s[ error_s > 1. ] = 1.
# np.save( 'Stress_errors'+sub+'.npy' , error_s )
print('mean of relative L2 error of Stress: {:.2e}'.format(np.mean(error_s)))
print('std of relative L2 error of Stress: {:.2e}'.format(np.std(error_s)))

# error_s = err( y_test_original2 , y_pred_original2 )
# error_s[ error_s > 1. ] = 1.
# np.save( 'PEEQ_errors'+sub+'.npy' , error_s )
# print('mean of relative L2 error of PEEQ: {:.2e}'.format( np.mean(error_s) ))
# print('std of relative L2 error of PEEQ: {:.2e}'.format( np.std(error_s) ))
error_t = np.abs(y_test_original2 - y_pred_original2).flatten()
print('mean of MAE error of PEEQ: {:.2e}'.format(np.mean(error_t)))
print('std of MAE error of PEEQ: {:.2e}'.format(np.std(error_t)))
