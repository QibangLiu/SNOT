import time as TT
import tensorflow.keras.backend as K
from deepxde.data.sampler import BatchSampler
from deepxde.data.data import Data
import deepxde as dde
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from deepxde.backend import tf
tf.config.optimizer.set_jit(True)  # This_line_here_here
dde.config.disable_xla_jit()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# data_loc = '/scratch/bblv/sk89/DeepOnet_MultiPhysics/COUPLED_SLICE_UMAT/Data_prep_S_DeepONet'
# data_loc = '/scratch/bblv/sk89/DeepOnet_MultiPhysics/COUPLED_SLICE_UMAT'
# data_loc = '/scratch/bblv/skoric/PLASTIC_3D_LUG/GENERATE_DATA_5000/EXTRACTION_FROM_ODB'
# data_loc = '/scratch/bblv/skoric/PLASTIC_3D_LUG/GENERATE_DATA_5000_REFINED_30K_NODES/EXTRACTION_FROM_ODB'
data_loc = '../data/LUG_3D'


total_sims = 10000
# n_step = 101
n_step = 1
n_nodes = 29852

HIDDEN = 256
N_input_fn = 1
N_component = 2  # Predict temp and stress
N_output_frame = 1  # First, predicting the last frame

m = 101
batch_size = 64
seed = 2024
try:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
except:
    pass
dde.config.set_default_float("float64")


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

        # print( x_func.shape )
        # print( x_loc.shape )
        # exit()

        # Branch net to encode the input function
        x_func = self.branch(x_func)  # [ bs , HD , N_TS ]
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(
            self.trunk(x_loc))  # [ N_pts , HD , N_comp ]

        # Dot product
        x = tf.einsum("bht,nhc->btnc", x_func, x_loc)

        # Add bias
        x += self.b

        # if self._output_transform is not None:
        # 	x = self._output_transform(inputs, x)
        return tf.math.sigmoid(x)  # This_line_different_here_here


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


xy_train_testing = np.load(os.path.join(data_loc, 'xyz_coords.npy'))
# turn coordinates into mm
xy_train_testing = xy_train_testing * 1000.00


# s
data_t = np.load(os.path.join(data_loc, 'peeq_all_new_10K.npy'))
data_s = np.load(os.path.join(data_loc, 'ystress_all_new_10K.npy'))

# stress data in Mpa
data_s = data_s / 1.e06

# cap peeq to 5% only
# data_t[data_t > 0.05] = 0.05

print("data_t.shape  = ", data_t.shape)
print("data_s.shape  = ", data_s.shape)

Heat_Amp = np.load(os.path.join(data_loc, 'data_amp_10K.npy'))

n_cases = len(Heat_Amp)
print("n_cases = ", n_cases)


# Scale
scalerT = MinMaxScaler()
# scalerT = PowerTransformer()
# scalerT = MaxAbsScaler()
# scalerT = StandardScaler()
scalerT.fit(data_t)
scaled_temp = scalerT.transform(data_t)
# Survey(scaled_temp)

scalerS = MinMaxScaler()
# scalerS = PowerTransformer()
# scalerS = MaxAbsScaler()
# scalerS = StandardScaler()
scalerS.fit(data_s)
scaled_stress = scalerS.transform(data_s)
# Survey(scaled_stress)


Temp = np.zeros((n_cases, N_output_frame, n_nodes, N_component))
# using only last frame since N_output_frame = 1
Temp[:, -1, :n_nodes, 0] = scaled_temp
Temp[:, -1, :n_nodes, 1] = scaled_stress
print('Temp shape: ', Temp.shape)


# u0_train = Heat_Amp[:4900]
# u0_testing = Heat_Amp[4900:]

# s_train = Temp[:4900]
# s_testing = Temp[4900:]

fraction_train = 0.90
N_train = int(n_cases * fraction_train)
print("N_train = ", N_train)
"""
train_case = np.random.choice(n_cases , N_train , replace=False )
test_case = np.setdiff1d( np.arange(n_cases) , train_case )
print('Training with ' , N_train , ' points')

u0_train = Heat_Amp[train_case]
u0_testing =  Heat_Amp[test_case]
s_train = Temp[train_case]
s_testing = Temp[test_case]
"""
u0_train = Heat_Amp[:N_train]
u0_testing = Heat_Amp[N_train:]
s_train = Temp[:N_train]
s_testing = Temp[N_train:]


print('u0_train.shape = ', u0_train.shape)
print('type of u0_train = ', type(u0_train))
print('u0_testing.shape = ', u0_testing.shape)
print('s_train.shape = ', s_train.shape)
print('s_testing.shape = ', s_testing.shape)
print('xy_train_testing.shape', xy_train_testing.shape)

x_train = (u0_train, xy_train_testing)
y_train = s_train
x_test = (u0_testing, xy_train_testing)
y_test = s_testing
# data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
data = TripleCartesianProd(x_train, y_train, x_test, y_test)

branch = tf.keras.models.Sequential([
    tf.keras.layers.GRU(units=256, batch_input_shape=(batch_size, m, N_input_fn),
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
    tf.keras.layers.InputLayer(input_shape=(3,)),
    tf.keras.layers.Dense(256, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(256, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(256, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(256, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(256, activation=my_act,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(HIDDEN * N_component,
                          activation=my_act, kernel_initializer='GlorotNormal'),
    tf.keras.layers.Reshape([HIDDEN, N_component]),
])
trunk.summary()

net = DeepONetCartesianProd(
    [m, branch], [trunk], my_act, "Glorot normal")

model = dde.Model(data, net)
print("y_train shape:", y_train.shape)


def MSE(y_true, y_pred):
    tmp = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
    data_loss = tf.math.reduce_mean(tmp)
    return data_loss


def ABS(y_true, y_pred):
    tmp = tf.math.abs(K.flatten(y_true) - K.flatten(y_pred), name='abs')
    data_loss = tf.math.reduce_mean(tmp)
    return data_loss

# Metrics


def err(y_train, y_pred):
    ax = -1
    return np.linalg.norm(y_train - y_pred, axis=ax) / (np.linalg.norm(y_train, axis=ax) + 1e-8)


def L2_S(y_train, y_pred):
    my_shape = y_train.shape[:-1]
    y_train_original = scalerS.inverse_transform(y_train[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    y_pred_original = scalerS.inverse_transform(y_pred[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    return np.mean(err(y_train_original, y_pred_original).flatten())


def ABS_S(y_train, y_pred):
    my_shape = y_train.shape[:-1]
    y_train_original = scalerS.inverse_transform(y_train[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    y_pred_original = scalerS.inverse_transform(y_pred[:, :, :, 1].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    return np.mean(np.abs(y_train_original - y_pred_original).flatten())


def ABS_EP(y_train, y_pred):
    my_shape = y_train.shape[:-1]
    y_train_original = scalerT.inverse_transform(y_train[:, :, :, 0].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    y_pred_original = scalerT.inverse_transform(y_pred[:, :, :, 0].reshape(
        [my_shape[0]*my_shape[1], my_shape[2]])).reshape(my_shape)
    return np.mean(np.abs(y_train_original - y_pred_original).flatten())


model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 1, 1e-4),
    loss=ABS,
    metrics=[L2_S, ABS_S, ABS_EP],
)
# losshistory, train_state = model.train(iterations=350000, batch_size=batch_size, model_save_path="./mdls/TrainFrac_"+str(idx) )
# losshistory, train_state = model.train(iterations=200000, batch_size=batch_size, model_save_path="TrainFrac_"+str(idx) )
losshistory, train_state = model.train(
    iterations=300000, batch_size=batch_size, model_save_path="3D_Lug_30K_Nodes_ABS_200K")
np.save('losshistory_3D_Lug_30K_Nodes_ABS_300K.npy', losshistory)


st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Inference took ', duration, ' s')
print('Prediction speed = ', duration / float(len(y_pred)), ' s/case')

my_shape = y_test.shape[:-1]


# Stress
y_test_stress = scalerS.inverse_transform(
    y_test[:, :, :, 1].reshape([my_shape[0]*my_shape[1], my_shape[2]]))
y_pred_stress = scalerS.inverse_transform(
    y_pred[:, :, :, 1].reshape([my_shape[0]*my_shape[1], my_shape[2]]))

# PEEQ
y_test_peeq = scalerT.inverse_transform(
    y_test[:, :, :, 0].reshape([my_shape[0]*my_shape[1], my_shape[2]]))
y_pred_peeq = scalerT.inverse_transform(
    y_pred[:, :, :, 0].reshape([my_shape[0]*my_shape[1], my_shape[2]]))

print("y_test_peeq.shape = ", y_test_peeq.shape)
print("y_pred_peeq.shape =", y_pred_peeq.shape)

print("y_test_stress.shape = ", y_test_stress.shape)
print("y_pred_stress.shape =", y_pred_stress.shape)

# absolute difference of each 2D element
abs_error_p = np.abs(y_test_peeq - y_pred_peeq)

# mean absolute error of each row, sample
abs_error_p_samp = np.mean(abs_error_p, axis=1)


# mean absolute error for all rows (samples)
print('abs_error_p_samp = ', abs_error_p_samp)
print('mean of absolute error of Peeq: {:.2e}'.format(
    np.mean(abs_error_p_samp)))
print('std of absolute error of Peeq: {:.2e}'.format(np.std(abs_error_p_samp)))


# L2 stress error for each row (sample)
l2_error_s_samp = err(y_test_stress, y_pred_stress)
# mean L2 stress error for all rows (samples)
print('l2_error_s_samp = ', l2_error_s_samp)
print('mean of L2 error of Stress: {:.2e}'.format(np.mean(l2_error_s_samp)))
print('std of L2 error of Stress: {:.2e}'.format(np.std(l2_error_s_samp)))

np.savez_compressed('Test_s-don_3DLug_results.npz', a=y_test_peeq,
                    b=y_test_stress, c=y_pred_peeq, d=y_pred_stress, e=xy_train_testing)

"""
		np.save('losshistory'+str(idx)+'.npy',losshistory)

		st = TT.time()
		y_pred = model.predict(data.test_x)
		duration = TT.time() - st
		print('y_pred.shape =', y_pred.shape)
		print('Prediction took ' , duration , ' s' )
		print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )
		#np.savez_compressed('TestData'+str(idx)+'.npz',a=y_test,b=y_pred,c=u0_testing,d=xy_train_testing)
		np.savez_compressed('Ver_2_TestData'+str(idx)+'.npz',a=y_test,b=y_pred,c=u0_testing,d=xy_train_testing, e=train_case, f=test_case, g=losshistory, h=train_state)
		Org_temp_test = scalerT.inverse_transform(y_test[:,0,:,0])
		Org_stress_test = scalerS.inverse_transform(y_test[:,0,:,1])
		Org_temp_pred = scalerT.inverse_transform(y_pred[:,0,:,0])
		Org_stress_pred = scalerS.inverse_transform(y_pred[:,0,:,1])
		print('Successful to convert to original forms')
		print()
		np.savez_compressed('Ver_2_Org_y_try_'+str(idx)+'.npz', a=Org_temp_test, b=Org_stress_test, c=Org_temp_pred, d=Org_stress_pred)
		print('Saving all successful')

		error_s = []
		error_t = []
		org_error_s = []
		org_error_t = []
		for i in range(len(y_pred)):
			error_t_tmp = np.linalg.norm(y_test[i, 0, :, 0] - y_pred[i, 0, :, 0]) / np.linalg.norm(y_test[i, 0, :, 0])
			error_s_tmp = np.linalg.norm(y_test[i, 0, :, 1] - y_pred[i, 0, :, 1]) / np.linalg.norm(y_test[i, 0, :, 1])
			org_error_t_tmp = np.linalg.norm(Org_temp_test[i] - Org_temp_pred[i]) / np.linalg.norm(Org_temp_test[i])
			org_error_s_tmp = np.linalg.norm(Org_stress_test[i] - Org_stress_pred[i]) / np.linalg.norm(Org_stress_test[i])

			if error_s_tmp > 1:
				error_s_tmp = 1
			if error_t_tmp > 1:
				error_t_tmp = 1

			error_s.append(error_s_tmp)
			error_t.append(error_t_tmp)
			org_error_s.append(org_error_s_tmp)
			org_error_t.append(org_error_t_tmp)

		error_s = np.stack(error_s)
		error_t = np.stack(error_t)
		org_error_s = np.stack(org_error_s)
		org_error_t = np.stack(org_error_t)

		print()
		print("error_t = ", error_t)
		print()
		print('----------------------------------------')
		print()
		print("error_s = ", error_s)
		print('----------------------------------------')
		print()
		print()
		#Calculate mean and std for all testing data samples
		print('Scaled L2 error')
		print('mean of temperature relative L2 error of s: {:.2e}'.format(error_t.mean()))
		print('std of temperature relative L2 error of s: {:.2e}'.format(error_t.std()))
		print('--------------------------------------------------------------')
		print('mean of stress relative L2 error of s: {:.2e}'.format(error_s.mean()))
		print('std of stress relative L2 error of s: {:.2e}'.format(error_s.std()))
		print('--------------------------------------------------------------')
		print('--------------------------------------------------------------')
		print()
		print('Origianl L2 error')
		print('mean of temperature relative L2 error of s: {:.2e}'.format(org_error_t.mean()))
		print('std of temperature relative L2 error of s: {:.2e}'.format(org_error_t.std()))
		print('--------------------------------------------------------------')
		print('mean of stress relative L2 error of s: {:.2e}'.format(org_error_s.mean()))
		print('std of stress relative L2 error of s: {:.2e}'.format(org_error_s.std()))
		print('--------------------------------------------------------------')
		print('--------------------------------------------------------------')

		print()
		print()
		if True: #error_s.mean() < 0.03 and error_t.mean() < 0.03:
			break

		count = count + 1

	plt.hist( error_s.flatten() , bins=25 )
	plt.savefig('Stress_Err_hist_DeepONet'+str(idx)+'.jpg' , dpi=300)

	plt.hist( error_t.flatten() , bins=25 )
	plt.savefig('Temp_Err_hist_DeepONet'+str(idx)+'.jpg' , dpi=300)
"""
print('done with job')
