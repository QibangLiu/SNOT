import time as TT
import tensorflow.keras.backend as K
from deepxde.data.sampler import BatchSampler
from deepxde.data.data import Data
import deepxde as dde
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from deepxde.backend import tf
tf.config.optimizer.set_jit(True)  # This_line_here
# import keras.backend as K
dde.config.disable_xla_jit()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# data_loc = '/scratch/bblv/sk89/DeepOnet_MultiPhysics/COUPLED_SLICE_UMAT/Data_prep_S_DeepONet'
# data_loc = '/projects/bblv/sk89/DeepOnet_MultiPhysics/COUPLED_SLICE_UMAT'
# data_loc = '/scratch/bbkg/jaewanp2/Solidification_Multiphysics/Last_time_step/All_data_GRU_MinMax_COP_ver2/DATA'
# data_loc = '/projects/bblv/skoric/TMP/TMP/AM_S-DeepONet_SLICE_SOLIDIF_Last_Frame/DATA'
data_loc = '/projects/bblv/skoric/DEEPXDE_TEST_MULTIPHYSICS_MULTI_BRANCH/Data_Clean'
data_loc = '../data/steel_solidification'

n_cases = 5000
print("n_cases = ", n_cases)
n_step = 101
n_nodes = 602

HIDDEN = 100
N_input_fn = 2  # number of input functions (flux and disp amplitudes)
N_component = 2  # of predicted solution fileds,  temp and stress
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
        #       x = self._output_transform(inputs, x)
        # return tf.math.sigmoid(x) # This_line_different_here_here
        return x


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


# problem_data = [3222, 4150]
# failed_sims = np.load(os.path.join(data_loc, 'failed_sims_time_steps.npy')).astype(np.float64)
# failed_sims = np.append(failed_sims, problem_data)

# all_sims = np.arange(total_sims)
# successful_sims = np.setdiff1d(all_sims, failed_sims).astype(int)
# print('failed_sims', failed_sims)

trunk_scaling_factor = 1000
xy_train_testing = np.load(os.path.join(
    data_loc, 'xy_train_testing.npy')).astype(np.float64)
xy_train_testing = xy_train_testing * trunk_scaling_factor
'''
# Normalize
x = xy_train_testing[:,0]
xy_train_testing[:,0] = ( x - np.min(x) ) / ( np.max(x) - np.min(x) )
x = xy_train_testing[:,1]
xy_train_testing[:,1] = ( x - np.min(x) ) / ( np.max(x) - np.min(x) )
'''

# time = np.loadtxt(os.path.join(data_loc, 'time.txt'))
# flux = np.loadtxt(os.path.join(data_loc, 'flux_amp.txt'))
# disp = np.loadtxt(os.path.join(data_loc, 'disp_amp.txt'))
flux_filtered = np.load(os.path.join(data_loc, 'flux_filtered.npy'))[:n_cases]
print('flux_filtered shape')
print(flux_filtered.shape)

disp_filtered = np.load(os.path.join(data_loc, 'disp_filtered.npy'))[:n_cases]
print('disp_filtered shape')
print(disp_filtered.shape)

temp_data = np.load(os.path.join(data_loc, 'filtered_temp_data.npy'))[:n_cases]
# temp_data = np.delete(temp_data, problem_data, 0)
print('temp_data shape')
print(temp_data.shape)

stress_scaling_factor = 1
stress_data = np.load(os.path.join(
    data_loc, 'filtered_stress_data.npy'))[:n_cases]
# stress_data = np.delete(stress_data, problem_data, 0)
stress_data = stress_data / stress_scaling_factor
print('stress_data shape')
print(stress_data.shape)


# Filter the data based on successful simulations
# flux_filtered = flux[successful_sims]
# disp_filtered = disp[successful_sims]

# np.save("filtered_temp_data", temp_data)
# np.save("filtered_stress_data", stress_data)
# np.save("flux_filtered", flux_filtered)
# np.save("disp_filtered", disp_filtered)

# exit()


def Survey(data):
    print('Mean ', np.mean(data), ' , max ',
          np.max(data), ' , min ', np.min(data))


# Scale
scalerFlux = MinMaxScaler()
scalerFlux.fit(flux_filtered)
flux_filtered = scalerFlux.transform(flux_filtered)
print('flux_filtered')
Survey(flux_filtered)
print('----------')

scalerDisp = MinMaxScaler()
scalerDisp.fit(disp_filtered)
disp_filtered = scalerDisp.transform(disp_filtered)
print('disp_filtered')
Survey(disp_filtered)
print('----------')


# Number of successful simulations
# n_cases = len(successful_sims)
# n_cases = stress_data.shape[0]

# u                n_case n_step  n_input
Heat_Amp = np.stack([flux_filtered[:n_cases, :],
                    disp_filtered[:n_cases, :]], axis=-1)

# s
data_t = temp_data[:n_cases, -1, :-1]
data_s = stress_data[:n_cases, -1, :-1]

# Scale
scalerT = MinMaxScaler()
# scalerT = PowerTransformer()
# scalerT = MaxAbsScaler()
# scalerT = StandardScaler()
scalerT.fit(data_t)
scaled_temp = scalerT.transform(data_t)
print('Temp survey: ')
Survey(scaled_temp)
print('----------')

scalerS = MinMaxScaler()
# scalerS = PowerTransformer()
# scalerS = MaxAbsScaler()
# scalerS = StandardScaler()
scalerS.fit(data_s)
scaled_stress = scalerS.transform(data_s)
print('Stress survey: ')
Survey(scaled_stress)
print('----------')


Temp = np.zeros((n_cases, N_output_frame, n_nodes, N_component))
Temp[:, -1, :602, 0] = scaled_temp
Temp[:, -1, :602, 1] = scaled_stress
print('Target shape: ', Temp.shape)


for idx, fraction_train in enumerate([0.8]):
    # for idx , fraction_train in enumerate([ 0.5 , 0.6 , 0.7 , 0.8 ]):
    print('fraction_train = ' + str(fraction_train))
    count = 1
    while (1):
        print('--------------------------------------')
        print('Trial number: ' + str(count))
        # Train / test split
        """
                N_valid_case = len(Heat_Amp)
                N_train = int( N_valid_case * fraction_train )
                train_case = np.random.choice( N_valid_case , N_train , replace=False )
                test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )


                u0_train = Heat_Amp[ train_case , :: ]
                u0_testing = Heat_Amp[ test_case , :: ]
                s_train = Temp[ train_case , : ]
                s_testing = Temp[ test_case , : ]
                """

        N_valid_case = len(Heat_Amp)
        N_train = int(N_valid_case * fraction_train)
        print("N_train = ", N_train)

        u0_train = Heat_Amp[:N_train, ::]
        u0_testing = Heat_Amp[N_train:, ::]

        s_train = Temp[:N_train, :]
        s_testing = Temp[N_train:, :]

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

        my_act1 = "tanh"
        branch = tf.keras.models.Sequential([
            tf.keras.layers.GRU(units=256, batch_input_shape=(batch_size, m, N_input_fn),
                                activation=my_act1, return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
            # tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.GRU(units=128, activation=my_act1,
                                return_sequences=False, dropout=0.00, recurrent_dropout=0.00),
            # tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.RepeatVector(HIDDEN),
            tf.keras.layers.GRU(units=128, activation=my_act1,
                                return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
            # tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.GRU(units=256, activation=my_act1,
                                return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
            # tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(N_output_frame))])
        branch.summary()

        my_act2 = "relu"
        trunk = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(101, activation=my_act2,
                                  kernel_initializer='GlorotNormal'),
            tf.keras.layers.Dense(101, activation=my_act2,
                                  kernel_initializer='GlorotNormal'),
            tf.keras.layers.Dense(101, activation=my_act2,
                                  kernel_initializer='GlorotNormal'),
            tf.keras.layers.Dense(101, activation=my_act2,
                                  kernel_initializer='GlorotNormal'),
            tf.keras.layers.Dense(101, activation=my_act2,
                                  kernel_initializer='GlorotNormal'),
            tf.keras.layers.Dense(
                HIDDEN * N_component, activation=my_act2, kernel_initializer='GlorotNormal'),
            tf.keras.layers.Reshape([HIDDEN, N_component]),
        ])
        trunk.summary()

        net = DeepONetCartesianProd(
            [m, branch], [trunk], my_act2, "Glorot normal")

        model = dde.Model(data, net)
        print("y_train shape:", y_train.shape)
        '''
                def Back_to_origin(y):
                        tmp_y = tf.make_ndarray(y)
                        Temp_tmp = scalerT.inverse_transform(tmp_y[:,0,:,0])
                        Stress_tmp = scalerS.inverse_transform(tmp_y[:,0,:,1])
                        new_y = np.append(Temp_tmp, Stress_tmp)
                        return tf.convert_to_tensor(new_y, dtype=float64)
                '''
        def MSE(y_true, y_pred):
            # tmp = tf.math.square( K.flatten(Back_to_origin(y_true)) - K.flatten(Back_to_origin(y_pred)) )
            tmp = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
            print('-----Inside MSE function-----')
            print('y_true type: ', type(y_true))
            print('y_true: ', y_true)
            print('y_pred type: ', type(y_pred))
            print('y_pred: ', y_pred)
            data_loss = tf.math.reduce_mean(tmp)
            return data_loss

        def MAE(y_true, y_pred):
            tmp = tf.math.abs(K.flatten(y_true) - K.flatten(y_pred))
            data_loss = tf.math.reduce_mean(tmp)
            print('-----Inside MAE function-----')
            print(data_loss)
            print('y_true type: ', type(y_true))
            print('y_true shape: ', tf.shape(y_true))
            print('y_pred shape: ', tf.shape(y_pred))
            print('y_true: ', y_true)
            return data_loss

        def COP(y_true, y_pred):
            sqr_err = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
            var_true = y_true.shape[0] * \
                tf.math.reduce_variance(K.flatten(y_true))
            data_loss = tf.math.divide(tf.math.reduce_sum(sqr_err), var_true)
            return data_loss

        def err(y_train, y_pred):
            ax = 1
            return np.linalg.norm(y_train - y_pred, axis=ax) / np.linalg.norm(y_train, axis=ax)

        def err2(y_train, y_pred):
            ax = 1
            abs_diff = np.abs(y_train - y_pred)
            test_minus_mean = np.abs(
                y_train - y_train.mean(axis=ax).reshape(-1, 1))
            rel_abs_err = np.sum(abs_diff, axis=ax) / \
                np.sum(test_minus_mean, axis=ax)
            return rel_abs_err

        def err3(y_train, y_pred):
            ax = 1
            sum_sqr_error = np.sum(np.power(y_train - y_pred, 2), axis=1)
            test_variance = np.sum(
                np.power(y_train - np.mean(y_train, axis=1).reshape(-1, 1), 2), axis=1)
            Cop_sam = 1 - sum_sqr_error / test_variance
            return Cop_sam

        def metric1(y_train, y_pred):
            y_train_original = scalerS.inverse_transform(y_train[:, 0, :, 1])
            y_pred_original = scalerS.inverse_transform(y_pred[:, 0, :, 1])
            return np.mean(err(y_train_original, y_pred_original).flatten())

        def metric2(y_train, y_pred):
            y_train_original = scalerS.inverse_transform(y_train[:, 0, :, 1])
            y_pred_original = scalerS.inverse_transform(y_pred[:, 0, :, 1])
            return np.mean(err2(y_train_original, y_pred_original).flatten())

        def metric3(y_train, y_pred):
            y_train_original = scalerS.inverse_transform(y_train[:, 0, :, 1])
            y_pred_original = scalerS.inverse_transform(y_pred[:, 0, :, 1])
            return np.mean(err3(y_train_original, y_pred_original).flatten())

        def metric_mae_stress(y_train, y_pred):
            y_train_original = scalerS.inverse_transform(y_train[:, 0, :, 1])
            y_pred_original = scalerS.inverse_transform(y_pred[:, 0, :, 1])
            tmp = tf.math.abs(K.flatten(y_train_original) -
                              K.flatten(y_pred_original))
            data_loss = tf.math.reduce_mean(tmp)
            return data_loss

        model.compile(
            "adam",
            lr=1e-3,
            decay=("inverse time", 1, 1e-4),
            loss=COP,
            metrics=[metric1, metric_mae_stress],
        )
        # losshistory, train_state = model.train(iterations=350000, batch_size=batch_size, model_save_path="./mdls/TrainFrac_"+str(idx) )
        losshistory, train_state = model.train(
            iterations=310000, batch_size=batch_size, model_save_path="./Models/model_1br_"+str(count))
        np.save('losshistory_1br_'+str(count)+'.npy', losshistory)

        st = TT.time()
        y_pred = model.predict(data.test_x)
        duration = TT.time() - st
        print('y_pred.shape =', y_pred.shape)
        print('Prediction took ', duration, ' s')
        print('Prediction speed = ', duration / float(len(y_pred)), ' s/case')
        # np.savez_compressed('TestData'+str(idx)+'.npz',a=y_test,b=y_pred,c=u0_testing,d=xy_train_testing)
        # np.savez_compressed('Ver_4_TestData'+str(count)+'.npz',a=y_test,b=y_pred,c=u0_testing,d=xy_train_testing, e=train_case, f=test_case, g=losshistory, h=train_state)
        Org_temp_test = scalerT.inverse_transform(y_test[:, 0, :, 0])
        Org_stress_test = scalerS.inverse_transform(y_test[:, 0, :, 1])
        Org_temp_pred = scalerT.inverse_transform(y_pred[:, 0, :, 0])
        Org_stress_pred = scalerS.inverse_transform(y_pred[:, 0, :, 1])
        print('Successful to convert to original forms')
        print()
        np.savez_compressed('model_s-don_'+str(count)+'.npz', a=Org_temp_test,
                            b=Org_stress_test, c=Org_temp_pred, d=Org_stress_pred)
        print('Saving all successful')

        error_s = np.linalg.norm(Org_stress_test - Org_stress_pred,
                                 axis=1) / (np.linalg.norm(Org_stress_test, axis=1) + 1e-8)

        print(
            f"Mean L2 error (Stress) for test data: {np.mean(error_s)}, std: {np.std(error_s)}, max: {np.max(error_s)}")

        error_t = np.linalg.norm(Org_temp_test - Org_temp_pred,
                                 axis=1) / (np.linalg.norm(Org_temp_test, axis=1) + 1e-8)
        print(
            f"Mean L2 error (Temperature) for test data: {np.mean(error_t)}, std: {np.std(error_t)}, max: {np.max(error_t)}")

        # error_s = []
        # error_t = []
        # org_error_s = []
        # org_error_t = []
        '''
                for i in range(len(y_pred)):
                        error_t_tmp = np.linalg.norm(y_test[i, 0, :, 0] - y_pred[i, 0, :, 0]) / np.linalg.norm(y_test[i, 0, :, 0])
                        #error_t_tmp = err(y_test[:, 0, :, 0], y_pred[:, 0, :, 0])
                        error_s_tmp = np.linalg.norm(y_test[i, 0, :, 1] - y_pred[i, 0, :, 1]) / np.linalg.norm(y_test[i, 0, :, 1])
                        #error_s_tmp = err(y_test[i, 0, :, 1], y_pred[i, 0, :, 1])
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
                '''

        error_t = err(y_test[:, 0, :, 0], y_pred[:, 0, :, 0])
        error_s = err(y_test[:, 0, :, 1], y_pred[:, 0, :, 1])
        org_error_t = err(Org_temp_test, Org_temp_pred)
        org_error_s = err(Org_stress_test, Org_stress_pred)

        rae_org_error_t = err2(Org_temp_test, Org_temp_pred)
        rae_org_error_s = err2(Org_stress_test, Org_stress_pred)

        # error_s = np.stack(error_s)
        # error_t = np.stack(error_t)
        # org_error_s = np.stack(org_error_s)
        # org_error_t = np.stack(org_error_t)
        print('$$$$$$$$$$$$$$$$$$$$$Trial num: ' +
              str(count) + '$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print()
        print("error_t = ", error_t)
        print()
        print('----------------------------------------')
        print()
        print("error_s = ", error_s)
        print('----------------------------------------')
        print()
        print()
        print('Stress_scaling_factor: ', stress_scaling_factor)
        print('Trunk_scaling_factor: ', trunk_scaling_factor)
        print()
        # Calculate mean and std for all testing data samples
        print('Scaled L2 error')
        print('mean of temperature relative L2 error of s: {:.2e}'.format(
            error_t.mean()))
        print('std of temperature relative L2 error of s: {:.2e}'.format(
            error_t.std()))
        print('--------------------------------------------------------------')
        print('mean of stress relative L2 error of s: {:.2e}'.format(
            error_s.mean()))
        print('std of stress relative L2 error of s: {:.2e}'.format(
            error_s.std()))
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print()
        print('Origianl L2 error')
        print('mean of Original temperature relative L2 error of t: {:.2e}'.format(
            org_error_t.mean()))
        print('std of Original temperature relative L2 error of t: {:.2e}'.format(
            org_error_t.std()))
        print('--------------------------------------------------------------')
        print('mean of Orginal stress relative L2 error of s: {:.2e}'.format(
            org_error_s.mean()))
        print('std of Orginal stress relative L2 error of s: {:.2e}'.format(
            org_error_s.std()))
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print()
        print('Origianl RAE')
        print('mean of Original temperature relative RAE error of t: {:.2e}'.format(
            rae_org_error_t.mean()))
        print('std of Orginal temperature relative RAE error of t: {:.2e}'.format(
            rae_org_error_t.std()))
        print('--------------------------------------------------------------')
        print('mean of Original stress relative RAE error of s: {:.2e}'.format(
            rae_org_error_s.mean()))
        print('std of Orginal stress relative RAE error of s: {:.2e}'.format(
            rae_org_error_s.std()))
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print()
        print()
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print()
        print()
        print()
        if count > 1 or org_error_s.mean() < 0.10:
            break

        count = count + 1

    # plt.hist( error_s.flatten() , bins=25 )
    # plt.savefig('Stress_Err_hist_1br'+str(idx)+'.jpg' , dpi=300)

    # plt.hist( error_t.flatten() , bins=25 )
    # plt.savefig('Temp_Err_hist_1br'+str(idx)+'.jpg' , dpi=300)
