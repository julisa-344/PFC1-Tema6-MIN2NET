import numpy as np
import csv
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.model_selection import KFold, train_test_split
import wget
import os
import time
import tensorflow as tf
from sklearn.utils import class_weight
from scipy.interpolate import CubicSpline 
from sklearn.utils import resample
from scipy import ndimage
import argparse

# lib path
PATH = os.path.dirname(os.path.realpath(__file__))

# def load_raw(dataset):
#     # folder_name = str(PATH)+'/datasets'
#     folder_name = 'datasets'
#     if dataset == 'BCIC2a':
#         try:
#             num_subjects = 9
#             sessions = ['T', 'E']
#             save_path = folder_name + '/' + dataset + '/raw'
#             if save_path is not None:
#                 if not os.path.exists(save_path):
#                     os.makedirs(save_path)

#             for session in sessions:
#                 for person in range(1, num_subjects+1):
#                     file_name = '/A{:02d}{}.mat'.format(person, session)
#                     if os.path.exists(save_path+file_name):
#                         os.remove(save_path+file_name) # if exist, remove file
#                     print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
#                     url = 'https://lampx.tugraz.at/~bci/database/001-2014'+file_name
#                     print('save to: '+save_path+file_name)
#                     wget.download(url, save_path+file_name)
#             print('\nDone!')
#         except:
#             raise Exception('Path Error: file does not exist, please direccly download at http://bnci-horizon-2020.eu/database/data-sets')
#     elif dataset == 'SMR_BCI':
#         try:
#             num_subjects = 14
#             sessions = ['T', 'E']
#             save_path = folder_name + '/' + dataset + '/raw'
#             if save_path is not None:
#                 if not os.path.exists(save_path):
#                     os.makedirs(save_path)
#             for session in sessions:
#                 for person in range(1, num_subjects+1):
#                     file_name = '/S{:02d}{}.mat'.format(person, session)
#                     if os.path.exists(save_path+file_name):
#                         os.remove(save_path+file_name) # if exist, remove file
#                     print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
#                     url = 'https://lampx.tugraz.at/~bci/database/002-2014'+file_name
#                     print('save to: '+save_path+file_name)
#                     wget.download(url,  save_path+file_name)
#             print('\nDone!')
#         except:
#             raise Exception('Path Error: file does not exist, please direccly download at http://bnci-horizon-2020.eu/database/data-sets')

class DataLoader:
    def __init__(self, data_path, num_class=2, subject=None, data_format='NCTD', **kwargs):
        self.data_path = data_path
        self.subject = subject # id, start at 1
        self.data_format = data_format # 'channels_first', 'channels_last'
        self.fold = None # fold, start at 1
        self.prefix_name = 'A'
        self.num_class = num_class
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])


        self.path = self.data_path + "partitioned"

        os.makedirs(self.path, exist_ok=True)
        num_folds = kwargs.get('num_folds', 5)
        self.prepare_dataset(num_folds=num_folds, ratio=0.6)
    
    def _change_data_format(self, X):
        if self.data_format == 'NCTD':
            # (#n_trial, #channels, #time, #depth) ***
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
            print("entegando X con shape: ", X.shape)
        elif self.data_format == 'NDCT':
            # (#n_trial, #depth, #channels, #time)
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        elif self.data_format == 'NTCD':
            # (#n_trial, #time, #channels, #depth)
            X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])
            # X = np.swapaxes(X, 1, 3)
        elif self.data_format == 'NSHWD':
            # (#n_trial, #Freqs, #height, #width, #depth)
            X = zero_padding(X)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
        elif self.data_format == None:
            pass
        else:
            raise Exception('Value Error: data_format requires None, \'NCTD\', \'NDCT\', \'NTCD\' or \'NSHWD\', found data_format={}'.format(self.data_format))
        print('change data_format to \'{}\', new dimention is {}'.format(self.data_format, X.shape))
        return X

    def load_train_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
    
        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'X_train_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'y_train_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x))
            y = np.load(self.file_y)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y

    def load_val_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'X_val_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'y_val_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x))
            y = np.load(self.file_y)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y
    
    def load_test_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'X_test_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'y_test_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x))
            y = np.load(self.file_y)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y

    def prepare_dataset(self, num_folds, ratio=0.6):
        # Load data
        class1_data = np.load(self.path+"/class_1/{}{:02d}.npy".format(self.prefix_name, self.subject)).swapaxes(0, 1)
        class2_data = np.load(self.path+"/class_2/{}{:02d}.npy".format(self.prefix_name, self.subject)).swapaxes(0, 1)

        # Balance the dataset by oversampling the minority class (class2_data in this case)
        if class1_data.shape[0] > class2_data.shape[0]:
            class2_data = resample(class2_data, replace=True, n_samples=class1_data.shape[0], random_state=123)
        elif class1_data.shape[0] < class2_data.shape[0]:
            class1_data = resample(class1_data, replace=True, n_samples=class2_data.shape[0], random_state=123)

        print(f"Class 1 shape: {class1_data.shape}")
        print(f"Class 2 shape: {class2_data.shape}")

        data = np.concatenate((class1_data, class2_data), axis=0)
        ## class1 (NOP300) = 0.0   y class2 (P300) = 1.0
        labels = np.concatenate((np.zeros(class1_data.shape[0]), np.ones(class2_data.shape[0])), axis=0)

        # Split data into train and test 60 porciento de la data total va para el train
        # Donde ratio = 0.6 (60% de la data va para el rain)
        X_train_test, X_test, y_train_test, y_test = train_test_split(data, labels, test_size=ratio)

        # Split train_test into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_test, y_train_test, test_size=ratio)

        # Initialize KFold
        kf = KFold(n_splits=num_folds)
        self.path = self.data_path + "processed/"

        # For each fold, save train, validation, and test sets
        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            print(f"saving fold {fold}")
            np.save(self.path + f'X_train_{self.prefix_name}{self.subject:03d}_fold{fold:03d}.npy', X_train[train_index])
            np.save(self.path + f'X_val_{self.prefix_name}{self.subject:03d}_fold{fold:03d}.npy', X_train[val_index])
            np.save(self.path + f'X_test_{self.prefix_name}{self.subject:03d}_fold{fold:03d}.npy', X_test)
            np.save(self.path + f'y_train_{self.prefix_name}{self.subject:03d}_fold{fold:03d}.npy', y_train[train_index])
            np.save(self.path + f'y_val_{self.prefix_name}{self.subject:03d}_fold{fold:03d}.npy', y_train[val_index])
            np.save(self.path + f'y_test_{self.prefix_name}{self.subject:03d}_fold{fold:03d}.npy', y_test)


def compute_class_weight(y_train):
    """compute class balancing

    Args:
        y_train (list, ndarray): [description]

    Returns:
        (dict): class weight balancing
    """
    return dict(zip(np.unique(y_train), 
                    class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train))) 
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, save_path=None):
        self.save_path = save_path
    def on_train_begin(self, logs={}):
        self.logs = []
        if self.save_path:
            write_log(filepath=self.save_path, data=['time_log'], mode='w')
    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs={}):
        time_diff = time.time()-self.start_time
        self.logs.append(time_diff)
        if self.save_path:
            write_log(filepath=self.save_path, data=[time_diff], mode='a')

def write_log(filepath='test.log', data=[], mode='w'):
    '''
    filepath: path to save
    data: list of data
    mode: a = update data to file, w = write a new file
    '''
    try:
        with open(filepath, mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)
    except IOError:
        raise Exception('I/O error')

def zero_padding(data, pad_size=4):
    if len(data.shape) != 4:
        raise Exception('Dimension is not match!, must have 4 dims')
    new_shape = int(data.shape[2]+(2*pad_size))
    data_pad = np.zeros((data.shape[0], data.shape[1], new_shape, new_shape))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_pad[i,j,:,:] = np.pad(data[i,j,:,:], [pad_size, pad_size], mode='constant')
    return data_pad 


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def resampling(data, new_smp_freq, data_len):
    if len(data.shape) != 3:
        raise Exception('Dimesion error', "--> please use three-dimensional input")
    new_smp_point = int(data_len*new_smp_freq)
    data_resampled = np.zeros((data.shape[0], data.shape[1], new_smp_point))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i,j,:] = signal.resample(data[i,j,:], new_smp_point)
    return data_resampled

def psd_welch(data, smp_freq):
    if len(data.shape) != 3:
        raise Exception("Dimension Error, must have 3 dimension")
    n_samples,n_chs,n_points = data.shape
    data_psd = np.zeros((n_samples,n_chs,89))
    for i in range(n_samples):
        for j in range(n_chs):
            freq, power_den = signal.welch(data[i,j], smp_freq, nperseg=n_points)
            index = np.where((freq>=8) & (freq<=30))[0].tolist()
            data_psd[i,j] = power_den[index]
    return data_psd