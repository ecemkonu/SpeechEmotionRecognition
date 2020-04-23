import librosa
import librosa.display
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from cnn1dLSTM import Conv1DLSTM
from cnn2dLSTM import Conv2DLSTM
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
import datetime
import tensorflow as tf 


print(tf.config.list_physical_devices('GPU'))
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()


NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_EXAMPLES = 2820


modality_dict = {'01': 'full-AV', '02': 'video-only', '03': 'audio-only'}
vocal_channel_dict = {'01': 'speech', '02': 'song'}
emotion_dict = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
emotion_intensity_dict = {'01': 'normal', '02': 'strong'}
statement_dict = {'01': "Kids are talking by the door",
                  '02': 'Dogs are sitting by the door'}
repetition_dict = {'01': '1st', '02': '2nd'}
Actor_dict = {1: 'Male', 0: 'Female'}
"""
def read_data():
    data_path = os.path.join(os.getcwd(), 'data')
    speech_path = os.path.join(data_path, 'speech')
    song_path = os.path.join(data_path, 'song')

    x_data, x_meldata, y_emotions, y_intensity = [], [], [], []

    for directories in (speech_path, song_path):
        for actor_dir in os.listdir(directories):
            load_path = os.path.join(speech_path, actor_dir)
            for data in os.listdir(load_path):
                emotion_type = data[6:8]
                emotion_strength = data[9:11]
                y_emotions.append( emotion_dict[emotion_type] )
                y_intensity.append(emotion_intensity_dict[emotion_strength])
                signal, sr = librosa.load(os.path.join(load_path, data))
                signal, _ = librosa.effects.trim(signal)
                spect = librosa.feature.melspectrogram(y=signal, sr=sr)
                x_data.append(signal)
                x_meldata.append(spect)
    return x_data, x_meldata, y_emotions, y_intensity


x_data, x_meldata, y_emotion, y_intensity = read_data()

pad_1 = len(max(x_data, key=len))
x_data_np = np.zeros((len(x_data), pad_1), dtype=np.float)
print(x_data_np.shape)

for i, j in enumerate(x_data):
    x_data_np[i][:len(j)] = j

max_data = np.amax(x_data_np)
min_data = np.amin(x_data_np)
x_data_np = (x_data_np - min_data) / (max_data - min_data)

pad_2 = 0

for i in range(len(x_meldata)):
    pad_temp = len(max(x_meldata[i], key=len))
    if pad_2 < pad_temp:
        pad_2 = pad_temp

print(x_data_np.shape)

np.savetxt('signal_data.txt', x_data_np) ##save before reshaping to keep 2d

np.savetxt('emotion_labels.txt', y_emotion,  fmt="%s")
np.savetxt('intensity_labels.txt', y_intensity, fmt= '%s')

x_mel_np = np.zeros((len(x_data), 128, pad_2), dtype=np.float)

print(x_mel_np.shape)
for i in range(len(x_meldata)):
    for j, k in enumerate(x_meldata[i]):
        x_mel_np[i][j][:len(k)] = k



xmel_save = x_mel_np.reshape(NUM_EXAMPLES, -1) ###reduce to 2 dimension

np.savetxt('spect_data.txt', xmel_save)
"""

y_emotion = np.genfromtxt("emotion_labels.txt",  dtype='str')
#y_intensity = np.genfromtxt("intensity_labels.txt", dtype = 'str')

x_data_np = np.loadtxt("signal_data.txt")

x_data_np = x_data_np.reshape((-1, x_data_np.shape[1], 1))

x_mel_np = np.loadtxt("spect_data.txt")
x_mel_np = np.reshape(x_mel_np, (NUM_EXAMPLES, 128, -1))
pad_2 = x_mel_np.shape[2]

x_mel_np = np.reshape(x_mel_np, (NUM_EXAMPLES, 128, pad_2, 1))

le = preprocessing.LabelEncoder()
y_labels = le.fit_transform(y_emotion)

###pad 1 pad 2 and n_mels define the dimensions! 

con1dModel = Conv1DLSTM((110615, 1), len(le.classes_))

y_labels = tf.keras.utils.to_categorical(y_labels)

#opt = optimizers.SGD(lr = .0001, momentum = 0.9, decay = 1e-6, nesterov= True)
opt = optimizers.Adam()
opt_Conv1d = sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
con1dModel.compile(loss="categorical_crossentropy",
                   optimizer=opt_Conv1d, metrics=['accuracy'])


x_signalTr,x_signalTe, y_train_sig, y_test_sig = train_test_split(x_data_np, y_labels, test_size = 0.2)
x_spectTr,x_spectTe, y_train_spe, y_test_spe = train_test_split(x_mel_np, y_labels, test_size = 0.2)


tboard_log_dir = os.path.join("logs",'timeSignal')
tboard_log_dir = os.path.join(tboard_log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tb_callbacktime = TensorBoard(log_dir = tboard_log_dir, histogram_freq = 5)
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=15, mode='min', verbose=1)
checkpoint = ModelCheckpoint('model_best_weights_1dcnn.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

con1dModel.fit(x = x_signalTr, y = y_train_sig, epochs = 2* NUM_EPOCHS, batch_size = BATCH_SIZE, validation_data = (x_signalTe, y_test_sig), callbacks = [tb_callbacktime, early_stop, checkpoint] )
"""
tboard_log_dir = os.path.join("logs",'freq')
tboard_log_dir = os.path.join(tboard_log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

log_dir_2dconv = "logs/freqSignal/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = ModelCheckpoint('model_best_weights_2dcnn.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

tb_callbackfreq = TensorBoard(log_dir = tboard_log_dir, histogram_freq = 5)
con2dModel = Conv2DLSTM((128, 217, 1), len(le.classes_))
con2dModel.compile(loss="categorical_crossentropy",
                   optimizer=opt, metrics=['accuracy'])

con2dModel.fit(x = x_spectTr, y = y_train_spe, epochs = 3 * NUM_EPOCHS,batch_size = BATCH_SIZE, validation_data = (x_spectTe, y_test_spe), callbacks = [tb_callbackfreq, early_stop, checkpoint ] )

"""