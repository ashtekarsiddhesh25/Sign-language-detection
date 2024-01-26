from function import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# Assuming frame_shape is the desired shape for each frame
frame_shape = (63,)  # Adjust this based on your actual frame shape

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)),
                allow_pickle=True
            )
            if res is not None:
                print(f"Content of loaded array: {res}")
                window.append(res)
        
        # Check if the window is not empty and contains valid elements before appending
        if window and all(isinstance(elem, np.ndarray) and elem.ndim == 1 for elem in window):
            sequences.append(window)
            labels.append(label_map[action])

# Filter out empty sequences
non_empty_sequences = [seq for seq in sequences if seq]

# Pad each frame individually
padded_frames = [pad_sequences(seq, dtype='float32', padding='post', truncating='post', maxlen=frame_shape[0]) for seq in non_empty_sequences]

# Combine padded frames into sequences
padded_sequences = np.array(padded_frames)

X = padded_sequences
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')

