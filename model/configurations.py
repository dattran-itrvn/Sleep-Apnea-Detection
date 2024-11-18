import keras
# Training
EPOCHS = 50
BATCH_SIZE = 64

WINDOW_SIZE = 150
N_CHANNELS = 2

LOSS_FUNCTION = keras.losses.CategoricalCrossentropy(from_logits=False)

OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)

ADD_CALLBACKS = [keras.callbacks.EarlyStopping(monitor='val_loss')]

# hyper params (paper)
K = 5
N = 8
P = 0.2
M = 3
