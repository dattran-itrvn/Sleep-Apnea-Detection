import keras
# Training

WINDOW_SIZE = 150
N_CHANNELS = 2

# LOSS_FUNCTION = keras.losses.CategoricalCrossentropy(from_logits=False) # paper => not good
LOSS_FUNCTION = keras.losses.BinaryCrossentropy(from_logits=False)

# hyper params (paper)
K = 5
N = 8
P = 0.2
M = 3
