import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf

from model.architecture import LeNetLike, customized_loss
from model.configurations import *
import argparse
import math
import keras
import nni
from sklearn.metrics import roc_auc_score, f1_score

TRAIN_SEED = 1
tf.random.set_seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED) # for reproducibility

def parse_tfrecord_fn(example):
    """Parses a single tf.train.Example back into the required input format."""
    feature_description = {
        'ext': tf.io.FixedLenFeature([WINDOW_SIZE * N_CHANNELS], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    
    X_ext = tf.reshape(parsed_example['ext'], (WINDOW_SIZE, N_CHANNELS))
    y = parsed_example['label']
    y = tf.one_hot(y[0], depth=2)
    return X_ext, y

def load_tfrecord_dataset(tfrecord_file, batch_size, shuffle=True, cache_in_memory=False, cache_file=None):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="GZIP")
    
    # Parse the dataset
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if cache_in_memory: # Cache in memory
        parsed_dataset = parsed_dataset.cache()  
    elif cache_file: # Cache to a file on disk
        parsed_dataset = parsed_dataset.cache(cache_file)  
    
    #dataset_size = sum(1 for _ in parsed_dataset) if parsed_dataset.cardinality() <= 0 else parsed_dataset.cardinality().numpy()

    dataset_size = -2
    
    # Shuffle, batch, and prefetch for performance
    if shuffle:
        dataset = (parsed_dataset
                .shuffle(buffer_size=100000, seed=TRAIN_SEED, reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    else:
        dataset = (parsed_dataset
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))
    return dataset, dataset_size


def train_model(params, train_path, val_path, checkpoint_path, using_nni=False):
    if using_nni:
        learning_rate = params['learning_rate'] 
        batch_size = params['batch_size']
        num_epochs = params['num_epochs']
        conv_activation = params['conv_activation']
        dense_activation = params['dense_activation']
    else:
        learning_rate =  0.003
        batch_size =  1024
        num_epochs =  10
        conv_activation =  "tanh"
        dense_activation = None
    
    model = LeNetLike(kernel_size=K,
                      filters=N,
                      pool_size=M,
                      conv_activation=conv_activation,
                      dense_activation=dense_activation,
                      dropout_rate=P)

    # Print model configuration
    
    if not using_nni:
        print(model.get_config())

    # Compile model
    model.compile(
        loss=LOSS_FUNCTION, # customized_loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[keras.metrics.CategoricalAccuracy()], # keras.metrics.BinaryAccuracy()], # [customized_acc],
        run_eagerly=True
    )

    if not using_nni:
        print(model.summary())

    # Load data
    train_dataset, train_size = load_tfrecord_dataset(train_path, batch_size=batch_size,
                                                      shuffle=True, cache_in_memory=False, cache_file=None)
    if not using_nni:
        print(f"Training size: {train_size}")
    
    val_dataset, val_size = load_tfrecord_dataset(val_path, batch_size=batch_size, 
                                                  shuffle=False, cache_in_memory=False, cache_file=None)
    if not using_nni:
        print(f"Validation size: {val_size}")
    
    selected_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss')]
    if not using_nni:
        # Setup checkpoint callback
        callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        
        last_cp_path = checkpoint_path.replace(".keras", "_last.keras")
        last_callback = keras.callbacks.ModelCheckpoint(
            filepath=last_cp_path,
            verbose=0,
        )
        
        selected_callbacks += [callback, last_callback]
    
    # Train model
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=selected_callbacks,
        verbose=not using_nni
    )
    
    y_true = []
    y_pred = []

    for X, y in val_dataset:
        y_true.append(y.numpy())
        y_pred.append(model.predict(X, verbose=False))  # Take the probability for class 1

    # Calculate ROCAUC
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    # Report the ROCAUC back to NNI
    nni.report_final_result(roc_auc)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LeNet-like model for Sleep apnea detection using Sp02 and PR')

    # Add arguments
    parser.add_argument('-train', '--train_file',  type=str, help='Training data path (should endswith *.tfrecord)')
    parser.add_argument('-val', '--val_file',  type=str, help='Validation data path (should endswith *.tfrecord)')
    parser.add_argument('-cp', '--checkpoint', type=str, help='Checkpoint path (*.keras)')

    # Parse the arguments
    args = parser.parse_args()

    using_nni = args.checkpoint == 'nni.keras'
    
    if not using_nni and os.path.exists(args.checkpoint):
        ans = input(f"\n-----------The checkpoint already existed, do you want to override it? [Y / other:exit program] ")
        if ans.lower() != 'y':
            sys.exit(1)
    
    params = nni.get_next_parameter()
    
    train_model(params, args.train_file, args.val_file, args.checkpoint, using_nni)
    if not using_nni:
        print("-----------Completed-----------")
