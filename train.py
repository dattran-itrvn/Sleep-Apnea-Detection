import numpy as np
import sys
import tensorflow as tf
from model.configurations import *
from model.architecture import LeNetLike, customized_loss
import os
import argparse
import math
import keras

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

def main(train_path, val_path, checkpoint_path):
    model = LeNetLike(kernel_size=K,
                      filters=N,
                      pool_size=M,
                      dropout_rate=P)

    # Print model configuration
    print(model.get_config())

    # Compile model
    model.compile(
        loss=LOSS_FUNCTION, # customized_loss
        optimizer=OPTIMIZER,
        metrics=[keras.metrics.BinaryAccuracy()], # keras.metrics.BinaryAccuracy()], # [customized_acc],
        run_eagerly=True
    )

    print(model.summary())

    # Load data
    train_dataset, train_size = load_tfrecord_dataset(train_path, batch_size=BATCH_SIZE,
                                                      shuffle=True, cache_in_memory=False, cache_file=None)
    print(f"Training size: {train_size}")
    
    val_dataset, val_size = load_tfrecord_dataset(val_path, batch_size=BATCH_SIZE, 
                                                  shuffle=False, cache_in_memory=False, cache_file=None)
    print(f"Validation size: {val_size}")
    
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
    
    # Train model
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[callback, last_callback] + ADD_CALLBACKS
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LeNet-like model for Sleep apnea detection using Sp02 and PR')

    # Add arguments
    parser.add_argument('-train', '--train_file',  type=str, help='Training data path (should endswith *.tfrecord)')
    parser.add_argument('-val', '--val_file',  type=str, help='Validation data path (should endswith *.tfrecord)')
    parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint path (*.keras)')

    # Parse the arguments
    args = parser.parse_args()

    if os.path.exists(args.checkpoint):
        ans = input(f"\n-----------The checkpoint already existed, do you want to override it? [Y / other:exit program] ")
        if ans.lower() != 'y':
            sys.exit(1)
    
    main(args.train_file, args.val_file, args.checkpoint)
    print("-----------Completed-----------")
