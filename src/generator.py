import tensorflow as tf
import h5py
import numpy as np
import os
import random

def collect_example_paths(filepath: str):
    """Scans HDF5 and returns paths to specific examples."""
    example_paths = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")

    with h5py.File(filepath, 'r') as f:
        for class_name in f.keys():
            class_group = f[class_name]
            for example_name in class_group.keys():
                example_paths.append(class_group[example_name].name)
    return example_paths

def _data_generator(filepath: str, example_paths: list):
    """Yields (Image, Label) tuples from the HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        for path in example_paths:
            if path not in f: continue
            
            example_group = f[path]
            
            # Logic: Determine label from path (Custom logic: adapt as needed)
            # Example: if parent folder is 'super' -> 1, else -> 0
            path_parts = path.split(os.path.sep)
            class_label = 1 if 'super' in path_parts else 0

            # Iterate over frames (Currently restricted to frame_0001)
            # To train on all frames, iterate over example_group.keys() filtering for 'frame_'
            if 'frame_0001' in example_group:
                frame_group = example_group['frame_0001']
                try:
                    homo = frame_group['homo_cube'][:]
                    
                    # Normalize and Add Channel Dim: (100,100,100) -> (100,100,100,1)
                    img = ((homo.astype(np.float32) + 0.25) / 0.5)[..., np.newaxis]
                    yield (img, np.int32(class_label))
                except KeyError:
                    continue

def create_tf_dataset(filepath: str, b_size=10, shape=(100, 100, 100, 1)):
    """Creates a shuffled, batched tf.data.Dataset."""
    all_paths = collect_example_paths(filepath)
    random.shuffle(all_paths)
    
    output_signature = (
        tf.TensorSpec(shape=shape, dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: _data_generator(filepath, all_paths),
        output_signature=output_signature
    )
    
    dataset = dataset.batch(b_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
