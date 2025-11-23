import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D, MaxPooling3D, Dense, BatchNormalization, GlobalAveragePooling3D
)
from tensorflow.keras import Model
import argparse
import os

# Import generator from local directory
try:
    from . import generator
except ImportError:
    import generator

class CNN3DModel(Model):
    """
    3D CNN for Volumetric Chemical Data.
    """
    def __init__(self, num_output_classes=2):
        super().__init__()
        
        # Block 1
        self.conv1 = Conv3D(32, (3, 3, 3), padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling3D(pool_size=(3, 3, 3))
        
        # Block 2
        self.conv2 = Conv3D(64, (3, 3, 3), padding='same', activation='relu')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling3D(pool_size=(2, 2, 2))
        
        # Global Pooling (Replaces Flatten for memory efficiency)
        self.global_pool = GlobalAveragePooling3D()
        
        # Dense Layers
        self.dense1 = Dense(128, activation='relu')
        self.classifier = Dense(num_output_classes) # Logits (no activation)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.global_pool(x)
        x = self.dense1(x)
        return self.classifier(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .h5 data file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.")
        exit(1)

    # 1. Load Data
    print(f"Loading data from {args.data}...")
    dataset = generator.create_tf_dataset(args.data, b_size=args.batch)
    
    # 2. Init Model
    model = CNN3DModel(num_output_classes=2)
    
    # 3. Compile
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )

    # 4. Train
    model.fit(dataset, epochs=args.epochs)
