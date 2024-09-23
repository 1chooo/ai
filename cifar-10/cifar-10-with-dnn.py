import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from random import randrange

class CIFAR10Trainer:
    # Constants
    TITLE = "Train History"
    LEGEND_LOCATION = "upper left"
    ACCURACY_MESSAGE = 'Test accuracy:'
    
    def __init__(self):
        # Load CIFAR-10 dataset
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = keras.datasets.cifar10.load_data()
        # Normalize the pixel values
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

        # Class labels for CIFAR-10 dataset
        self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    def display_samples(self):
        """Displays random images from the CIFAR-10 dataset with their labels."""
        plt.figure(figsize=(16, 10), facecolor='w')
        for i in range(5):
            for j in range(8):
                index = randrange(0, 50000)
                plt.subplot(5, 8, i * 8 + j + 1)
                plt.title("label: {}".format(self.labels[self.train_labels[index][0]]))
                plt.imshow(self.train_images[index])
                plt.axis('off')
        plt.show()

    def build_model(self, optimizer='adam'):
        """Builds and compiles the model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model

    def train_and_evaluate(self, epochs, batch_size=32, optimizer='adam'):
        """Trains and evaluates the model."""
        model = self.build_model(optimizer)
        
        history = model.fit(
            self.train_images, 
            self.train_labels, 
            epochs=epochs, 
            validation_data=(self.test_images, self.test_labels),
            batch_size=batch_size
        )
        
        # Evaluate the model
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)
        print(self.ACCURACY_MESSAGE, test_acc)
        
        self.plot_history(history)

    def plot_history(self, history):
        """Plots the loss history for training and validation."""
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title(self.TITLE)
        plt.ylabel("loss")
        plt.xlabel("Epoch")
        plt.legend(["loss", "val_loss"], loc=self.LEGEND_LOCATION)
        plt.show()

def main():
    # Instantiate the CIFAR10Trainer class
    trainer = CIFAR10Trainer()

    # Display random samples from the dataset
    trainer.display_samples()

    # Run the 7 experiments
    trainer.train_and_evaluate(epochs=20)    # First experiment
    trainer.train_and_evaluate(epochs=14)    # Second experiment
    trainer.train_and_evaluate(epochs=7)     # Third experiment
    trainer.train_and_evaluate(epochs=14, batch_size=100)  # Fourth experiment
    trainer.train_and_evaluate(epochs=9, batch_size=100)   # Fifth experiment
    trainer.train_and_evaluate(epochs=9, batch_size=500)   # Sixth experiment
    trainer.train_and_evaluate(epochs=14, batch_size=500, optimizer='rmsprop')  # Seventh experiment

if __name__ == "__main__":
    main()
