import datetime
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LambdaCallback, TensorBoard
from tensorflow.keras.models import Model, load_model

from DataGenerator import DataGenerator


class BuildModel:
    def __init__(self, data_generator):
        self.width = data_generator.width
        self.height = data_generator.height
        self.class_num = data_generator.class_num
        self.class_names = data_generator.class_names
        self.steps = data_generator.steps
        self.val_steps = data_generator.val_steps
        self.batch_size = data_generator.batch_size
        self.train_data = data_generator.train_data
        self.valid_data = data_generator.valid_data
        self.test_data = data_generator.test_data
        self.test_labels = data_generator.test_labels
        # self.publisher_names = list(self.train_data.class_indices.keys())[0:self.class_num]

    def compile_model(self, epochs, network, pooling, optimizer, learn_rate, summary):
        self.epochs = epochs
        self.network = network
        self.lr = learn_rate
        self.opt = optimizer
        self.pool = pooling
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

        base = self.network(include_top=False, weights='imagenet',
                            input_shape=(self.width, self.height, 3), pooling=self.pool)
        opt = self.opt(learning_rate=self.lr)
        x = base.output
        # x = layers.Dense(512, activation= 'relu')(x)
        x = layers.Dense(self.class_num, activation='softmax')(x)
        self.model = Model(base.input, x)
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)
        # self.embeddings = Model(inputs=self.model.inputs,
        #                        outputs=self.model.layers[-2].output)

        if summary:
            self.model.summary()
