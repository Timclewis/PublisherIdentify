import tensorflow as tf
import os
import gc
from tensorflow.keras.applications import ResNet50V2, Xception, EfficientNetB3, EfficientNetB4, EfficientNetB5
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from DataGenerator import DataGenerator
from BuildModel import BuildModel
from Logger import Logger
from RunModel import run_model


# import Predictor

def main():
    print(f'Tensorflow version {tf.version.VERSION}')
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    dat1 = DataGenerator(val_split=0.15, height=300, width=200, batch_size=2, steps=10)
    dat1.generator_info()

    mod1 = BuildModel(data_generator=dat1)
    mod1.compile_model(epochs=5, network=Xception, pooling='avg', optimizer=Adam, learn_rate=1E-4, summary=False)
    #log1 = Logger(build_model=mod1, sprite_height=100, sprite_width=70, data_points=500)
    run_model(build_model=mod1, logger=log1, metrics=True, confusion_matrix=False, image_visual=True, projector=False)

    # pred1 = Predictor.Predictor(mod1)
    # pred1.img_predict()
    # pred1.batch_predict(1000)
    # pred1.batch_evaluate(49)


if __name__ == '__main__':
    main()
    tf.keras.backend.clear_session()
    del tf
    gc.collect()
