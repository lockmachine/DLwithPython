from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

conv_base = VGG16(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(150, 150, 3)
                )
                
#conv_base.summary()

# smallデータセットを格納したディレクトリパスを設定
base_dir = '/home/makoto/work/shake/DLwithPython/chapter5/dataset_small'

train_dir = os.path.join(base_dir, 'train')
validaton_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
                                            directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary'
                                            )
    i = 0
    for inputs_batch, labels in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i+1) * batch_size] = features_batch
        labels[i * batch_size : (i+1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # ジェネレータはデータを無限ループで生成するため、画像を一通り処理したらbreakしなければならない
            break
    return features, labels
    