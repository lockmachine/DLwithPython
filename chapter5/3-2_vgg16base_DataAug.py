import os
import numpy as np
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"

base_dir = '/home/makoto/work/shake/DLwithPython/chapter5/dataset_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

batch_size = 20

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()

# 訓練用データジェネレータインスタンスの生成
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 検証用データジェネレータインスタンスの生成
# 検証データは水増しすべきではないことに注意
test_datagen = ImageDataGenerator(rescale=1./255,)

# 訓練用データジェネレータ
train_generator = train_datagen.flow_from_directory(
    train_dir,              # ターゲットディレクトリ
    target_size=(150, 150), # すべての画像を150x150に変更
    batch_size=batch_size,  # バッチサイズ
    class_mode='binary'     # 損失関数としてbinary_crossentropyを使用するため、二値のラベルが必要
)

# 検証用データジェネレータ
validation_generator = test_datagen.flow_from_directory(
    validation_dir,         # ターゲットディレクトリ
    target_size=(150, 150), # すべての画像を150x150に変更
    batch_size=batch_size,  # バッチサイズ
    class_mode='binary'     # 損失関数としてbinary_crossentropyを使用するため、二値のラベルが必要
)

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=range(len(acc))

# 正解率をプロット
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

# 損失値をプロット
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




