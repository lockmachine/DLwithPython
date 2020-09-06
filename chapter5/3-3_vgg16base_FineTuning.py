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

'''
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   ←ここまで凍結
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
'''
# 一旦すべての層を解凍
conv_base.trainable = True

# 特定の層(block4_conv3)までを凍結
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.summary()

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

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=range(len(acc))

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 正解率をプロット
plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

# 損失値をプロット
plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




