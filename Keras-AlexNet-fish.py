from keras.models import Sequential
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.callbacks import TensorBoard
import time
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Alex_net import Alex_Net

#指定GPU，限制GPU内存
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))


BATCHSIZE = 128
IMG_SIZE = (227, 227)

# 数据部分参考：https://blog.csdn.net/mieleizhi0522/article/details/82191331
# 训练集，测试集文件路径
train_path = './data/train'
test_path = './data/val'


s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())  #时间戳

# image_batch_generator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


#训练集batch生成器
train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCHSIZE,
        shuffle=1,
        color_mode='rgb',
        class_mode='categorical')

#测试集batch生成器
validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=IMG_SIZE,
        color_mode='rgb',
        shuffle=1,
        batch_size=BATCHSIZE,
        class_mode='categorical')

labels = train_generator.class_indices
print(validation_generator.classes)

model = Alex_Net(IMG_SIZE, class_num=16)

#优化器
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()

#logs文件路径
logs_path = './logs'


try:
    os.makedirs(logs_path)
except:
    pass

#将loss ，acc， val_loss ,val_acc记录tensorboard
tensorboard = TensorBoard(log_dir=logs_path)

# 模型训练
history = model.fit_generator(
                              train_generator,
                              epochs=350,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks=[tensorboard]
                             )

# # 保存整个模型
# model.save('model_WVD_3_64.h5')

# 保存模型的权重
model.save_weights('model_weights_AlexNet.h5')

# 加载权重
# model.load_weights('model_weights_WVD_3_64.h5')

# 模型测试
# train_predicts = model.predict_generator(train_generator, steps=85, verbose=1)
# predicts = model.predict_generator(validation_generator, steps=29, verbose=1)
#
# y_train = np.argmax(train_predicts, axis=1)
# y_pred = np.argmax(predicts, axis=1)
#
# train_classes = train_generator.classes
# true_classes = validation_generator.classes

################################### 绘制acc and loss ##########################################

plt.plot(history.history['acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()

plt.plot(history.history['loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

################################### 绘制acc and loss ends ##########################################

'''
################################### 绘制混淆矩阵 ##########################################
# For synthetic test
abels = ['anemone fish', 'barracouta', 'coho', 'eel', 'electric ray', 'gar', 'goldfish', 'great white shark',
              'hammerhead shark',
              'lionfish', 'puffer', 'rock beauty', 'stingray', 'sturgeon', 'tench', 'tiger shark']

# For real test
# labels = ['Flicker', 'Sag & Har', 'Swell']


tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

################################### 绘制训练数据混淆矩阵 ##########################################
cm_train = confusion_matrix(train_classes, y_train)
np.set_printoptions(precision=2)
cm_train_normalized = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
# cm_normalized = cm
print(cm_train_normalized)
plt.figure(figsize=(16, 16), dpi=150)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_train_normalized[y_val][x_val]
    if c > 0.001:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=10, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_train_normalized, title='Normalized confusion matrix train')
plt.show()

################################### 绘制测试数据混淆矩阵 ##########################################
cm_pred = confusion_matrix(true_classes, y_pred)
np.set_printoptions(precision=2)
cm_pred_normalized = cm_pred.astype('float') / cm_pred.sum(axis=1)[:, np.newaxis]
# cm_normalized = cm
print(cm_pred_normalized)
plt.figure(figsize=(16, 16), dpi=150)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_pred_normalized[y_val][x_val]
    if c > 0.001:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=10, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_pred_normalized, title='Normalized confusion matrix test')
plt.show()
'''
