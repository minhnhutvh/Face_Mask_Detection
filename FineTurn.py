# Định nghĩa lớp Top cho mô hình
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from tensorflow.keras.layers import AveragePooling2D

class FCHeadNet:
    @staticmethod
    def build(baseModel,classes):
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes, activation="softmax")(headModel)
        return headModel