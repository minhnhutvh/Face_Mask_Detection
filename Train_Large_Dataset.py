# import các package cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import FineTurn as FCN
import matplotlib.pyplot as plt
import numpy as np

INIT_LR = 1e-4
EPOCHS = 50
BS = 32

# Tăng cường dữ liệu
train_datagen = ImageDataGenerator(
                          rescale=1./255,
                          rotation_range=40,
                          width_shift_range=0.2,
                          height_shift_range=0.2,
                          shear_range=0.2,
                          zoom_range=0.2,
                          horizontal_flip=True,
                          fill_mode='nearest')

# Data flow
train_generator = train_datagen.flow_from_directory(
                            directory="data/train/",
                            target_size=(224, 224),
                            color_mode="rgb",
                            batch_size=BS,
                            class_mode="categorical",
                            shuffle=True,
                            seed=42
                        )
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
                            directory="data/valid/",
                            target_size=(224, 224),
                            color_mode="rgb",
                            batch_size=BS,
                            class_mode="categorical",
                            shuffle=True,
                            seed=42
                        )

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
                            directory="data/test/",
                            target_size=(224, 224),
                            color_mode="rgb",
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                            )
n_train_steps = train_generator.n//train_generator.batch_size
n_valid_steps = valid_generator.n//valid_generator.batch_size


# Nạp lớp cơ sở của mạng MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# Tạo ra head của model và đặt vào phần (top) phái trên lớp cơ sở của model

headModel = FCN.FCHeadNet.build(baseModel,3)
model = Model(inputs=baseModel.input, outputs=headModel)

# Lặp qua các lớp cơ sở và đóng băng
for layer in baseModel.layers:
	layer.trainable = False

# Biên dịch model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"]) # 3 lớp

# train head của mạng
print("[INFO] training head...")

H = model.fit(train_generator,
            steps_per_epoch=n_train_steps,
            validation_data=valid_generator,
            validation_steps=n_valid_steps,
            epochs=EPOCHS)

model.summary()
model.save("MobileNetV2.h5")  # Lưu file model sau khi train

# Đánh giá model: độ chính xác, hàm mất, vẽ đồ thị ....
print("[INFO] evaluating network...")
n_test_steps = test_generator.n
test_generator.reset()
y_pred = model.predict(test_generator,steps=n_test_steps,verbose=1)
y_pred = np.argmax(y_pred,axis=1)
print(classification_report(test_generator.classes,y_pred,target_names=["Correct_mask", "InCorrect_mask","Without_mask"]))

def plot_trend_by_epoch(tr_value,val_value,title,y_plot,figure):
    epoch_num =range(len(tr_value))
    plt.plot(epoch_num,tr_value,'r')
    plt.plot(epoch_num, val_value,'b')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_plot)
    plt.legend(['Training ' + y_plot, 'Validation ' + y_plot])
    plt.savefig(figure)
tr_accuracy,val_accuracy = H.history["accuracy"],H.history["val_accuracy"]
plot_trend_by_epoch(tr_accuracy,val_accuracy,"Model Accuracy","Accuracy","plot_accu.png")
plt.clf()
tr_loss,val_loss = H.history["loss"],H.history["val_loss"]
plot_trend_by_epoch(tr_loss,val_loss,"Model Loss","Loss","plot_loss.png")
