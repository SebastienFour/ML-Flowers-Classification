import time
from Flowers_Classification import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

#Preparing input data
batch_size = 32
epochs1 = 20
epochs2 = 10

#Preventing overfitting via data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(train_images)

#Training process
#Simple model case
start = time.time()
Simple_model_info = model_simple.fit_generator(datagen.flow(train_images,train_labels, batch_size=batch_size),
                              epochs = epochs1, validation_data = (test_images,test_labels),
                              verbose = 2, steps_per_epoch=train_images.shape[0] // batch_size, callbacks=[cp_callback_simple_model])

#Showing results for simple model
plt.plot(Simple_model_info.history['loss'])
plt.plot(Simple_model_info.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(Simple_model_info.history['acc'])
plt.plot(Simple_model_info.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

end = time.time()
duration = end - start
print ('\n simple_model took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs1), "\n")

#MobileNetV2 Model
start = time.time()
MobileNet_model_Info = MobileNet.fit_generator(datagen.flow(train_images,train_labels, batch_size=batch_size),
                              epochs = epochs2, validation_data = (test_images,test_labels),
                              verbose = 2, steps_per_epoch=train_images.shape[0] // batch_size, callbacks=[cp_callback_mobileNetV2])

#Showing results for MobileNetV2 model
plt.plot(MobileNet_model_Info.history['loss'])
plt.plot(MobileNet_model_Info.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(MobileNet_model_Info.history['acc'])
plt.plot(MobileNet_model_Info.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

end = time.time()
duration = end - start
print ('\n MobileNetV2 model took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs2), "\n")

def return_name(label_arr):
    idx = np.where(label_arr == 1)
    return idx[0][0]

#Testing true accuracy
test_loss, test_acc = model_simple.evaluate(test_images, test_labels)
print('\nTest accuracy for simple model:\n', test_acc)

test_loss, test_acc = MobileNet.evaluate(test_images, test_labels)
print('Test accuracy for ModelNetV2 model:\n', test_acc)