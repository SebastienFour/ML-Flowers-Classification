from Flowers_Classification import *
from Model_Train import return_name

def predict_one_image(img, model):
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
    img = np.reshape(img, (1, IMG_WIDTH, IMG_HEIGHT, 3))
    img = img/255.
    pred = model.predict(img)
    class_num = np.argmax(pred)
    return class_num, np.max(pred)

test_img = cv2.imread('D:\\Datasets\\Predictions images\\SUNFLOW.jpg')
pred, probability = predict_one_image(test_img, model_simple)
print('%s %d%%' % (labels[pred], round(probability, 2) * 100))
_, ax = plt.subplots(1)
plt.imshow(convertToRGB(test_img))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()

#Saving the model
saved_model_path = 'C:\\Users\\sebas\\Desktop\\Project_Telespazio\\Saved_Model'
tf.saved_model.save(model_simple, saved_model_path)
model_simple.save_weights('D:\\Datasets\\Weights')