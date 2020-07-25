import tensorflow as tf

#Simple model conversion
converter1 = tf.lite.TFLiteConverter.from_saved_model('C:\\Users\\sebas\\Desktop\\Project_Telespazio\\Final Code\\Saved_Models\\Simple_Model')
tf_lite_model1 = converter1.convert()

open("Flowers_Classification_Lite(Simple_model).tflite", "wb").write(tf_lite_model1)

optimize = "Speed"
if optimize == 'Speed':
  converter1.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
elif optimize =='Storage':
      converter1.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
else:
      converter1.optimizations = [tf.lite.Optimize.DEFAULT]
      
#reduce the size of a floating point model by quantizing the weights to float16
converter1.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter1.convert()

#save the quanitized model to a binary file
open("Opt_Flowers_Classification_Lite(Simple_model).tflite", "wb").write(tflite_quant_model)

#MobilNetV2 model conversion (Keras Format)
model_dir = 'C:\\Users\\sebas\\Desktop\\Project_Telespazio\\Final Code\\Saved_Models\\MobilNetV2'
converter2 = tf.lite.TFLiteConverter.from_saved_model(model_dir, signature_keys=['serving_default'])

converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter2.convert()

open('tflite_model.tflite','wb').write(tflite_model)