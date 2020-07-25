import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('C:\\Users\\sebas\\Desktop\\Project_Telespazio\\Saved_Model')
tf_lite_model = converter.convert()

open("Flowers_Classification_Lite.tflite", "wb").write(tf_lite_model)

optimize = "Speed"
if optimize == 'Speed':
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
elif optimize =='Storage':
      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
else:
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      
#reduce the size of a floating point model by quantizing the weights to float16
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

#save the quanitized model to a binary file
open("Opt_Flowers_Classification_Lite.tflite", "wb").write(tflite_quant_model)