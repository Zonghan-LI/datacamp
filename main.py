import numpy as np
import tensorflow as tf
import gradio

model = tf.keras.models.load_model('cnn.h5')

def pneumoniaPrediction(img):
    img = (np.array(img)/255).reshape(-1, 64, 64, 3)
    return "Normal" if np.argmax(model.predict(img)[0]) < 0.5 else "Pneumonic"
   

img = gradio.inputs.Image(shape=(64, 64))
label = gradio.outputs.Label(num_top_classes=1)

interface = gradio.Interface(fn = pneumoniaPrediction,
                            title = "Pneumonia Detection using Chest X-Ray",
                            inputs = img,
                            outputs = label,
                            interpretation = "default",
                            server_port = 8000)

interface.launch(debug=True, share=True)
