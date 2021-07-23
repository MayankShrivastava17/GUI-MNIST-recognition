import tensorflow as tf
import keras
import gradio as gr
import cv2 as cv
import numpy as np

model = tf.keras.models.load_model('gui-mnist.h5')

def prepration(img):
    img = np.asarray(img)              # convert to array 
    img = cv.resize(img, (28, 28 ))   # resize to target shape 
    img = cv.bitwise_not(img)         # [optional] my input was white bg, I turned it to black - {bitwise_not} turns 1's into 0's and 0's into 1's
    img = img / 255                    # normalize 
    img = img.reshape(1, 784)          # reshaping 
    return img

def classify(img):
    image = prepration(img)
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

# sketchpad = gr.inputs.Sketchpad()

label = gr.outputs.Label(num_top_classes=3)

# interface = gr.Interface(classify, sketchpad, label, live=True, capture_session=True)
gr.Interface(fn=classify, inputs="sketchpad", outputs=label).launch()

# interface.launch(share=True)