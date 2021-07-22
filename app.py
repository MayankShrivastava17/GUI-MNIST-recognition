import tensorflow as tf
import keras
import gradio as gr

model = tf.keras.models.load_model('gui-mnist.h5')

def classify(image):
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

sketchpad = gr.inputs.Sketchpad()

label = gr.outputs.Label(num_top_classes=3)

interface = gr.Interface(classify, sketchpad, label, live=True, capture_session=True)

interface.launch(share=True)