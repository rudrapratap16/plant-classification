from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = FastAPI()

# Load the model architecture
base_model = tf.keras.applications.InceptionResNetV2(weights=None, include_top=False, input_shape=(299, 299, 3))

# Custom classification head
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(80, activation='softmax')(x)  # 80 classes

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Load the weights
model.load_weights("model_weights_v2.weights.h5")

# Set model to evaluation mode
model.trainable = False

def preprocess_image(image_data: Image.Image):
    """Preprocess image to match model input shape (299x299)."""
    image_data = image_data.resize((299, 299))  # Resize to match model input size
    image_array = np.array(image_data).astype(np.float32)
    image_array = preprocess_input(image_array)  # Apply model-specific preprocessing
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_tensor = preprocess_image(image_data)

    # Get predictions
    prediction = model.predict(input_tensor)

    # Convert predictions to a list
    prediction_list = prediction.tolist()
    
    return {"prediction": prediction_list}

