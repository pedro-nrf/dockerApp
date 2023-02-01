import tensorflow as tf
import requests
from PIL import Image
import numpy as np
import sys
from io import BytesIO



def predict_image(image_uri):
  response = requests.get(image_uri)
  img = Image.open(BytesIO(response.content))
  image_array = np.array(img)

  # Resize the image
  image_array = tf.keras.preprocessing.image.array_to_img(image_array)
  image_array = image_array.resize((224, 224))
  image_array = tf.keras.preprocessing.image.img_to_array(image_array)
  image_array = tf.keras.applications.mobilenet.preprocess_input(image_array[tf.newaxis, ...])

  # Load the pre-trained model
  model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

  # Make a prediction
  prediction = model.predict(image_array)

  # Get the class label with the highest probability
  label = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)[0][0][1]

  return label

   
if __name__ == "__main__":
   while True:
    image_uri = input("Please enter the image URI (or 0 to exit): ")
    if image_uri == '0':
            break
    prediction = predict_image(image_uri)
    print(f"The image is predicted to be a {prediction}")