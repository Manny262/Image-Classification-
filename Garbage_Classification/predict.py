import tensorflow as tf 
from tensorflow import keras # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
import numpy as np

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = keras.models.load_model('garbage_classifier.keras', custom_objects={'preprocess_input': preprocess_input})


def predict_image(img_path: str) -> str:
    """
    Classify a garbage image using the trained model.
    
    Arguments:
        img_path: Path to the image file to classify
        
    Returns:
        A formatted string with the predicted class and confidence
    """
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = tf.nn.softmax(predictions[0])

    result = f"Predicted: {predicted_class} ({100 * np.max(confidence):.1f}% confidence)"
    print(result)
    return result


if __name__ == "__main__":
    # Example usage: change the image path here
    predict_image("Dataset/Garbage classification/glass/glass1.jpg")