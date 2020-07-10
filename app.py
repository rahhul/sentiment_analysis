# python3
import streamlit as st
import tensorflow_datasets as tfds
import tensorflow as tf


print(f"Tf version: {tf.__version__}")
print(f"Tfds version: {tfds.__version__}")


# variables
padding_size = 1000
model = tf.keras.models.load_model('sentiment_analysis.hdf5')

text_encoder = tfds.features.text.TokenTextEncoder.load_from_file('sa_encoder.vocab')

"""
# Sentiment Analysis
#### Tensorflow model was trained on Amazon Customer Reviews - Mobile Electronics dataset.

"""


"""
Type a review text into the box below.\n
The model will return a sentiment score."""

# padding function
def pad_to_size(vec, size):
    # calculate padding
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

# prediction function
def predict_fn(pred_text, pad_size):
    """Apply padding, encode with tokens
    and return predictions"""
    encoded_pred_text = text_encoder.encode(pred_text)
    print(encoded_pred_text)
    # apply zero padding
    encoded_pred_text = pad_to_size(encoded_pred_text, pad_size)
    # cast to int64
    encoded_pred_text = tf.cast(encoded_pred_text, tf.int64)
    # run predictions
    predictions = model.predict(tf.expand_dims(encoded_pred_text, 0))
    return (predictions)




review = st.text_input('Enter review text here...and press Enter')
"""
##### Model and Vocabulary file loaded...
"""

output = predict_fn(review, 1000)

sentiment = 'negative' if output < 0 else 'positive'


st.markdown(f'## Prediction is {output[0][0]}')
st.markdown(f"## Sentiment is '{sentiment}'")
