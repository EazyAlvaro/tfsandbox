from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

def main():
    print('Just the estimation!')

    IRIS_TEST = "iris_test.csv"

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/var/www/tensorflow/iris_model")

    # Classify two new flower samples.
    new_samples = np.array(
          [[6.4, 3.2, 4.5, 1.5],
           [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": new_samples},
          num_epochs=1,
          shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
    main()
