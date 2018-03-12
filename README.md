# Machine Learning Demos
LIVE: [demo.jacobpolloreno.com](https://demo.jacobpolloreno.com)

##### Current Deployed Models
* [Cifar 10](https://demo.jacobpolloreno.com/cifar)

## Description
This is the **REST API** that sits between the frontend server and Google's Cloud ML, where my personal ML models are hosted.

It uses a Python API called [Hug](https://github.com/timothycrosley/hug) which facilitates the Google API client request. After the user inputs data into the request model, the server does some of the preprocessing for the models such as resizing and converting images into b64. A request is then sent to the model and a response received with labels and probs.

* Code for the front-end can be found [here](https://github.com/JacobPolloreno/ML_Demos_Showcase). It uses a python web server and Google's api client.
* Code for the custom model(s) can be found [here](https://github.com/JacobPolloreno/Tensorflow_Serving_Models). They are built with [Tensorflow](https://github.com/tensorflow/tensorflow) and saved to be used on a [Tensorflow Serving](https://github.com/tensorflow/serving) host(i.e. Google Cloud ML Engine)