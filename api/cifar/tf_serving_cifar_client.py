import logging
import operator
import tensorflow as tf
import settings

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

log = logging.getLogger(__name__)


def __open_tf_server_channel__(host: str, port: str):
    """Opens channel to TensorFlow server for requests

    :param host: String, server name(localhost, IP Address)
    :param port: String, server port
    :return: Channel stub
    """
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    return stub


def __create_prediction_request__(image):
    """Creates prediction request to TF server for CIFAR model

    :param np.ndarray, image for prediction
    :return: PredictRequest object
    """
    # Create request
    request = predict_pb2.PredictRequest()

    # Call CNN model to make prediciton on the image
    request.model_spec.name = settings.CIFAR_MODEL_NAME
    request.model_spec.signature_name = settings.CIFAR_MODEL_SIGNATURE_NAME
    request.inputs[settings.CIFAR_MODEL_INPUTS_KEY].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1]))

    return request


def __make_prediction_and_prepare_results__(stub, request):
    """Sends predict request over a channel stub to TF server

    :param stub: Channel stub
    :param request: PredictRequest object
    :return: List of Tuples, 3 most probable labels
    """
    label_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    result = stub.Predict(request, 60.0)  # 60 secs timeout
    probs = result.outputs['scores'].float_val
    value_dict = {label_names[idx]: prob for idx, prob in enumerate(probs)}
    sorted_values = sorted(
        value_dict.items(),
        key=operator.itemgetter(1),
        reverse=True)

    return sorted_values


def make_prediction(image):
    """Predict the CIFAR label on the image using the CIFAR model

    :param image: np.ndarray, image for prediction
    :return List of tuples, 3 most probable labels with their predictions
    """
    # Get TF server connection params
    host, port = settings.TF_SERVER_NAME, settings.TF_SERVER_PORT
    log.info('Connecting to TensorFlow server %s:%s', host, port)

    # Open channel to TF server
    stub = __open_tf_server_channel__(host, port)

    # Create predict request
    request = __create_prediction_request__(image)

    # Make prediction
    return __make_prediction_and_prepare_results__(stub, request)
