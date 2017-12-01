import logging
import settings
import tensorflow as tf


from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from typing import Callable
from typing import List

log = logging.getLogger(__name__)


class PredictService(object):
    def __init__(self,
                 model_name: str,
                 model_signature_name: str,
                 model_inputs_key: str,
                 post_process_fn: Callable,
                 host: str=settings.TF_SERVER_NAME,
                 port: str=settings.TF_SERVER_PORT) -> None:

        log.info('Connecting to TensorFlow server %s:%s', host, port)

        # Open TF Server Channel
        self.channel = implementations.insecure_channel(
            host,
            int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            self.channel)

        # Create request
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature_name

        # We'll set inputs in make_prediction method
        self.inputs_key = model_inputs_key

        self.post_process_fn = post_process_fn

    def make_prediction(self,
                        input_data,
                        timeout: float=10.0) -> List:
        """Perform prediction

        """

        self.request.inputs[self.inputs_key].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_data,
                                              shape=[1]))
        result = self.stub.Predict(
            self.request,
            timeout)

        return self.post_process_fn(result)
