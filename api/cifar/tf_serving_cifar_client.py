import operator
import settings

from api.client_base import PredictService


# TODO(can do this in the TF serving export model)
def cifar_post_process(result):
    """Prepares the results for output

    :param result
    :return: List of Tuples, 3 most probable labels
    """
    probs = result.outputs['scores'].float_val
    label_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    value_dict = {label_names[idx]: prob for idx, prob in enumerate(probs)}
    sorted_values = sorted(
        value_dict.items(),
        key=operator.itemgetter(1),
        reverse=True)

    return sorted_values

cifar_client = PredictService(
    settings.CIFAR_MODEL_NAME,
    settings.CIFAR_MODEL_SIGNATURE_NAME,
    settings.CIFAR_MODEL_INPUTS_KEY,
    cifar_post_process)
