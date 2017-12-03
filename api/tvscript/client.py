import settings

from api.client_base import PredictService


def tvscript_post_process(result):
    """Returns word

    :param result from TF server
    :return: str
    """
    return str(result.outputs['labels'].string_val.pop(), 'utf-8')


tvscript_client = PredictService(
    settings.TVSCRIPT_MODEL_NAME,
    settings.TVSCRIPT_MODEL_SIGNATURE_NAME,
    settings.TVSCRIPT_MODEL_INPUTS_KEY,
    tvscript_post_process)
