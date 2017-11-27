import base64
import hug
import io

from api.cifar.tf_serving_cifar_client import make_prediction

api = hug.API(__name__)
api.http.add_middleware(hug.middleware.CORSMiddleware(api, max_age=10))

# TODO: Add multiple image predicitons (batch?)


@hug.local()
@hug.post()
@hug.cli()
def predict_cifar(body,
                  sequence_length: hug.types.number=200):
    """Generate text from our bedtime stories model.

    :param int: sequence_length, how much text to generate
    """

    try:
        # Assume that image is coming as byte string
        image_file = body['image']
        image_bytes = io.BytesIO(image_file)
    except Exception as e:
        return {'message': 'Error in input requests.' +
                ' {}'.format(e)}, 400

    try:
        results = make_prediction(image_bytes.getvalue())
        results_json = [{'label': res[0],
                         'prob': res[1]} for res in results]
        img_64 = base64.b64encode(
            image_bytes.getvalue()).decode()
        # pred_label, _ = max(results, key=lambda r: r[1])
        return {'src': img_64,
                'prediction_result': results_json}, 200
    except Exception as e:
        return {'message': 'Internal error.' +
                ' {}'.format(e)}, 500


if __name__ == '__main__':
    predict_cifar.interface.cli()
