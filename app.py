import base64
import hug
import io

from api.cifar.tf_serving_cifar_client import cifar_client
from api.tvscript.tf_serving_tvscript_client import tvscript_client

api = hug.API(__name__)
api.http.add_middleware(hug.middleware.CORSMiddleware(api, max_age=10))

# TODO(Add multiple image predicitons, batch?)


@hug.local()
@hug.post()
@hug.cli()
def predict_cifar(body):
    """Generate text from our bedtime stories model.

    """

    try:
        # Assume that image is coming as byte string
        image_file = body['image']
        image_bytes = io.BytesIO(image_file)
    except Exception as e:
        return {'message': 'Error in input requests.' +
                ' {}'.format(e)}, 400

    try:
        results = cifar_client.make_prediction(image_bytes.getvalue())
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


@hug.local()
@hug.post()
@hug.cli()
def predict_tvscript(body):
    """Generate text from our bedtime stories model.

    """

    try:
        text = str(body['text'], 'utf-8')

        for _ in range(80):
            next_word = tvscript_client.make_prediction(text, 10.0)
            text = ' '.join([text, next_word])

        # TOKEN LOOKUP
        token_dict = dict([
            ('--', '||dash||'), ('.', '||period||'), (',', '||comma||'),
            ('"', '||quotation_mark||'), (';', '||semicolon||'),
            ('!', '||exclamation_mark||'), ('?', '||question_mark||'),
            ('(', '||left_parentheses||'), (')', '||right_parentheses||'),
            ('\n', '||return||')
        ])

        for key, token in token_dict.items():
            text = text.replace(' ' + token.lower(), key)
        # text = text.replace('\n ', '\n')
        text = text.replace('( ', '(')

        return {'prediction_result': text}, 200
    except Exception as e:
        return {'message': 'Internal error.' +
                ' {}'.format(e)}, 500

if __name__ == '__main__':
    predict_cifar.interface.cli()
