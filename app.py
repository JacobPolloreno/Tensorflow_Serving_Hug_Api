#!/bin/python3
import base64
import hug
import googleapiclient.discovery as discovery
import io

from PIL import Image
from falcon_require_https import RequireHTTPS
from random import randint

api = hug.API(__name__)
#api.http.add_middleware(hug.middleware.CORSMiddleware(api, max_age=10))
api.http.add_middleware(RequireHTTPS())

# TODO(Add multiple image predicitons, batch?)

_PROJECT = 'grounded-gizmo-187521'


@hug.local()
@hug.post()
@hug.cli()
def predict_cifar(body):
    _MODEL = 'cifar10'
    _VERSION = 'b1'

    try:
        # Assume that image is coming as byte string
        image_bytes = body['image']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((32, 32), Image.BILINEAR)
        resized_image = io.BytesIO()
        image.save(resized_image, format='JPEG')
        encoded_image = base64.b64encode(
            resized_image.getvalue()).decode('utf-8')
        instance = {'key': str(randint(1, 100)),
                    'image_bytes': {'b64': encoded_image}}
    except Exception as e:
        return {'message': 'Error in input requests.' +
                ' {}'.format(e)}, 400

    # CREATE SERVICE
    service = discovery.build('ml', 'v1')
    name = f'projects/{_PROJECT}/models/{_MODEL}/versions/{_VERSION}'

    try:
        response = service.projects().predict(
            name=name,
            body={'instances': [instance]}
        ).execute()

        if 'error' in response:
            return {'message': response['error']}, 500

        results = response['predictions'].pop()
        prediction_result = []
        for label, prob in zip(results['classes'],
                               results['scores']):

            prediction_result.append({
                'label': label,
                'prob': prob})

        return {'prediction_result': prediction_result}, 200
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

        # TOKEN LOOKUP
        token_dict = dict([
            ('--', '||dash||'), ('.', '||period||'), (',', '||comma||'),
            ('"', '||quotation_mark||'), (';', '||semicolon||'),
            ('!', '||exclamation_mark||'), ('?', '||question_mark||'),
            ('(', '||left_parentheses||'), (')', '||right_parentheses||'),
            ('\n', '||return||')
        ])

        for key, token in token_dict.items():
            text = text.replace(key, ' {} '.format(token))

        for _ in range(80):
            next_word = tvscript_client.make_prediction(text, 10.0)
            text = ' '.join([text, next_word])

        for key, token in token_dict.items():
            text = text.replace(' ' + token.lower(), key)
        text = text.replace('( ', '(')

        return {'prediction_result': text}, 200
    except Exception as e:
        return {'message': 'Internal error.' +
                ' {}'.format(e)}, 500

@hug.not_found()
def not_found_handler():
    return "Not Found"


@hug.post('/search')
def search_insight(body):
   return {"results" : "This is a sentence"}


if __name__ == '__main__':
    #predict_cifar.interface.cli()
    api.http.serve()
