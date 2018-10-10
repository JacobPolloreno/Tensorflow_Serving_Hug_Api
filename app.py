#!/bin/python3
import config
import base64
import hug
import googleapiclient.discovery as discovery
import io
import logging
import nmslib
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import sys

from PIL import Image
from falcon_require_https import RequireHTTPS
from random import randint
from matchzoo import engine

logger = logging.getLogger(__name__)

pkg_dir = os.path.join(config.WORKBUDDY_DIR, 'src')
sys.path.append(pkg_dir)

from officeanswers.preprocess import build_document_embeddings
from officeanswers.model import get_inference_model
from officeanswers.search import build_search_index
from officeanswers.util import Config

api = hug.API(__name__)
# api.http.add_middleware(RequireHTTPS())


_PROJECT = 'grounded-gizmo-187521'

logger.info("Build search index...")

config_path = config.WORKBUDDY_CONFIG
if not config_path or not os.path.exists(config_path):
    raise IOError("Config file not or envionmental variable not set." +
                  "Make sure OA_CONFIG is set to config file path" +
                  f"Path supplied {config_path}")

base_dir = config.WORKBUDDY_DIR

config = Config()
config.from_json_file(config_path)
data_dir = os.path.join(base_dir, 'data')
preprocess_dir = os.path.join(data_dir,
                              'preprocessed')
processed_dir = os.path.join(data_dir,
                             'processed')
config.paths['preprocess_dir'] = preprocess_dir
config.paths['processed_dir'] = processed_dir

# Universal Embeddings
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)

model_config = tf.ConfigProto(
    device_count={'GPU': 0}
)

dataset_path = config.inputs['share']['custom_corpus']
dataset_path = os.path.join(base_dir, dataset_path[2:])
index = 'ue_index'

docs = []
with open(dataset_path, 'r') as f:
    for line in f:
        line = line.strip()
        try:
            question, doc, label = line.split('\t')
        except ValueError:
            error_msg = "Invalid format for relation text." + \
                "Should be `question\tdocument\tlabel`\n" + f"{line}"
            raise ValueError(error_msg)
        docs.append(doc)

search = nmslib.init(method='hnsw', space='cosinesimil')
search.loadIndex(index)

session = tf.Session(config=model_config)
session.run([tf.global_variables_initializer(),
             tf.tables_initializer()])
text = tf.placeholder(dtype=tf.string, shape=[None])
embed_query = embed(text)

# embed_model = get_inference_model(config)
# if 'preprocess' in config.inputs['share']:
#     pre = engine.load_preprocessor(preprocess_dir,
#                                    config.inputs['share']['preprocess'])
# else:
#     pre = engine.load_preprocessor(preprocess_dir,
#                                    config.net_name)

# config.inputs['share']['custom_corpus'] = os.path.join(
#     base_dir,
#     config.inputs['share']['custom_corpus'])
# docs, embeds = build_document_embeddings(config)

# logger.info("Loading search index...")
# index_name = 'custom_index'
# if not os.path.exists(index_name):
#     logger.info("Search index not found. Building it...")
#     search_engine = build_search_index(embeds)
#     search_engine.saveIndex(index_name)
# else:
#     search_engine = nmslib.init(method='hnsw', space='cosinesimil')
#     search_engine.loadIndex(index_name)

# logger.info("Model ready to query.")


@hug.local()
@hug.not_found()
def not_found_handler():
    return "Not Found"


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


# @hug.local()
# @hug.post()
# def workbuddy(body, cors: hug.directives.cors="*"):
#     try:
#         text = str(body['text'], 'utf-8')

#         sparse_input = pre.transform_list([text])[0]
#         sparse_input = np.expand_dims(sparse_input, axis=0)
#         dense_input = embed_model.predict(sparse_input)[0]

#         idxs, dists = search_engine.knnQuery(dense_input, k=3)
#         res = []
#         for idx, dist in zip(idxs, dists):
#             res.append((dist, docs[idx]))
#         res.sort(key=lambda x: x[0], reverse=True)

#         output = {}
#         for k, v in enumerate(res):
#             dist, doc = v
#             output[str(k)] = {'dist': str(dist), 'doc': doc}

#         return {'results': output}, 200

#     except Exception as e:
#         return {'message': 'Internal error.' +
#                 ' {}'.format(e)}, 500


@hug.local()
@hug.post()
def unibuddy(body, cors: hug.directives.cors="*"):
    try:
        # data = str(body['text'], 'utf-8')
        data = body['text']
        dense_input = session.run(embed_query, feed_dict={text: [data]})

        idxs, dists = search.knnQuery(dense_input, k=3)
        res = []
        for idx, dist in zip(idxs, dists):
            res.append((dist, docs[idx]))
        res.sort(key=lambda x: x[0], reverse=True)

        output = {}
        for k, v in enumerate(res):
            dist, doc = v
            output[str(k)] = {'dist': str(dist), 'doc': doc}

        print(output)

        output['9'] = {'dist': '000000', 'doc': 'test'}

        return {'results': output}, 200
    except Exception as e:
        return {'message': 'Internal error.' +
                ' {}'.format(e)}, 500


if __name__ == '__main__':
    api.http.serve()
