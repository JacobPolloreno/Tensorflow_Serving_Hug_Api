import config
import os
import tensorflow as tf
import tensorflow_hub as hub
import sys

pkg_dir = os.path.join(config.WORKBUDDY_DIR, 'src')
sys.path.append(pkg_dir)

from officeanswers.search import build_search_index
from officeanswers.util import Config

base_dir = config.WORKBUDDY_DIR
config_path = config.WORKBUDDY_CONFIG
if not config_path or not os.path.exists(config_path):
    raise IOError("Config file not or envionmental variable not set." +
                  "Make sure OA_CONFIG is set to config file path" +
                  f"Path supplied {config_path}")

data_dir = os.path.join(base_dir, 'data')


def create_universal_index():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)

    model_config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    config = Config()
    config.from_json_file(config_path)
    dataset_path = config.inputs['share']['custom_corpus']
    dataset_path = os.path.join(base_dir, dataset_path[2:])
    index = 'ue_index'

    if os.path.exists(index):
        raise IOError(f"Index file({index}) already exists.")

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

    with tf.Session(config=model_config) as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
        embeds = session.run(embed(docs))

    search = build_search_index(embeds)
    search.saveIndex(index)


if __name__ == "__main__":
    create_universal_index()
