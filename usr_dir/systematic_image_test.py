import re, os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import image_utils, text_encoder, generator_utils, mnist
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
import numpy as np
import random
import string
import tensorflow as tf

_MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
_MNIST_TRAIN_DATA_FILENAME = "train-images-idx3-ubyte.gz"
_MNIST_TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
_MNIST_TEST_DATA_FILENAME = "t10k-images-idx3-ubyte.gz"
_MNIST_TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz"
_MNIST_IMAGE_SIZE = 28


class ImageSeq2TextProblem(image_utils.ImageProblem):
    """Base class for image-to-text problems."""

    @property
    def is_character_level(self):
        raise NotImplementedError()

    @property
    def vocab_problem(self):
        raise NotImplementedError()  # Not needed if self.is_character_level.

    @property
    def target_space_id(self):
        raise NotImplementedError()

    @property
    def train_shards(self):
        raise NotImplementedError()

    @property
    def dev_shards(self):
        raise NotImplementedError()

    def generator(self, data_dir, tmp_dir, is_training):
        raise NotImplementedError()

    def example_reading_spec(self):
        label_key = "image/class/label"
        data_fields = {
            "image/encoded": tf.FixedLenFeature((), tf.string),
            "image/format": tf.FixedLenFeature((), tf.string),
        }

        data_items_to_decoders = {
            "inputs":
                tf.contrib.slim.tfexample_decoder.Image(
                    image_key="image/encoded",
                    format_key="image/format",
                    channels=self.num_channels),
        }
        data_fields[label_key] = tf.VarLenFeature(tf.int64)
        data_items_to_decoders[
            "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(label_key)
        return data_fields, data_items_to_decoders

    def feature_encoders(self, data_dir):
        if self.is_character_level:
            encoder = text_encoder.ByteTextEncoder()
        else:
            vocab_filename = os.path.join(
                data_dir, self.vocab_problem.vocab_filename)
            encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        input_encoder = text_encoder.ImageEncoder(channels=self.num_channels)
        return {"inputs": input_encoder, "targets": encoder}

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.modality = {"inputs": modalities.ModalityType.IMAGE,
                      "targets": modalities.ModalityType.SYMBOL}
        p.vocab_size = {"inputs": 256,
                        "targets": self._encoders["targets"].vocab_size}
        p.batch_size_multiplier = 256
        p.loss_multiplier = 1.0
        p.input_space_id = problem.SpaceID.IMAGE
        p.target_space_id = self.target_space_id

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        generator_utils.generate_dataset_and_shuffle(
            self.generator(data_dir, tmp_dir, True),
            self.training_filepaths(data_dir, self.train_shards, shuffled=False),
            self.generator(data_dir, tmp_dir, False),
            self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))



@registry.register_problem
class AlgorithmicImageTiling(image_utils.ImageProblem):
    @property
    def is_character_level(self):
        return True

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def train_shards(self):
        return self.dataset_splits[0]["shards"]

    @property
    def dev_shards(self):
        return self.dataset_splits[1]["shards"]

    @property
    def test_shards(self):
        return self.dataset_splits[2]["shards"]

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 2,
        }]

    def possible_len_by_shard(self, shard):
        if shard == 0:
            return (20, 40)
        elif shard == 1:
            return (20, 400)


    def preprocess_example(self, example, mode, unused_hparams):
        image = example["inputs"]
        image.set_shape([_MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1])
        if not self._was_reversed:
            image = tf.image.per_image_standardization(image)
        example["inputs"] = image
        return example

    def generator(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        training = dataset_split == problem.DatasetSplit.TRAIN
        data_filename = _MNIST_TRAIN_DATA_FILENAME if training else _MNIST_TEST_DATA_FILENAME
        label_filename = _MNIST_TRAIN_LABELS_FILENAME if training else _MNIST_TEST_LABELS_FILENAME
        data_path = os.path.join(tmp_dir, data_filename)
        labels_path = os.path.join(tmp_dir, label_filename)
        images = _extract_mnist_images(data_path, 60000 if training else 10000)
        labels = _extract_mnist_labels(labels_path, 60000 if training else 10000)

        def image_sequence(image_list):
            image_bytes_list = []
            for image in image_list:
                image_bytes = image.tostring()
                image_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                image_bytes_list.append(image_bytes)
            return image_bytes_list

        def label_sequence(label_list):
            """this function takes a list of labels and returns the list in int64"""
            label_int_list = []
            for label in label_list:
                label_int = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                label_int_list.append(label_int)

            return label_int_list

        def _generator(images, labels, shard):
            (min_len, max_len) = self.possible_len_sym_by_shard(shard)
            seqlen = np.random.randint(min_len, max_len)
            pattern_len = np.random.randint(1, seqlen//3)
            indice = list(np.random.randint(0, len(images), size=pattern_len))
            total_indice = []
            while len(total_indice) < 2*seqlen:
                total_indice += indice
            img_indice = total_indice[:seqlen]
            label_indice = total_indice[seqlen:2*seqlen]

            image_byte_list = image_sequence([images[i] for i in img_indice])
            label_int_list = image_sequence([labels[i] for i in label_indice])

            yield {
                "image/encoded": [image[i] for i in img_indice],
                "image/format": ["png"],
                "image/class/label": "".join([str(labels[i]) for i in label_indice]),
                "image/height": [height],
                "image/width": [width]
            }

        if dataset_split == problem.DatasetSplit.TRAIN:
            num_data = self.num_train
            for num in range(num_data):
                return _generator(images, labels, 0)
        elif dataset_split == problem.DatasetSplit.EVAL:
            num_data = self.num_eval
            for num in range(num_data):
                return _generator(images, labels, 1)
        elif dataset_split == problem.DatasetSplit.TEST:
            num_data = self.num_test
            for num in range(num_data):
                for shard in range(2):
                    return _generator(images, labels, shard)


def mnist_seq_generator(tmp_dir, training, how_many, start_from=0):
    for filename in [
        _MNIST_TRAIN_DATA_FILENAME, _MNIST_TRAIN_LABELS_FILENAME,
        _MNIST_TEST_DATA_FILENAME, _MNIST_TEST_LABELS_FILENAME
    ]:
        generator_utils.maybe_download(tmp_dir, filename, _MNIST_URL + filename)
    d = _MNIST_TRAIN_DATA_FILENAME if training else _MNIST_TEST_DATA_FILENAME
    l = _MNIST_TRAIN_LABELS_FILENAME if training else _MNIST_TEST_LABELS_FILENAME
    return mnist_sequence_generator(tmp_dir, training, how_many, d, l, start_from)


def _extract_mnist_images(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
    return data


def _extract_mnist_labels(filename, num_labels):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def mnist_sequence_generator(tmp_dir,
                             training,
                             how_many,
                             data_filename,
                             label_filename,
                             start_from=0):
    data_path = os.path.join(tmp_dir, data_filename)
    labels_path = os.path.join(tmp_dir, label_filename)
    images = _extract_mnist_images(data_path, 60000 if training else 10000)
    labels = _extract_mnist_labels(labels_path, 60000 if training else 10000)
    # Shuffle the data to make sure classes are well distributed.
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = list(zip(*data))
    images = images[start_from:start_from + how_many]
    labels = labels[start_from:start_from + how_many]
    if not images:
        raise ValueError("Must provide some images for the generator.")
    width, height, _ = images[0].shape
    for (enc_image, label) in zip(image_utils.encode_images_as_png(images), labels):
        yield {
            "image/encoded": [enc_image],
            "image/format": ["png"],
            "image/class/label": [int(label)],
            "image/height": [height],
            "image/width": [width]
        }

def mscoco_generator(data_dir,
                     tmp_dir,
                     training,
                     how_many,
                     start_from=0,
                     eos_list=None,
                     vocab_filename=None):

    eos_list = [1] if eos_list is None else eos_list
    def get_vocab():
        """Get vocab for caption text encoder."""
        if data_dir is not None and vocab_filename is not None:
            vocab_filepath = os.path.join(data_dir, vocab_filename)
            if tf.gfile.Exists(vocab_filepath):
                tf.logging.info("Found vocab file: %s", vocab_filepath)
                vocab_symbolizer = text_encoder.SubwordTextEncoder(vocab_filepath)
                return vocab_symbolizer
            else:
                raise ValueError("Vocab file does not exist: %s" % vocab_filepath)
        return None

    vocab_symbolizer = get_vocab()
    _get_mscoco(tmp_dir)
    caption_filepath = (
        _MSCOCO_TRAIN_CAPTION_FILE if training else _MSCOCO_EVAL_CAPTION_FILE)
    caption_filepath = os.path.join(tmp_dir, caption_filepath)
    prefix = _MSCOCO_TRAIN_PREFIX if training else _MSCOCO_EVAL_PREFIX
    caption_file = io.open(caption_filepath)
    caption_json = json.load(caption_file)
    # Dictionary from image_id to ((filename, height, width), captions).
    image_dict = {}
    for image in caption_json["images"]:
        image_dict[image["id"]] = [(image["file_name"], image["height"],
                                    image["width"]), []]
    annotations = caption_json["annotations"]
    annotation_count = len(annotations)
    image_count = len(image_dict)
    tf.logging.info("Processing %d images and %d labels\n" % (image_count,
                                                              annotation_count))
    for annotation in annotations:
        image_id = annotation["image_id"]
        image_dict[image_id][1].append(annotation["caption"])

    data = list(image_dict.values())[start_from:start_from + how_many]
    random.shuffle(data)
    for image_info, labels in data:
        image_filename = image_info[0]
        image_filepath = os.path.join(tmp_dir, prefix, image_filename)
        with tf.gfile.Open(image_filepath, "rb") as f:
            encoded_image_data = f.read()
            height, width = image_info[1], image_info[2]
            for label in labels:
                if vocab_filename is None or vocab_symbolizer is None:
                    label = [ord(c) for c in label] + eos_list
                else:
                    label = vocab_symbolizer.encode(label) + eos_list
                yield {
                    "image/encoded": [encoded_image_data],
                    "image/format": ["jpeg"],
                    "image/class/label": label,
                    "image/height": [height],
                    "image/width": [width]
                }




class AlgorithmicString(text_problems.Text2TextProblem):
    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return True

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def num_train(self):
        return 100000

    @property
    def num_eval(self):
        return 2000

    @property
    def num_test(self):
        return 2000

    def rule(self):
        raise NotImplementedError()

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        raise NotImplementedError()


@registry.register_problem
class AlgorithmicUnseenTilingVVV(AlgorithmicString):
    @property
    def approx_vocab_size(self):
        return 2 ** 5  # ~8k

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 6,
        }]

    def possible_len_sym_by_shard(self, shard):
        if shard == 0:
            return (20, 40), (0, 10)
        elif shard == 1:
            return (20, 40), (10, 20)
        elif shard == 2:
            return (20, 40), (0, 20)
        elif shard == 3:
            return (20, 400), (0, 10)
        elif shard == 4:
            return (20, 400), (10, 20)
        elif shard == 5:
            return (20, 400), (0, 20)

    def generate_input(self, shard):
        (min_len, max_len), (min_sym, max_sym) = self.possible_len_sym_by_shard(shard)
        symbols = string.ascii_lowercase[min_sym:max_sym]
        seqlen = np.random.randint(min_len, max_len)
        pattern_len = np.random.randint(1, seqlen//3)
        pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
        res = ''
        while len(res) < 2*seqlen:
            res += pattern
        return res[:seqlen], res[seqlen:2*seqlen]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        if dataset_split == problem.DatasetSplit.TRAIN:
            num_data = self.num_train
            for num in range(num_data):
                enc, dec = self.generate_input(0)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }
        elif dataset_split == problem.DatasetSplit.EVAL:
            num_data = self.num_eval
            for num in range(num_data):
                enc, dec = self.generate_input(0)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }
        elif dataset_split == problem.DatasetSplit.TEST:
            num_data = self.num_test
            for num in range(num_data):
                for shard in range(6):
                    enc, dec = self.generate_input(shard)
                    yield {
                        "inputs": enc,
                        "targets": dec,
                    }


# class Tiling(text_problems.Text2TextProblem):
#     @property
#     def approx_vocab_size(self):
#         return 2 ** 5  # ~8k
#
#     @property
#     def is_generate_per_split(self):
#         return True
#
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 1,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 3,
#         }, {
#             "split": problem.DatasetSplit.TEST,
#             "shards": 3,
#         }]
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_pattern_lengths(self, dataset_split):
#         raise NotImplementedError()
#
#     def gen(self, pattern, total_len):
#         rep = (2*total_len) // len(pattern)
#         residue = 2*total_len-len(pattern)*rep
#         res = ''
#         while rep > 0:
#             rep-=1
#             res+=pattern
#         res = res + pattern[:residue]
#         return res[:total_len], res[total_len:]
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         lens = self.possible_pattern_lengths(dataset_split)
#         symbols = string.ascii_lowercase[:10]
#
#         if dataset_split == problem.DatasetSplit.TRAIN:
#             num_data = int(1e7)
#             for num in range(num_data):
#                 pattern_len = random.choice(lens)
#                 total_len = random.choice(range(pattern_len*2 + 1, 1+min(30, pattern_len*4)))
#                 pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
#                 enc, dec = self.gen(pattern, total_len)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }
#         else:
#             num_data = int(1e4)
#             for num in range(num_data):
#                 pattern_len = lens[num%3]
#                 total_len = random.choice(range(pattern_len*2 + 1, 1+min(30, pattern_len*4)))
#                 pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
#                 enc, dec = self.gen(pattern, total_len)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }

#     def eval_metrics(self):
#         return [metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ]

# @registry.register_problem
# class TilingExtension(Tiling):
#     def possible_pattern_lengths(self, dataset_split):
#         return [2,3,4,5,6,7,8,9] if dataset_split == problem.DatasetSplit.TRAIN else [10,11,12]
#
# @registry.register_problem
# class TilingInterpolation(Tiling):
#     def possible_pattern_lengths(self, dataset_split):
#         return [2,3,4,5,9,10,11,12] if dataset_split == problem.DatasetSplit.TRAIN else [6,7,8]
#
#
# @registry.register_problem
# class TilingSanity(text_problems.Text2TextProblem):
#     @property
#     def approx_vocab_size(self):
#         return 2 ** 5  # ~8k
#
#     @property
#     def is_generate_per_split(self):
#         return False
#
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 9,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 1,
#         }]
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_pattern_lengths(self, dataset_split):
#         raise NotImplementedError()
#
#     def gen(self, pattern, total_len):
#         rep = (2 * total_len) // len(pattern)
#         residue = 2 * total_len - len(pattern) * rep
#         res = ''
#         while rep > 0:
#             rep -= 1
#             res += pattern
#         res = res + pattern[:residue]
#         return res[:total_len], res[total_len:]
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         lens = [i for i in range(2, 13)]
#         symbols = string.ascii_lowercase[:10]
#
#         num_data = int(1e7)
#         for num in range(num_data):
#             pattern_len = random.choice(lens)
#             total_len = random.choice(range(pattern_len * 2 + 1, 1 + min(30, pattern_len * 4)))
#             pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
#             enc, dec = self.gen(pattern, total_len)
#             yield {
#                 "inputs": enc,
#                 "targets": dec,
#             }





# class SymbolBase(text_problems.Text2TextProblem):
#     @property
#     def is_generate_per_split(self):
#         # generate_data will shard the data into TRAIN and EVAL for us.
#         return True
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_lengths(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def possible_symbols(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def rule(self):
#         raise NotImplementedError()
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         min_len, max_len = self.possible_lengths(dataset_split)
#         min_sym, max_sym = self.possible_symbols(dataset_split)
#
#         if dataset_split == problem.DatasetSplit.TRAIN:
#             symbols = string.ascii_lowercase[:max_sym]
#             num_data = 100000
#             for num in range(num_data):
#                 seqlen = np.random.randint(min_len, max_len + 1)
#                 enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#                 dec = self.rule(enc)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }
#
#         else:
#             num_data = 10000
#             for num in range(num_data):
#                 for seqlen in range(min_len, max_len+1):
#                     for symlen in range(min_sym, max_sym+1):
#                         symbols = string.ascii_lowercase[:symlen]
#                         enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#                         dec = self.rule(enc)
#                         yield {
#                             "inputs": enc,
#                             "targets": dec,
#                         }

# class LengthBase(text_problems.Text2TextProblem):
#     @property
#     def approx_vocab_size(self):
#         return 2 ** 5  # ~8k
#
#     @property
#     def num_data(self, dataset_split):
#         raise NotImplementedError()
#
#     @property
#     def is_generate_per_split(self):
#         # generate_data will shard the data into TRAIN and EVAL for us.
#         return True
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_lengths(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def possible_symbols(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def rule(self):
#         raise NotImplementedError()
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         raise NotImplementedError()
#
# class GeneralizeLengthTrain(LengthBase):
#     def num_data(self, dataset_split):
#         return int(1e6) if dataset_split == problem.DatasetSplit.TRAIN else int(1e4)
#     @property
#     def dataset_splits(self):
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 3,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 1,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12)
#     def possible_symbols(self, dataset_split):
#         return (10, 10)
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         min_len, max_len = self.possible_lengths(dataset_split)
#         min_sym, max_sym = self.possible_symbols(dataset_split)
#
#         num_data = self.num_data(dataset_split)
#         symbols = string.ascii_lowercase[:max_sym]
#         for num in range(num_data):
#             seqlen = np.random.randint(min_len, max_len + 1)
#             enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#             dec = self.rule(enc)
#             yield {
#                 "inputs": enc,
#                 "targets": dec,
#             }
#
# class GeneralizeLengthTest(LengthBase):
#     def num_data(self, dataset_split):
#         return 0 if dataset_split == problem.DatasetSplit.TRAIN else int(1e4)
#
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 1,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 6,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (13, 18)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10)
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         min_len, max_len = self.possible_lengths(dataset_split)
#         min_sym, max_sym = self.possible_symbols(dataset_split)
#
#         num_data = self.num_data(dataset_split)
#         symbols = string.ascii_lowercase[:max_sym]
#         seqlens = [i for i in range(min_len, max_len+1)]
#         # print(dataset_split)
#         # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
#         # print(num_data)
#         for num in range(num_data):
#             for seqlen in seqlens:
#                 enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#                 dec = self.rule(enc)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }


# @registry.register_problem
# class GeneralizeSymbolsCyclic(SymbolBase):
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 10,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 5,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (5, 12)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10) if dataset_split == problem.DatasetSplit.TRAIN else (11, 15)
#
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
#
# @registry.register_problem
# class GeneralizeSymbolsReverse(SymbolBase):
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 10,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 5,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (5, 12)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10) if dataset_split == problem.DatasetSplit.TRAIN else (11, 15)
#
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
#
# @registry.register_problem
# class GeneralizeSymbolsIdentity(SymbolBase):
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 10,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 5,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (5, 12)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10) if dataset_split == problem.DatasetSplit.TRAIN else (11, 15)
#
#     def rule(self, enc_inp):
#         return enc_inp


# @registry.register_problem
# class GeneralizeLengthCyclicSanity(GeneralizeLengthTrain):
#     def possible_lengths(self, dataset_split):
#         return (1, 18)
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
# @registry.register_problem
# class GeneralizeLengthCyclicTrain(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
# @registry.register_problem
# class GeneralizeLengthCyclicTest(GeneralizeLengthTest):
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
#
# @registry.register_problem
# class GeneralizeLengthReverseSanity(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
#     def possible_lengths(self, dataset_split):
#         return (1, 18)
#
# @registry.register_problem
# class GeneralizeLengthReverseTrain(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
# @registry.register_problem
# class GeneralizeLengthReverseTest(GeneralizeLengthTest):
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
# @registry.register_problem
# class GeneralizeLengthIdentitySanity(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return enc_inp
#
#     def possible_lengths(self, dataset_split):
#         return (1, 18)
#
# @registry.register_problem
# class GeneralizeLengthIdentityTrain(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return enc_inp
#
# @registry.register_problem
# class GeneralizeLengthIdentityTest(GeneralizeLengthTest):
#     def rule(self, enc_inp):
#         return enc_inp
