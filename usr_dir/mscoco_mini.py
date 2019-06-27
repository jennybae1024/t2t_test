from tensor2tensor.data_generators import mscoco
from tensor2tensor.utils import registry


@registry.register_problem
class ImageMsCocoMiniTokens32k(mscoco.ImageMsCocoTokens32k):
    """MSCOCO, 8k tokens vocab."""

    def generator(self, data_dir, tmp_dir, is_training):
        # We use the translate vocab file as the vocabulary for captions.
        # This requires having the vocab file present in the data_dir for the
        # generation pipeline to succeed.
        vocab_filename = self.vocab_problem.vocab_filename
        if is_training:
            return mscoco.mscoco_generator(
                data_dir,
                tmp_dir,
                True,
                8000,
                vocab_filename=vocab_filename)
        else:
            return mscoco.mscoco_generator(
                data_dir,
                tmp_dir,
                False,
                4000,
                vocab_filename=vocab_filename)