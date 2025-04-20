import logging
import typing
import numpy as np


class Transformer:
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError


class LabelIndexer(Transformer):
    """Convert label to index by vocab

    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """

    def __init__(
            self,
            vocab: typing.List[str]
    ) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):

        indices = []
        for l in label:
            if l in self.vocab:
                indices.append(self.vocab.index(l))

        # Convert the list to a NumPy array
        indices_array = np.array(indices)

        return data, indices_array


class LabelPadding(Transformer):
    """Pad label to max_word_length

    Attributes:
        padding_value (int): Value to pad
        max_word_length (int): Maximum length of label
        use_on_batch (bool): Whether to use on batch. Default: False
    """

    def __init__(
            self,
            padding_value: int,
            max_word_length: int = None,
            use_on_batch: bool = False
    ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_word_length is None:
            raise ValueError("max_word_length must be specified if use_on_batch is False")

    def __call__(self, data: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in label])
            padded_labels = []
            for l in label:
                padded_label = np.pad(l, (0, max_len - len(l)), "constant", constant_values=self.padding_value)
                padded_labels.append(padded_label)

            padded_labels = np.array(padded_labels)
            return data, padded_labels

        label = label[:self.max_word_length]

        padded_label = np.pad(label, (0, self.max_word_length - len(label)), "constant",
                            constant_values=self.padding_value)

        padint = padded_label.astype(int)

        return data, padint
