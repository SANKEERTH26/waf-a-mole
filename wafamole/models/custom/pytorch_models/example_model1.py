import pickle
import re

import torch

from wafamole.models import PyTorchModelWrapper
import wafamole.models.custom.pytorch_models.utils as ut
from wafamole.utils.check import type_check
from wafamole.exceptions.models_exceptions import (
    ModelNotLoadedError,
    PyTorchInternalError,
)
import numpy as np
import json

class PyTorchExample(PyTorchModelWrapper):
    """SQLiGoT wrapper"""

    def __init__(self, filepath: str):
        """Constructs model by loading pretrained net.

        Arguments:
            filepath (str) : the path to the pretrained h5 net

        Raises:
        TypeError: filepath not  string
        FileNotFoundError: filepath not pointing to anything
        NotKerasModelError: filepath not pointing to h5 keras model
        """
        type_check(filepath, str, "filepath")
        from wafamole.models.custom.pytorch_models.ModelClass import SentimentLSTM

        self.filepath = filepath
        p = re.compile('.*ModelWAF(\d+).*')
        model_num = p.findall(filepath)
        self.model_number = model_num[0]
        self.vocabfile = './vocab' + self.model_number + '.json'
        f = open(self.vocabfile)
        self.vocab_to_int = json.load(f)
        vocab_size = len(self.vocab_to_int) + 1  # +1 for the 0 padding
        output_size = 1
        embedding_dim = 100
        hidden_dim = 32
        n_layers = 2
        net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        self.load_model(filepath, net)
        super(PyTorchExample, self).__init__(self._pytorch_classifier)
    def extract_features(self, value: str):
        """Extract feature vector using SQLiGoT extractor.

        Arguments:
            value (str) : the input SQL query.

        Raises:
            TypeError: value is not string
            ModelNotLoadedError: calling function without having loaded or passed model as arg

        Returns:
            numpy ndarray : the feature vector
        """
        if self._pytorch_classifier is None:
            raise ModelNotLoadedError()
        type_check(value, str, "value")
        # print("Modified String", value)
        new_value = ut.PreProc(value, self.model_number, self.vocab_to_int)
        # print("pre processed value", new_value)
        return new_value

    def classify(self, value):
        """Computes the probability of being a sql injection.

        Arguments:
            value: the input query

        Raises:
            ModuleNotLoadedError: calling function without having loaded or passed model as arg
            SklearnInternalError: internal sklearn exception has been thrown

        Returns:
            probability of being a sql injection.
        """
        if self._pytorch_classifier is None:
            raise ModelNotLoadedError()
        feature_vector = self.extract_features(value)
        # print(feature_vector)
        if feature_vector is None:
            return 1

        feature_vector = torch.from_numpy(feature_vector)
        y_pred = ut.predict(self._pytorch_classifier, feature_vector)
        print(y_pred)
        return y_pred

