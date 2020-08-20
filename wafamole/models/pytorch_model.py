"""Wrapper for sci-kit learn classifiers.
"""
import os
import joblib
import numpy as np
from wafamole.models import Model
from wafamole.exceptions.models_exceptions import (
    NotPyTorchModelError,
    PyTorchInternalError,
    ModelNotLoadedError,
)
import torch
from wafamole.utils.check import type_check, file_exists

class PyTorchModelWrapper(Model):
    """Sci-kit learn classifier wrapper class"""

    def __init__(self, pytorch_classifier=None):
        """Constructs a wrapper around an scikit-learn classifier, or equivalent.
        It must implement predict_proba function.

        Arguments:
            pytorch_classifier :  pytorch-learn classifier or equivalent

        Raises:
            NotPyTorchModelError: not implement predict_proba
            NotPyTorchModelError: not implement fit
        """
        if pytorch_classifier is None:
            self._pytorch_classifier = None
        else:
            # if getattr(pytorch_classifier, "predict_proba", None) is None:
            #     raise NotPyTorchModelError(
            #         "object does not implement predict_proba function"
            #     )

            self._pytorch_classifier = pytorch_classifier

    def classify(self, value):
        """It returns the probability of belonging to a particular class.
        It calls the extract_features function on the input value to produce a feature vector.

        Arguments:
            value (numpy ndarray) : an input belonging to the input space of the model

        Raises:
            ModelNotLoadedError: calling function without having loaded or passed model as arg

        Returns:
            numpy ndarray : the confidence for each class of the problem.

        """
        if self._pytorch_classifier is None:
            raise ModelNotLoadedError()
        feature_vector = self.extract_features(value)
        try:
            y_pred = self._pytorch_classifier([feature_vector])
            return y_pred
        except Exception as e:
            raise PyTorchInternalError("Internal PyTorch error.") from e

    def load_model(self, filepath, ModelClass):
        """Loads a PyTorch classifier stored in filepath.

        Arguments:
            filepath (string) : The path of the PyTorch classifier.

        Raises:
            TypeError: filepath is not string.
            FileNotFoundError: filepath not pointing to any file.
            NotPyTorchModelError: model can not be loaded.

        Returns:
            self
        """
        type_check(filepath, str, "filepath")
        file_exists(filepath)
        ModelClass.load_state_dict(torch.load(filepath))
        ModelClass.eval()
        self._pytorch_classifier = ModelClass
        return self

    def extract_features(self, value: np.ndarray):
        """It returns the input. To modify this behaviour, extend this class and re-define this method.

        Arguments:
            value (numpy ndarray) : a sample that belongs to the input space of the model

        Returns:
            the input.
        """
        if type(value) != np.ndarray:
            raise TypeError(f"{type(value)} not an nd array")
        return value
