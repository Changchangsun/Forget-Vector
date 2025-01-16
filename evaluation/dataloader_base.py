import random
from typing import Union

import numpy as np

from unlearnbench.dataset import Dataset, get_dataset
from unlearnbench.utils import set_seed


class UnlearnLoader:
    # TODO: expand loader_kwargs
    def __init__(
        self, dataset, train_seed, data_seed, iterations=1, loader_kwargs=None
    ):
        if isinstance(dataset, str):
            # TODO: initialize dataset from string
            raise NotImplementedError("UnlearnLoader only supports Dataset object")
            dataset = get_dataset(dataset)

        self.dataset = dataset
        self.num_classes = dataset.num_classes

        def _init_fn(worker_id):
            worker_seed = self.train_seed  # + worker_id
            set_seed(worker_seed)

        if loader_kwargs is None:
            loader_kwargs = {}
        loader_kwargs["worker_init_fn"] = _init_fn
        self.loader_kwargs = loader_kwargs

        self._is_prepared = False
        self.iterations = iterations
        self.train_seed = train_seed
        self.set_data_seed(data_seed)

        self.loaders = {}

    def set_data_seed(self, data_seed):
        self.data_seed = data_seed
        self._is_prepared = False

    def set_iterations(self, iterations):
        self.iterations = iterations
        self._is_prepared = False

    def prepare_loaders(self):
        raise NotImplementedError(
            "{}.prepare_loaders() not implemented".format(type(self).__name__)
        )

    def ensure_prepared(self):
        if not self._is_prepared:
            set_seed(self.data_seed)
            self.loaders = self.prepare_loaders()
            assert isinstance(
                self.loaders, dict
            ), "prepare_loaders() must return a dict"
            self._is_prepared = True

    def get_unlearn_loader(self, name):
        self.ensure_prepared()

        assert name in self.loaders, "No loader named '{}'".format(name)
        loaders = self.loaders[name]
        return loaders

    def __getitem__(self, name):
        return self.get_unlearn_loader(name)

    def get_config(self):
        raise NotImplementedError(
            "{}.get_config() not implemented".format(type(self).__name__)
        )

    def get_name(self, full=True):
        if hasattr(self, "name"):
            return getattr(self, "name")
        else:
            if full:
                config = self.get_config()
                name_str = "_".join(
                    [self.__class__.__name__] + [f"{k}_{v}" for k, v in config.items()]
                )
                return name_str
            else:
                return self.__name__

    def get_pretrain_config(self):
        raise NotImplementedError(
            "{}.get_pretrain_config() not implemented".format(type(self).__name__)
        )

    def get_pretrain_name(self):
        config = self.get_pretrain_config()
        if "name" not in config:
            config["name"] = "vanilla"
        if "train_seed" not in config:
            config["train_seed"] = self.train_seed
        config = list(config.items())
        config.sort(key=lambda x: x[0])
        name_str = "_".join([f"{k}_{v}" for k, v in config])
        return name_str