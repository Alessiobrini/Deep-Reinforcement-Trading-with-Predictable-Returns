import hashlib
from inspect import getargvalues, stack
import json

from cytoolz import dicttoolz as dz

from .logger_mixin import LoggerMixin


class MixinCore(LoggerMixin):
    def _setattrs(self, name=None, verbose=True):
        """
        Setting attributes of itself by calling function's named
        arguments.
        """
        args = getargvalues(stack()[1][0])[-1]
        args = dz.keyfilter(lambda k: k not in ["self", "__class__"], args)

        if name is None:
            name = self.__class__.__name__

        self.logging.debug(f"Creating {name}")

        for k, v in args.items():
            self.logging.debug(f" Setting key-value pair {k}: {v}")
            setattr(self, k, v)

    def flatten(self, to_flat):
        for x in to_flat:
            if hasattr(x, "__iter__") and not isinstance(x, str):
                for y in self.flatten(x):
                    yield y
            else:
                yield str(x)

    def _save_json(self, save_to, keys=None, trial_dict=None):
        if keys:
            trial_dict = dict()
            for key in keys:
                trial_dict[key] = getattr(self, key)
        elif trial_dict is None:
            raise ValueError("Provide keys or parameters dictionary.")
        with open(save_to, "w") as outfile:
            json.dump(trial_dict, outfile)

    def _make_dict(self, keys):
        if keys:
            trial_dict = dict()
            for key in keys:
                trial_dict[key] = getattr(self, key)
        else:
            raise ValueError("Provide a list of keys")
        return trial_dict

    def _make_uuid(
        self, keys=None, truncate=False, string=None, additional=None,
    ):
        if keys:
            attributes = []

            for key in keys:
                attributes.append(getattr(self, key))
            attributes = list(self.flatten(attributes))
            unhashed = "-".join(attributes)
        elif string:
            unhashed = string
        else:
            if not additional:
                raise ValueError("Provide a list of keys or a string of values")
            else:
                unhashed = ""

        if additional:
            unhashed = "-".join([unhashed, "-".join(additional)])

        hashed = hashlib.sha256(unhashed.encode("utf-8")).hexdigest()

        if truncate:
            is_int = isinstance(truncate, int)
            is_bool = isinstance(truncate, bool)
            if is_bool or not is_int:
                raise AttributeError("truncation has to be an integer")
            hashed = hashed[:truncate]

        return hashed

    def _set_hash(self, keys, additional=None):
        model_hash = self._make_uuid(keys=keys, additional=additional, truncate=10,)
        return model_hash
