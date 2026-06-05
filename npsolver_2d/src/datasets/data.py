import enum
import torch
from types import SimpleNamespace


class Mesh(SimpleNamespace):
    def to(self, tgt):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(tgt))
        return self


class PatchType(enum.IntEnum):
    ZERO_GRADIENT = 0
    FIXED_VALUE = 1
    EMPTY = 2
    SYMMETRY = 3
    CYCLIC = 4


class BDVertexType(enum.IntEnum):
    ZERO_GRADIENT = 0
    FIXED_VALUE = 1
    SIZE = 2


class VertexType(enum.IntEnum):
    ZERO_GRADIENT = 0
    FIXED_VALUE = 1
    INTERIOR = 2
    SIZE = 3