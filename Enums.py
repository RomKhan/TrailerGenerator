from enum import Enum


class WorkType(Enum):
    BYLINK = 0
    BYPATH = 1


class ModelType(Enum):
    COSINUSDIST = 0
    COSINUSDISTWITHBATCHNORM = 1


class Status(Enum):
    CORRECT = 0
    ERROR = 1