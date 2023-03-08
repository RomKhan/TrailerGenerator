from enum import Enum


class WorkType(Enum):
    BYLINK = 0
    BYPATH = 1


class ModelType(Enum):
    COSINUSDISTKMEANS = 0
    COSINUSDISTWITHBATCHNORM = 1
    COSINUSDIST = 2


class Status(Enum):
    CORRECT = 0
    ERROR = 1