"""Model factory; each architecture follows the supplied notebook."""

from .it import FinalITStudent
from .lighthart import LightHART
from .shared_teacher import SharedCNNTeacher
from .teacher import FinalKDTeacher

MODELS = {
    "it_student": FinalITStudent,
    "teacher": FinalKDTeacher,
    "lighthart": LightHART,
    "shared_teacher": SharedCNNTeacher,
}


def build_model(name: str, kwargs: dict | None = None):
    """Instantiate a named model without loading weights or touching data."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](**(kwargs or {}))
