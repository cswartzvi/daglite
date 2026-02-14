from typing import Any, Callable, Coroutine

from typing_extensions import Literal

Submission = Callable[[], Coroutine[Any, Any, Any]]

MapMode = Literal["product", "zip"]
ParamKind = Literal["value", "ref", "sequence", "sequence_ref"]
NodeKind = Literal["task", "map", "dataset"]
