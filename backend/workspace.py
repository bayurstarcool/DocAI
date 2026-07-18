"""Safe access helpers for the image-restoration test workspace."""
from pathlib import Path


class WorkspaceError(ValueError):
    pass


def resolve_workspace_path(root: Path, relative_path: str) -> Path:
    """Return an existing workspace path, rejecting traversal and absolute paths."""
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise WorkspaceError("Path is outside workspace")
    if not candidate.exists():
        raise WorkspaceError("Path does not exist")
    return candidate


def safe_workspace_name(value: str) -> str:
    name = Path(value).name.strip()
    if not name or name in {".", ".."}:
        raise WorkspaceError("Invalid test folder name")
    return name
