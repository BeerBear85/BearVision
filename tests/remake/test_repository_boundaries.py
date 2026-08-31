import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def test_runtime_does_not_import_offline_or_legacy_code() -> None:
    forbidden = {"legacy", "pretraining"}
    offenders = {
        str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
        for path in (ROOT / "src" / "bearvision").rglob("*.py")
        if imported_roots(path) & forbidden
    }

    assert offenders == {}


def test_offline_yolo_does_not_import_the_legacy_application() -> None:
    offenders = []
    for path in (ROOT / "pretraining").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "code/modules" in source or '"code" / "modules"' in source:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
