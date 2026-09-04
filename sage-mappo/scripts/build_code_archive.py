"""Build a code-only archive; never copy outputs, weights, or workstation files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"README.md", "LICENSE", "ANONYMIZATION_REPORT.md", "tree.txt", "requirements.txt",
             "requirements-llm.txt", "requirements-vmas.txt", "environment.yml", "pyproject.toml",
             ".gitignore", ".gitattributes"}
SOURCE_DIRS = {"src", "scripts", "configs", "docs", "tests", ".github"}
SOURCE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".toml", ".sh", ".ps1"}


def code_files():
    selected = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(ROOT)
        if "__pycache__" in rel.parts or any(part.endswith(".egg-info") for part in rel.parts):
            continue
        include = str(rel) in TOP_FILES
        include |= rel.parts[0] in SOURCE_DIRS and path.suffix in SOURCE_SUFFIXES
        include |= rel.as_posix() in {"data/README.md", "data/models/README.md", "outputs/README.md"}
        if include:
            if path.stat().st_size > 2_000_000:
                raise ValueError(f"Unexpectedly large code file: {rel}")
            selected.append(path)
    return selected


def main():
    files = code_files()
    (ROOT / "tree.txt").write_text("anonymous-hippo-mappo/\n" + "\n".join(
        "  " + p.relative_to(ROOT).as_posix() for p in files if p.name != "tree.txt"
    ) + "\n", encoding="utf-8")
    files = code_files()
    dest = ROOT / "dist"
    dest.mkdir(exist_ok=True)
    archive = dest / "anonymous-hippo-mappo-code.zip"
    hashes = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            bundle.write(path, "anonymous-hippo-mappo/" + name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (dest / "file_checksums.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    print(f"Archive: {archive}")
    print(f"Files: {len(files)}; bytes: {archive.stat().st_size}")
    print(f"SHA-256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
