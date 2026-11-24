"""Check the release recipe, source files, and shell scripts."""

import ast
import builtins
from importlib.util import resolve_name
import json
import os
from pathlib import Path
import re
import symtable
import unittest

from dexopd.recipe import Settings


ROOT = Path(__file__).resolve().parents[1]
GENERATED = {".git", "__pycache__", ".pytest_cache", ".venv", ".cache", "runs"}
SOURCE_PACKAGES = {"dexopd", "scorers", "scripts"}
IMPLICIT_GLOBALS = set(vars(builtins)) | {
    "__cached__", "__file__", "__loader__", "__spec__", "__package__", "__path__",
}
PUBLIC_HYGIENE = {
    "secret token": re.compile(rb"\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}\b|\bAKIA[A-Z0-9]{16}\b"),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "absolute local path": re.compile(
        rb"(?<![\w:/.-])(?:/(?:mnt|home|root|workspace)/|[C]:\\Users(?:\\|\b)|[D]:\\)", re.I
    ),
}


def repository_files(pattern):
    files = []
    for directory, subdirectories, names in os.walk(ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in GENERATED]
        files.extend(Path(directory) / name for name in names if Path(name).match(pattern))
    return sorted(files)


def symbol_scopes(table):
    yield table
    for child in table.get_children():
        yield from symbol_scopes(child)


class RepositorySanityTest(unittest.TestCase):
    def test_dexopd_settings_and_budgets(self):
        path = ROOT / "dexopd/settings.json"
        expected = {
            "base_model": "stabilityai/stable-diffusion-3.5-medium",
            "teacher_repo": "quanhaol/DiffusionOPD",
            "teachers": {
                "aesthetic": "AesTeacher/lora",
                "ocr": "OCRTeacher/lora",
                "geneval": "GenEvalTeacher/lora",
            },
            "teacher_extrapolation": 1.2,
            "auxiliary_weight": 1.8,
            "student_gains": [0.0, 0.05, 0.25, 0.5],
            "history_decay": 0.995,
            "learning_rate": 0.0003,
            "cases_per_task_per_rank": 3,
            "updates_per_round": 2,
            "updates": 1000,
            "save_every": 100,
            "seed": 42,
        }
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)
        for small, per_rank, per_update in ((False, 3, 72), (True, 1, 24)):
            with self.subTest(small_batch=small):
                cfg = Settings.load(path, small_batch=small)
                self.assertEqual(cfg.cases_per_task_per_rank, per_rank)
                self.assertEqual(cfg.budget(8), {
                    "ranks": 8,
                    "cases_per_task_per_update": 8 * per_rank,
                    "cases_per_update": per_update,
                    "cases_per_task_per_round": 16 * per_rank,
                    "cases_per_round": 2 * per_update,
                    "updates_per_round": 2,
                    "rounds": 500,
                    "updates": 1000,
                })

    def test_teacher_root(self):
        path = ROOT / "dexopd/settings.json"
        hub = Settings.load(path)
        root = Path("models/teachers")
        local = Settings.load(path, teacher_root=str(root))
        self.assertEqual(local.teacher_repo, "")
        self.assertEqual(local.teachers, {
            task: str(root / subfolder) for task, subfolder in hub.teachers.items()
        })

    def test_python_files_compile(self):
        for path in repository_files("*.py"):
            relative = path.relative_to(ROOT)
            with self.subTest(path=relative):
                compile(path.read_text(encoding="utf-8"), relative.as_posix(), "exec")

    def test_local_imports_and_globals(self):
        sources = {}
        exports = {}
        for path in repository_files("*.py"):
            relative = path.relative_to(ROOT)
            if relative.parts[0] not in SOURCE_PACKAGES:
                continue
            parts = relative.with_suffix("").parts
            package = ".".join(parts[:-1])
            module = package if path.stem == "__init__" else ".".join(parts)
            code = path.read_text(encoding="utf-8")
            table = symtable.symtable(code, relative.as_posix(), "exec")
            exports[module] = {
                symbol.get_name() for symbol in table.get_symbols()
                if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
            }
            sources[relative] = package, module, ast.parse(code, filename=str(relative)), table
        exports.setdefault("scripts", set())

        for path, (package, module, tree, table) in sources.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in SOURCE_PACKAGES:
                            with self.subTest(path=path, line=node.lineno, name=alias.name):
                                self.assertIn(alias.name, exports)
                elif isinstance(node, ast.ImportFrom):
                    imported = node.module or ""
                    if node.level:
                        imported = resolve_name("." * node.level + imported, package)
                    if imported.split(".")[0] in SOURCE_PACKAGES:
                        with self.subTest(path=path, line=node.lineno, module=imported):
                            self.assertIn(imported, exports)
                            for alias in node.names:
                                self.assertTrue(
                                    alias.name in exports[imported] or f"{imported}.{alias.name}" in exports,
                                    f"{imported}.{alias.name} is missing",
                                )
            known = IMPLICIT_GLOBALS | exports[module]
            for scope in symbol_scopes(table):
                undefined = {symbol.get_name() for symbol in scope.get_symbols()
                             if symbol.is_global() and symbol.is_referenced()} - known
                with self.subTest(path=path, scope=scope.get_name(), line=scope.get_lineno()):
                    self.assertEqual(undefined, set())

    def test_shell_files_use_lf(self):
        for path in repository_files("*.sh"):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(b"\r", path.read_bytes())

    def test_public_hygiene(self):
        for path in repository_files("*"):
            content = path.read_bytes()
            if b"\0" in content:
                continue
            for label, pattern in PUBLIC_HYGIENE.items():
                with self.subTest(path=path.relative_to(ROOT), pattern=label):
                    self.assertIsNone(pattern.search(content), f"Found {label}")


if __name__ == "__main__":
    unittest.main()
