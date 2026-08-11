"""
Clean-room static regression guard for vLLM-only architecture and zero llama.cpp / GGUF runtime residue.

This module inspects repository source files, configuration manifests, packaging metadata,
and backend factory contracts using standard-library filesystem/text/AST analysis.
It enforces zero runtime imports, backend classes, deleted-module references, dependency extras,
or backend-selection support, while allowing explicit user-facing admission rejection messages.
"""

import ast
import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_deleted_modules_do_not_exist() -> None:
    """Ensure legacy llama.cpp backend wrapper and GGUF finder modules have been permanently removed."""
    deleted_paths = [
        REPO_ROOT / "src" / "forgeai" / "core" / "backends" / "llamacpp_backend.py",
        REPO_ROOT / "src" / "forgeai" / "models" / "gguf_finder.py",
    ]
    for path in deleted_paths:
        assert not path.exists(), f"Forbidden legacy module still exists: {path.relative_to(REPO_ROOT)}"


def test_python_source_no_runtime_llamacpp_symbols() -> None:
    """
    Ensure Python source under src/ contains no live imports or runtime symbol/attribute references to
    llama_cpp, LlamaCppBackend, llamacpp_backend, or gguf_finder.

    Qualified imports (e.g. 'from forgeai.models.gguf_finder import x' or 'import forgeai.core.backends.llamacpp_backend')
    and attribute access (e.g. 'obj.LlamaCppBackend') are detected and rejected.

    Intentional user-facing rejection strings (e.g. 'llama.cpp has been removed') and legacy-option
    rejection validators (e.g. 'reject_legacy_llamacpp_options') are permitted.
    """
    src_dir = REPO_ROOT / "src"
    assert src_dir.is_dir(), f"Source directory not found at {src_dir}"

    forbidden_module_tokens = {"llama_cpp", "llamacpp_backend", "gguf_finder"}
    forbidden_symbols = {"LlamaCppBackend", "llamacpp_backend", "gguf_finder", "llama_cpp"}

    violations = []

    for py_path in src_dir.rglob("*.py"):
        rel_path = py_path.relative_to(REPO_ROOT)
        content = py_path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(content, filename=str(py_path))
        except SyntaxError as e:
            violations.append(f"Syntax error in {rel_path}: {e}")
            continue

        for node in ast.walk(tree):
            # Check import statements (detect qualified module path tokens)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = set(alias.name.split("."))
                    if parts & forbidden_module_tokens:
                        violations.append(
                            f"{rel_path}:{node.lineno} - Forbidden module import: 'import {alias.name}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod_parts = set(node.module.split("."))
                    if mod_parts & forbidden_module_tokens:
                        violations.append(
                            f"{rel_path}:{node.lineno} - Forbidden module import: 'from {node.module} import ...'"
                        )
                for alias in node.names:
                    if alias.name in forbidden_symbols:
                        violations.append(
                            f"{rel_path}:{node.lineno} - Forbidden imported symbol: '{alias.name}'"
                        )

            # Check class definitions
            elif isinstance(node, ast.ClassDef):
                if node.name in forbidden_symbols:
                    violations.append(
                        f"{rel_path}:{node.lineno} - Forbidden class definition: '{node.name}'"
                    )

            # Check AST Name references for runtime symbols
            elif isinstance(node, ast.Name):
                if node.id in forbidden_symbols and node.id != "reject_legacy_llamacpp_options":
                    violations.append(
                        f"{rel_path}:{node.lineno} - Forbidden symbol reference: '{node.id}'"
                    )

            # Check AST Attribute references for runtime symbol access (e.g. module.LlamaCppBackend)
            elif isinstance(node, ast.Attribute):
                if node.attr in forbidden_symbols and node.attr != "reject_legacy_llamacpp_options":
                    violations.append(
                        f"{rel_path}:{node.lineno} - Forbidden attribute access: '.{node.attr}'"
                    )

    assert not violations, "Found forbidden runtime llama.cpp/GGUF symbols in src/:\n" + "\n".join(violations)


def _extract_package_name(req_str: str) -> str:
    """Extract and normalize package name from a PEP 508 requirement string."""
    req_no_marker = req_str.split(";")[0].strip()
    match = re.match(r"^([a-zA-Z0-9_\-\.]+)", req_no_marker)
    if match:
        return match.group(1).lower().replace("_", "-")
    return ""


def test_pyproject_toml_and_dockerfile_dependencies() -> None:
    """
    Verify pyproject.toml and Dockerfile contain no llama-cpp-python, normalized llamacpp optional extras,
    or GGUF dependency/keyword, and enforce live Dockerfile directive checks with backslash-continuation joining.
    """
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"

    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject_data.get("project", {})

    # Check main dependencies
    deps = project.get("dependencies", [])
    for dep in deps:
        dep_lower = dep.lower()
        assert "llama-cpp" not in dep_lower and "llamacpp" not in dep_lower and "gguf" not in dep_lower, (
            f"pyproject.toml main dependencies contain forbidden package: {dep}"
        )

    # Check optional-dependencies (extras) with normalized names
    opt_deps = project.get("optional-dependencies", {})
    for extra_name, extra_list in opt_deps.items():
        normalized_extra = re.sub(r"[-_]", "", extra_name.lower().strip())
        assert "llamacpp" not in normalized_extra and normalized_extra not in {"llama", "llamacpppython"}, (
            f"pyproject.toml exposes forbidden optional extra '{extra_name}'"
        )
        assert normalized_extra not in {"gpucuda", "gpurocm"}, (
            f"pyproject.toml exposes obsolete extra '{extra_name}' instead of 'gpu'"
        )
        for dep in extra_list:
            dep_lower = dep.lower()
            assert "llama-cpp" not in dep_lower and "llamacpp" not in dep_lower and "gguf" not in dep_lower, (
                f"pyproject.toml extra '{extra_name}' contains forbidden dependency: {dep}"
            )

    # Check keywords
    keywords = project.get("keywords", [])
    for kw in keywords:
        kw_lower = kw.lower()
        assert kw_lower != "gguf" and kw_lower != "llama-cpp" and kw_lower != "llamacpp", (
            f"pyproject.toml keywords contain forbidden entry: '{kw}'"
        )

    # Check Dockerfile live directives after joining backslash-continuation lines
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile not found"
    dockerfile_text = dockerfile_path.read_text(encoding="utf-8")

    # Join backslash-continuation lines into logical instructions
    raw_lines = dockerfile_text.splitlines()
    instructions = []
    current_lines = []
    start_line_num = 1

    for idx, raw_line in enumerate(raw_lines, 1):
        line_stripped = raw_line.strip()

        if not current_lines and (not line_stripped or line_stripped.startswith("#")):
            continue

        if not current_lines:
            start_line_num = idx

        if line_stripped.startswith("#"):
            continue

        if line_stripped.endswith("\\"):
            current_lines.append(line_stripped[:-1].rstrip())
        else:
            current_lines.append(line_stripped)
            combined = " ".join(current_lines).strip()
            instructions.append((start_line_num, combined))
            current_lines = []

    if current_lines:
        combined = " ".join(current_lines).strip()
        instructions.append((start_line_num, combined))

    for line_num, instruction in instructions:
        inst_upper = instruction.upper()
        target_keywords = ("ARG ", "RUN ", "ENV ", "COPY ", "ADD ")
        if any(inst_upper.startswith(kw) for kw in target_keywords):
            inst_lower = instruction.lower()
            if inst_upper.startswith("ARG "):
                assert "backend=llamacpp" not in inst_lower and "backend=llama-cpp" not in inst_lower and "backend=llama_cpp" not in inst_lower and "backend=llama.cpp" not in inst_lower and "backend=llama" not in inst_lower, (
                    f"Dockerfile:{line_num} contains forbidden build argument: '{instruction}'"
                )

            forbidden_tokens = [
                "llama-cpp-python",
                "llama_cpp",
                "llama-cpp",
                "llama.cpp",
                "llamacpp",
                "gguf",
            ]
            for token in forbidden_tokens:
                assert token not in inst_lower, (
                    f"Dockerfile:{line_num} live instruction contains forbidden '{token}' reference: '{instruction}'"
                )


def test_pyproject_toml_python_version_and_vllm_pinning() -> None:
    """
    Verify pyproject.toml requires Python >=3.12,<3.13 and requires every vLLM dependency
    across all optional extras to be strictly 'vllm==0.22.1' with no extra markers or specifier variants,
    and requires implemented 'gpu', 'vllm', and 'all' extras each to contain exactly one such pin.
    """
    pyproject_path = REPO_ROOT / "pyproject.toml"
    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    project = pyproject_data.get("project", {})
    req_python = project.get("requires-python", "")
    assert req_python == ">=3.12,<3.13", f"requires-python in pyproject.toml is '{req_python}', expected '>=3.12,<3.13'"

    opt_deps = project.get("optional-dependencies", {})
    required_vllm_extras = ["gpu", "vllm", "all"]

    # 1. Require gpu, vllm, and all extras to exist and contain exactly one 'vllm==0.22.1' pin
    for extra in required_vllm_extras:
        assert extra in opt_deps, f"Expected optional dependency extra '{extra}' in pyproject.toml"
        extra_deps = opt_deps[extra]
        vllm_reqs = [dep for dep in extra_deps if _extract_package_name(dep) == "vllm"]
        assert len(vllm_reqs) == 1, (
            f"Extra '{extra}' must contain exactly one vLLM requirement pin, found {len(vllm_reqs)}: {vllm_reqs}"
        )
        assert vllm_reqs[0].strip() == "vllm==0.22.1", (
            f"Extra '{extra}' vLLM requirement must be exactly 'vllm==0.22.1', found {vllm_reqs[0]!r}"
        )

    # 2. Require ALL vLLM requirements across ALL optional-dependency extras to be strictly 'vllm==0.22.1'
    for extra_name, dep_list in opt_deps.items():
        for dep in dep_list:
            if _extract_package_name(dep) == "vllm":
                assert dep.strip() == "vllm==0.22.1", (
                    f"Extra '{extra_name}' has invalid vLLM requirement specifier: {dep!r}. "
                    "Every vLLM requirement across all optional dependencies must be exactly 'vllm==0.22.1'."
                )


def test_backend_factory_and_config_no_auto_or_llama_selection() -> None:
    """
    Verify backend factory and config do not expose auto or llama.cpp backend selection,
    while permitting explicit legacy-input rejection error handling.
    """
    config_path = REPO_ROOT / "src" / "forgeai" / "core" / "config.py"
    factory_path = REPO_ROOT / "src" / "forgeai" / "core" / "backends" / "factory.py"

    assert config_path.exists(), "config.py not found"
    assert factory_path.exists(), "factory.py not found"

    config_tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))

    # Check BackendType enum in config.py
    backend_enum_values = []
    for node in ast.walk(config_tree):
        if isinstance(node, ast.ClassDef) and node.name == "BackendType":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.value, ast.Constant):
                        backend_enum_values.append(stmt.value.value)

    assert backend_enum_values == ["vllm"], (
        f"BackendType Enum in config.py must contain only ['vllm'], found {backend_enum_values}"
    )

    # Check resolve_backend and create_backend in factory.py
    factory_content = factory_path.read_text(encoding="utf-8")
    assert "BackendType.VLLM" in factory_content, "factory.py must resolve to BackendType.VLLM"
    assert "VLLMBackend" in factory_content, "factory.py must instantiate VLLMBackend"
    assert "LlamaCppBackend" not in factory_content, "factory.py must not reference LlamaCppBackend"

    # Verify legacy rejection validator or error raising is present
    assert "GGUF model format is unsupported" in factory_content or "GGUF model format" in factory_content, (
        "factory.py should retain explicit actionable GGUF rejection message"
    )
