#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_GPU="${INSTALL_GPU:-1}"
PROJECT_ROOT="$(pwd -P)"

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.12 not found. Set PYTHON_BIN to a valid Python 3.12 interpreter." >&2
  exit 1
fi

py_ver="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${py_ver}" != "3.12" ]]; then
  echo "ForgeAI 2.0.0 requires Python 3.12, found Python ${py_ver} at ${PYTHON_BIN}." >&2
  exit 1
fi

VENV_TARGET="${VENV_DIR}"

if [[ "${PROJECT_ROOT}" == /mnt/* ]]; then
  repo_name="$(basename "${PROJECT_ROOT}")"
  venv_name="$(basename "${VENV_DIR}")"
  sanitized_name="$(printf '%s' "${repo_name}" | tr -cs '[:alnum:]' '-')"
  sanitized_venv_name="$(printf '%s' "${venv_name}" | tr -cs '[:alnum:]' '-')"
  VENV_TARGET="${TMPDIR:-/tmp}/${sanitized_name}-${sanitized_venv_name}"

  echo "Detected a WSL-mounted checkout at ${PROJECT_ROOT}."
  echo "Creating the actual virtualenv at ${VENV_TARGET} and linking ${VENV_DIR} -> ${VENV_TARGET}."

  if [[ -e "${VENV_DIR}" && ! -L "${VENV_DIR}" ]]; then
    echo "Refusing to overwrite existing ${VENV_DIR}. Remove it or set VENV_DIR to a different path." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" -m venv "${VENV_TARGET}"

if [[ "${VENV_TARGET}" != "${VENV_DIR}" ]]; then
  rm -f "${VENV_DIR}"
  ln -s "${VENV_TARGET}" "${VENV_DIR}"
fi

source "${VENV_TARGET}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_GPU}" == "1" ]]; then
  python -m pip install --no-build-isolation -e ".[dev,gpu]"
else
  echo "============================================================"
  echo "NOTE: INSTALL_GPU=0 selected. Installing base dev environment."
  echo "Base installation is for management/unit-test workflows only."
  echo "Model inference is UNAVAILABLE without vLLM GPU dependencies."
  echo "CPU inference is unsupported (no CPU fallback exists)."
  echo "============================================================"
  python -m pip install --no-build-isolation -e ".[dev]"
fi

echo
echo "WSL bootstrap complete."
echo "Activate environment with: source ${VENV_DIR}/bin/activate"
if [[ "${VENV_TARGET}" != "${VENV_DIR}" ]]; then
  echo "Virtualenv target: ${VENV_TARGET}"
fi
echo
echo "Deferred validation commands to run manually when ready:"
echo "  python -m pytest tests -v --tb=short"
echo "  python -m ruff check src tests"
echo "  python -m mypy src"
