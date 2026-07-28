#!/usr/bin/env bash
# One-step setup. Creates a venv, installs deps, verifies the layout.
#
#   ./setup.sh          create .venv and run self-tests
#   ./setup.sh --system install into the active environment instead
set -euo pipefail

cd "$(dirname "$0")"

# Fail early with a useful message rather than a confusing ImportError later.
for f in requirements.txt piharness/__init__.py piharness/cli.py tests/test_harness.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: $f is missing. Run this from the project root, the directory" >&2
    echo "       that contains piharness/ and tests/." >&2
    exit 1
  fi
done

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

if [[ "${1:-}" == "--system" ]]; then
  echo "==> installing into the current environment"
  # Debian/Ubuntu mark the system interpreter externally managed (PEP 668) and
  # refuse plain installs. Report that clearly instead of dumping a wall of pip
  # text, and let the caller decide rather than overriding it silently.
  if ! "$PY" -m pip install -r requirements.txt 2>/tmp/piharness_pip.err; then
    if grep -q "externally-managed-environment" /tmp/piharness_pip.err; then
      cat >&2 <<'MSG'

ERROR: this Python is externally managed (PEP 668), so system-wide installs
       are blocked. Either:

         ./setup.sh                                  # use a venv (recommended)
         python3 -m pip install -r requirements.txt --break-system-packages

       The second option can interfere with system packages. The venv does not.
MSG
    else
      cat /tmp/piharness_pip.err >&2
    fi
    rm -f /tmp/piharness_pip.err
    exit 1
  fi
  rm -f /tmp/piharness_pip.err
else
  if [[ ! -d .venv ]]; then
    echo "==> creating .venv"
    # Ubuntu splits venv into a separate package; say so plainly if it's absent.
    "$PY" -m venv .venv || {
      echo "ERROR: venv creation failed. On Debian/Ubuntu:" >&2
      echo "       sudo apt install python3-venv" >&2
      exit 1
    }
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "==> installing dependencies"
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements.txt
  PY=python
fi

echo "==> running self-tests (offline, no model needed)"
"$PY" -m tests.test_harness

cat <<'EOF'

Setup complete.

  source .venv/bin/activate        # if you used the venv path

  # 1. confirm your local LLaMA answers
  python -m piharness.cli --base-url http://localhost:11434/v1 \
    --model llama3.1:8b check

  # 2. see the attack families and their matcher flags
  python -m piharness.cli list

  # 3. baseline run (clones the corpus on first use)
  python -m piharness.cli --model llama3.1:8b --outdir runs/baseline run --repeats 5

Always run from this directory: "piharness.cli" is a module path
(piharness/cli.py), not a file you invoke directly.
EOF
