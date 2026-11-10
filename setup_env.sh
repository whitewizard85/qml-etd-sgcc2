#!/usr/bin/env bash
# Creates .venv with all dependencies (Ubuntu 22.04/24.04, Python >= 3.11).
#   ./setup_env.sh          # auto: CUDA build of torch if nvidia-smi works, else CPU
#   CPU_ONLY=1 ./setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if ! $PY -c "import venv" 2>/dev/null; then
  echo "Install venv first:  sudo apt install -y python3-venv python3-dev build-essential"; exit 1
fi
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
if [ "${CPU_ONLY:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  echo ">>> NVIDIA GPU found: installing CUDA build of torch from PyPI"
  pip install torch
else
  echo ">>> CPU-only torch"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
python - <<'EOF'
import torch, qiskit, sklearn
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("qiskit", qiskit.__version__, "| sklearn", sklearn.__version__)
EOF
echo ">>> running unit tests (simulator vs Qiskit, QUBO, metrics)"
python -m pytest -q tests
