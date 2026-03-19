#!/usr/bin/env bash
# Froggy installer — one-line install:
#   curl -fsSL https://raw.githubusercontent.com/overtimepog/froggy/main/install.sh | bash
set -euo pipefail

REPO="https://github.com/overtimepog/froggy.git"
INSTALL_DIR="${FROGGY_INSTALL_DIR:-$HOME/.froggy-src}"
VENV_DIR="${INSTALL_DIR}/.venv"
MIN_PYTHON=(3 11)

# ── helpers ──────────────────────────────────────────────────────────────────

info()  { printf '\033[1;32m==> %s\033[0m\n' "$*" >&2; }
warn()  { printf '\033[1;33m==> %s\033[0m\n' "$*" >&2; }
fail()  { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

# ── find python ≥ 3.11 ──────────────────────────────────────────────────────

find_python() {
    local candidates=("python3.13" "python3.12" "python3.11" "python3" "python")
    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'  2>/dev/null)" || continue
            local major minor
            major="${ver%%.*}"
            minor="${ver#*.}"
            if (( major > MIN_PYTHON[0] || (major == MIN_PYTHON[0] && minor >= MIN_PYTHON[1]) )); then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON="$(find_python)" || fail "Python ${MIN_PYTHON[0]}.${MIN_PYTHON[1]}+ is required but not found.
Install it from https://www.python.org/downloads/ or via your package manager:
  macOS:  brew install python@3.12
  Ubuntu: sudo apt install python3.12 python3.12-venv
  Fedora: sudo dnf install python3.12"

PYTHON_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
info "Found Python ${PYTHON_VER} ($(command -v "$PYTHON"))"

# ── detect platform for extras ───────────────────────────────────────────────

detect_extras() {
    local extras="tools"
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    if [[ "$os" == "Darwin" && ("$arch" == "arm64" || "$arch" == "aarch64") ]]; then
        extras="mlx,tools"
        info "Apple Silicon detected — will install MLX backend"
    elif "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        extras="gpu,tools"
        info "CUDA detected — will install GPU/Transformers backend"
    else
        info "CPU-only install (add GPU/MLX support later with extras)"
    fi
    echo "$extras"
}

EXTRAS="$(detect_extras)"

# ── clone or update repo ─────────────────────────────────────────────────────

if [[ -d "${INSTALL_DIR}/.git" ]]; then
    info "Updating existing installation..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    info "Cloning froggy..."
    git clone "$REPO" "$INSTALL_DIR"
fi

# clean stale build artifacts that confuse setuptools
rm -rf "${INSTALL_DIR}/build" "${INSTALL_DIR}"/*.egg-info

# ── create venv & install ────────────────────────────────────────────────────

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

info "Installing froggy [${EXTRAS}]..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install "${INSTALL_DIR}[${EXTRAS}]" --quiet

# ── symlink into PATH ────────────────────────────────────────────────────────

link_binary() {
    local src="${VENV_DIR}/bin/froggy"
    local targets=("$HOME/.local/bin" "/usr/local/bin")

    # prefer a dir already on PATH
    for dir in "${targets[@]}"; do
        if echo "$PATH" | tr ':' '\n' | grep -qx "$dir"; then
            mkdir -p "$dir"
            ln -sf "$src" "${dir}/froggy"
            info "Linked froggy -> ${dir}/froggy"
            return 0
        fi
    done

    # fallback: ~/.local/bin
    mkdir -p "${targets[0]}"
    ln -sf "$src" "${targets[0]}/froggy"
    warn "Linked froggy -> ${targets[0]}/froggy"
    warn "Add ${targets[0]} to your PATH:  export PATH=\"${targets[0]}:\$PATH\""
}

link_binary

# ── done ─────────────────────────────────────────────────────────────────────

info "froggy installed successfully!"
echo ""
echo "  Get started:"
echo "    froggy --help              # see all commands"
echo "    froggy download <model>    # grab a model from HuggingFace"
echo "    froggy chat                # start chatting"
echo ""
echo "  To uninstall:"
echo "    rm -rf ${INSTALL_DIR}"
echo ""
