"""Arranque reproducible de Fooocus en Google Colab.

Prioridad: una sesión limpia debe descargar el checkpoint, abrir la UI por
Cloudflare y mantener el proceso vivo sin depender de gradio.live.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 7865
CHECKPOINT_NAME = "juggernautXL_v8Rundiffusion.safetensors"
CHECKPOINT_URL = (
    "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/"
    "juggernautXL_v8Rundiffusion.safetensors"
)
INPAINT_NAME = "inpaint_v26.fooocus.patch"
INPAINT_URL = (
    "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/"
    "inpaint_v26.fooocus.patch"
)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check)


def require_gpu() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu = result.stdout.strip()
    if result.returncode != 0 or not gpu:
        raise RuntimeError(
            "No se detectó una GPU NVIDIA. En Colab selecciona "
            "Entorno de ejecución > Cambiar tipo de entorno > T4 GPU."
        )
    print(f"[GPU] {gpu}", flush=True)


def install_system_tools() -> None:
    if shutil.which("aria2c") is None:
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "aria2"])

    cloudflared = Path("/usr/local/bin/cloudflared")
    if not cloudflared.exists():
        run([
            "wget", "-q", "--show-progress", "-O", str(cloudflared),
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64",
        ])
        cloudflared.chmod(cloudflared.stat().st_mode | stat.S_IEXEC)


def safetensors_header_is_valid(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if not 2 <= header_size <= 100_000_000:
                return False
            header = json.loads(handle.read(header_size))
        return isinstance(header, dict) and len(header) > 10
    except Exception:
        return False


def aria2_download(url: str, destination: Path, minimum_bytes: int, *, safetensors: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    valid = destination.exists() and destination.stat().st_size >= minimum_bytes
    if valid and safetensors:
        valid = safetensors_header_is_valid(destination)
    if valid:
        print(f"[Modelo] Ya existe: {destination.name}", flush=True)
        return

    print(f"[Modelo] Descargando {destination.name}", flush=True)
    run([
        "aria2c", "--console-log-level=notice", "--summary-interval=5", "-c",
        "-x", "16", "-s", "16", "-k", "1M", "--file-allocation=none",
        "--dir", str(destination.parent), "--out", destination.name, url,
    ])

    if not destination.exists() or destination.stat().st_size < minimum_bytes:
        raise RuntimeError(f"Descarga incompleta: {destination}")
    if safetensors and not safetensors_header_is_valid(destination):
        raise RuntimeError(f"Cabecera safetensors inválida: {destination}")


def wait_for_ui(process: subprocess.Popen, timeout: int = 1_200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Fooocus terminó antes de abrir la UI (código {process.returncode}).")
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5) as response:
                    if response.status < 500:
                        print("[UI] Servidor local listo.", flush=True)
                        return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Fooocus no abrió el puerto 7865 en 20 minutos.")


def start_cloudflare() -> tuple[subprocess.Popen, str]:
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    deadline = time.monotonic() + 90
    url = ""
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line:
            print(f"[Cloudflare] {line.rstrip()}", flush=True)
            for token in line.split():
                clean = token.strip("()[]<>,.\"'")
                if clean.startswith("https://") and clean.endswith(".trycloudflare.com"):
                    url = clean
                    break
        elif process.poll() is not None:
            break
        else:
            time.sleep(0.1)
        if url:
            break
    if not url:
        process.terminate()
        raise RuntimeError("Cloudflare no entregó una URL pública.")

    def drain() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[Cloudflare] {line.rstrip()}", flush=True)

    threading.Thread(target=drain, daemon=True).start()
    return process, url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extras", action="store_true")
    parser.add_argument("--high-vram", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    require_gpu()
    install_system_tools()

    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    print(f"[Disco] {free_gb:.1f} GB libres", flush=True)
    if free_gb < 10:
        raise RuntimeError("Se necesitan al menos 10 GB libres para modelos y dependencias.")

    aria2_download(
        CHECKPOINT_URL,
        ROOT / "models" / "checkpoints" / CHECKPOINT_NAME,
        5_000_000_000,
        safetensors=True,
    )
    aria2_download(
        INPAINT_URL,
        ROOT / "models" / "inpaint" / INPAINT_NAME,
        100_000_000,
    )

    command = [
        sys.executable, "-u", "entry_with_update.py", "--skip-update",
        "--listen", "127.0.0.1", "--port", str(PORT),
        "--preset", "default", "--disable-analytics",
    ]
    if args.extras:
        command.append("--install-optional")
    if args.high_vram:
        command.append("--always-high-vram")

    print("[Fooocus] Iniciando backend...", flush=True)
    backend = subprocess.Popen(command, cwd=ROOT)
    tunnel = None
    try:
        wait_for_ui(backend)
        tunnel, public_url = start_cloudflare()
        print("\n" + "=" * 78, flush=True)
        print(f"FOOOCUS LISTO: {public_url}", flush=True)
        print("Prueba: escribe un prompt, deja 1 imagen y pulsa Generate.", flush=True)
        print("=" * 78 + "\n", flush=True)
        return backend.wait()
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
        if backend.poll() is None:
            backend.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
