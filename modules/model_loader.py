import os
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from typing import Optional

# Numero de conexiones paralelas de aria2c. 16 es el punto donde Hugging Face deja de
# escalar; subirlo mas no acelera y aumenta el riesgo de que corten la conexion.
ARIA2_CONNECTIONS = os.environ.get('FOOOCUS_ARIA2_CONNECTIONS', '16')


def aria2c_path() -> Optional[str]:
    """Ruta a aria2c si conviene usarlo, o None.

    El descargador de torch.hub usa una sola conexion HTTP y Hugging Face lo limita
    con fuerza: en Colab se han medido ~70 kB/s (26 horas para un checkpoint de 6.6 GB)
    y cortes a mitad de descarga que dejan el archivo inservible. aria2c con varias
    conexiones baja el mismo archivo a 150+ MB/s y reanuda si se corta.

    Se puede desactivar con FOOOCUS_DISABLE_ARIA2=1.
    """
    if os.environ.get('FOOOCUS_DISABLE_ARIA2', '0') == '1':
        return None
    if not sys.platform.startswith('linux'):
        return None
    return shutil.which('aria2c')


def download_with_aria2c(binary: str, url: str, model_dir: str, file_name: str) -> bool:
    """Descarga multi-hilo. True si el archivo quedo completo."""
    target = os.path.join(model_dir, file_name)
    partial = f'{target}.aria2'

    command = [
        binary,
        '--console-log-level=warn',
        '--summary-interval=15',
        '--continue=true',
        '--allow-overwrite=true',
        '--auto-file-renaming=false',
        f'--max-connection-per-server={ARIA2_CONNECTIONS}',
        f'--split={ARIA2_CONNECTIONS}',
        '--min-split-size=1M',
        '--max-tries=5',
        '--retry-wait=3',
        '--timeout=60',
        '--dir', model_dir,
        '--out', file_name,
        url,
    ]

    print(f'[Downloader] aria2c, {ARIA2_CONNECTIONS} conexiones -> {file_name}')
    try:
        code = subprocess.run(command, check=False).returncode
    except Exception as e:
        print(f'[Downloader] No se pudo ejecutar aria2c ({e}). Se usa torch.hub.')
        return False

    if code == 0 and os.path.exists(target) and not os.path.exists(partial):
        return True

    print(f'[Downloader] aria2c termino con codigo {code}. Se reintenta con torch.hub.')
    # Un archivo a medias haria que Fooocus lo diera por descargado en el siguiente
    # arranque, asi que se limpia antes del fallback.
    for path in (target, partial):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return False


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        binary = aria2c_path()
        if binary is None or not download_with_aria2c(
                binary, url, os.path.dirname(cached_file), os.path.basename(cached_file)):
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
    return cached_file
