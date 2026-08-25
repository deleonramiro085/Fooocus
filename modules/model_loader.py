import os
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from typing import Optional

ARIA2_CONNECTIONS = os.environ.get('FOOOCUS_ARIA2_CONNECTIONS', '16')


def aria2c_path() -> Optional[str]:
    if os.environ.get('FOOOCUS_DISABLE_ARIA2', '0') == '1':
        return None
    if not sys.platform.startswith('linux'):
        return None
    return shutil.which('aria2c')


def download_with_aria2c(binary: str, url: str, model_dir: str, file_name: str) -> bool:
    """Download with resume and parallel connections, returning success only when complete."""
    target = os.path.join(model_dir, file_name)
    partial = f'{target}.aria2'
    command = [
        binary, '--console-log-level=warn', '--summary-interval=15',
        '--continue=true', '--allow-overwrite=true', '--auto-file-renaming=false',
        '--check-integrity=true',
        f'--max-connection-per-server={ARIA2_CONNECTIONS}',
        f'--split={ARIA2_CONNECTIONS}', '--min-split-size=1M',
        '--max-tries=5', '--retry-wait=3', '--timeout=60',
        '--dir', model_dir, '--out', file_name, url,
    ]
    print(f'[Downloader] aria2c, {ARIA2_CONNECTIONS} conexiones -> {file_name}', flush=True)
    try:
        code = subprocess.run(command, check=False).returncode
    except Exception as e:
        print(f'[Downloader] aria2c no disponible ({e}); fallback a torch.hub.', flush=True)
        return False
    if code == 0 and os.path.isfile(target) and os.path.getsize(target) > 0 and not os.path.exists(partial):
        return True
    print(f'[Downloader] aria2c termino con codigo {code}; se reintenta con torch.hub.', flush=True)
    for path in (target, partial):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    return False


def load_file_from_url(url: str, *, model_dir: str, progress: bool = True,
                       file_name: Optional[str] = None) -> str:
    """Download a model, preferring resumable aria2c on Linux and falling back safely."""
    domain = os.environ.get('HF_MIRROR', 'https://huggingface.co').rstrip('/')
    url = url.replace('https://huggingface.co', domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        file_name = os.path.basename(urlparse(url).path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if os.path.isfile(cached_file) and os.path.getsize(cached_file) > 0:
        return cached_file
    print(f'Downloading: "{url}" to {cached_file}\n', flush=True)
    binary = aria2c_path()
    if binary is not None and download_with_aria2c(binary, url, os.path.dirname(cached_file),
                                                   os.path.basename(cached_file)):
        return cached_file
    from torch.hub import download_url_to_file
    download_url_to_file(url, cached_file, progress=progress)
    if not os.path.isfile(cached_file) or os.path.getsize(cached_file) == 0:
        raise RuntimeError(f'Descarga incompleta o vacia: {cached_file}')
    return cached_file
