"""Arranque de Fooocus 2.6 en Google Colab.

La logica de arranque vive aqui y no en la celda del notebook: asi un `git pull`
corrige el arranque sin tener que volver a pegar codigo en Colab.

Dos fallos concretos que se arreglan aqui:

1. La celda antigua lanzaba Fooocus con `subprocess.run(cmd)` sin tuberia. El
   proceso hijo hereda el descriptor 1 real del kernel, que NO es el stream que
   pinta la celda de Colab (ipykernel solo sustituye sys.stdout dentro del
   proceso). Toda la salida de Fooocus acababa en el log del runtime y la celda
   se quedaba muda: lo unico visible era el repr de CompletedProcess. Aqui se
   canaliza stdout+stderr y se reimprime linea a linea.

2. Se lanza con --always-high-vram y --disable-offload-from-vram. Sin ellos
   ldm_patched usa la RAM del sistema como zona de descarga de los pesos, y en
   un T4 (15 GiB de VRAM contra ~12.7 GiB de RAM) el OOM killer mata el proceso:
   returncode=-9 sin una sola linea de traceback.
"""

import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

GIB = 1024 ** 3
PORT = 7865
WORKDIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_CACHE = '/content/drive/MyDrive/Fooocus/models'
SUBDIRS = ['checkpoints', 'loras', 'inpaint', 'controlnet', 'clip_vision',
           'upscale_models', 'vae', 'vae_approx', 'sam', 'safety_checker']
CF_DEB = ('https://github.com/cloudflare/cloudflared/releases/latest/download/'
          'cloudflared-linux-amd64.deb')
VRAM_FLAGS = ('--always-gpu', '--always-high-vram', '--always-normal-vram',
              '--always-low-vram', '--always-no-vram', '--always-cpu')
MIN_CHECKPOINT_BYTES = 1000000000


def log(message=''):
    print(message, flush=True)


def _total_ram_gib():
    try:
        import psutil
        return psutil.virtual_memory().total / GIB
    except Exception:
        return 0.0


def check_gpu():
    gpu = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
        capture_output=True, text=True).stdout.strip()
    if not gpu:
        raise RuntimeError('No hay GPU activa. Entorno de ejecucion > Cambiar tipo '
                           'de entorno > Acelerador por hardware > T4 GPU.')
    ram = _total_ram_gib()
    log('GPU: ' + gpu)
    if ram:
        log('RAM del sistema: {:.1f} GiB'.format(ram))
    if ram and ram < 20:
        log('[Memoria] RAM justa para un checkpoint SDXL. Fooocus arranca con '
            '--always-high-vram para dejar los pesos residentes en VRAM en vez de '
            'usar la RAM como zona de descarga.')
    return gpu


class RamWatchdog:
    """Vigila la RAM del sistema para poder demostrar un OOM despues del hecho."""

    def __init__(self, interval=5.0):
        self.interval = interval
        self.peak_gib = 0.0
        self._stop = threading.Event()

    def start(self):
        try:
            import psutil  # noqa: F401
        except Exception:
            return self
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        import psutil
        warned = set()
        while not self._stop.is_set():
            vm = psutil.virtual_memory()
            used = (vm.total - vm.available) / GIB
            self.peak_gib = max(self.peak_gib, used)
            for level in (85, 95):
                if vm.percent >= level and level not in warned:
                    warned.add(level)
                    log('[RAM] {:.0f}% en uso ({:.1f} de {:.1f} GiB). Si la celda '
                        'muere sin traceback a partir de aqui, es el OOM killer.'
                        .format(vm.percent, used, vm.total / GIB))
            self._stop.wait(self.interval)


def ensure_aria2c():
    found = shutil.which('aria2c')
    if found:
        log('aria2c presente: ' + found)
        return
    log('Instalando aria2...')
    subprocess.run(['apt-get', 'update', '-qq'], check=False)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'aria2'], check=True)
    log('aria2c listo: ' + str(shutil.which('aria2c')))


def link_models_to_drive():
    from google.colab import drive
    drive.mount('/content/drive')
    for sub in SUBDIRS:
        target = os.path.join(DRIVE_CACHE, sub)
        os.makedirs(target, exist_ok=True)
        local = os.path.join(WORKDIR, 'models', sub)
        if not os.path.islink(local):
            shutil.rmtree(local, ignore_errors=True)
            os.symlink(target, local)
    log('Cache de modelos enlazada a Drive.')


def download_checkpoint(model_url, model_filename):
    if not (model_url or '').strip():
        raise ValueError('Pega la URL directa del checkpoint en MODEL_URL.')
    model_dir = os.path.join(WORKDIR, 'models', 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if os.path.isfile(model_path) and os.path.getsize(model_path) > MIN_CHECKPOINT_BYTES:
        size = os.path.getsize(model_path) / GIB
        log('Modelo ya presente: {} ({:.1f} GiB)'.format(model_path, size))
        if size > 9:
            log('[Memoria] Este checkpoint es grande. En un entorno de ~12.7 GiB de '
                'RAM conviene uno fp16 de 5 a 7 GiB.')
        return model_path

    log('Descargando checkpoint con aria2c (16 conexiones)...')
    subprocess.run([
        'aria2c', '--console-log-level=notice', '--summary-interval=5',
        '--continue=true', '--allow-overwrite=true', '--auto-file-renaming=false',
        '--check-integrity=true', '--max-connection-per-server=16', '--split=16',
        '--min-split-size=1M', '--max-tries=5', '--retry-wait=3', '--timeout=60',
        '--dir', model_dir, '--out', model_filename, model_url], check=True)

    if not os.path.isfile(model_path) or os.path.getsize(model_path) <= MIN_CHECKPOINT_BYTES:
        raise RuntimeError('El checkpoint no parece completo. Revisa MODEL_URL.')
    log('Modelo listo: {} ({:.1f} GiB)'.format(
        model_path, os.path.getsize(model_path) / GIB))
    return model_path


def start_cloudflare_tunnel(timeout=60.0):
    if shutil.which('cloudflared') is None:
        log('Instalando cloudflared...')
        subprocess.run(['wget', '-q', '-O', '/content/cloudflared.deb', CF_DEB], check=False)
        subprocess.run(['dpkg', '-i', '/content/cloudflared.deb'], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if shutil.which('cloudflared') is None:
        log('cloudflared no se pudo instalar; se usara el enlace de Gradio.')
        return None, None

    proc = subprocess.Popen(
        ['cloudflared', 'tunnel', '--no-autoupdate', '--url',
         'http://127.0.0.1:{}'.format(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout],
                     daemon=True).start()

    url = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and url is None:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            continue
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            url = match.group(0)

    if url is None:
        log('Cloudflare no respondio a tiempo; se usara el enlace de Gradio.')
        proc.kill()
        return None, None

    log('')
    log('=' * 72)
    log('URL PUBLICA: ' + url)
    log('El enlace responde cuando Fooocus termine de cargar el modelo.')
    log('=' * 72)
    log('')
    return url, proc


def build_command(tunnel_url, extra_args):
    extra = shlex.split(extra_args or '')
    cmd = [sys.executable, '-u', 'entry_with_update.py', '--skip-update',
           '--preset', 'default', '--disable-preset-download',
           '--disable-analytics', '--port', str(PORT)]
    # Los flags de VRAM viven en un grupo mutuamente excluyente de argparse, asi
    # que solo se añade el nuestro si el usuario no eligio uno en ARGUMENTOS_EXTRA.
    if not any(flag in extra for flag in VRAM_FLAGS):
        cmd.append('--always-high-vram')
    if '--disable-offload-from-vram' not in extra:
        cmd.append('--disable-offload-from-vram')
    cmd += ['--listen', '127.0.0.1'] if tunnel_url else ['--share']
    return cmd + extra


def stream(cmd):
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    log('Ejecutando: ' + ' '.join(cmd))
    log('')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    return proc.wait()


def explain_exit(code, peak_gib):
    if code == 0:
        log('Fooocus termino de forma limpia.')
        return
    log('')
    log('=' * 72)
    if code == -signal.SIGKILL:
        log('Fooocus fue TERMINADO por el sistema: SIGKILL (returncode=-9).')
        log('')
        log('No es un error de Python, por eso no hay traceback: el kernel de Linux')
        log('mato el proceso al agotarse la RAM del sistema (OOM killer). La VRAM de')
        log('la GPU no tiene nada que ver; un OOM de CUDA si deja traceback.')
        if peak_gib:
            log('Pico de RAM observado antes de morir: {:.1f} GiB'.format(peak_gib))
        log('')
        log('Que hacer, en orden:')
        log(' 1. Reinicia el entorno para soltar el page cache del safetensors y')
        log('    vuelve a ejecutar la celda.')
        log(' 2. Usa un checkpoint SDXL fp16 de 5 a 7 GiB. Los fp32 o de 12 GiB no')
        log('    caben en un entorno de ~12.7 GiB de RAM.')
        log(' 3. Si se repite, pon ARGUMENTOS_EXTRA = --always-low-vram')
        log(' 4. Con Colab Pro, elige un entorno de RAM alta.')
    elif code == -signal.SIGTERM:
        log('Fooocus recibio SIGTERM (returncode=-15): normalmente el entorno de')
        log('Colab se reinicio o se detuvo la celda a mano.')
    elif code < 0:
        log('Fooocus murio por la señal {}.'.format(-code))
    else:
        log('Fooocus salio con codigo {}. El traceback esta justo arriba.'.format(code))
    log('=' * 72)


def main(model_url, model_filename='model.safetensors', tunnel='cloudflare',
         cache_in_drive=False, extra_args=''):
    os.chdir(WORKDIR)
    os.environ['PYTHONUNBUFFERED'] = '1'

    check_gpu()
    ensure_aria2c()
    if cache_in_drive:
        link_models_to_drive()
    download_checkpoint(model_url, model_filename)

    tunnel_url, tunnel_proc = None, None
    if str(tunnel).lower() == 'cloudflare':
        tunnel_url, tunnel_proc = start_cloudflare_tunnel()

    watchdog = RamWatchdog().start()
    try:
        code = stream(build_command(tunnel_url, extra_args))
    finally:
        watchdog.stop()
        if tunnel_proc is not None and tunnel_proc.poll() is None:
            tunnel_proc.kill()

    explain_exit(code, watchdog.peak_gib)
    return code
