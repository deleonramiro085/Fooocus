"""Estabilidad de la UI de Gradio detras de un tunel (Colab).

Dos arreglos independientes:

1. Cola de Gradio por loopback.
   Gradio 3.x ejecuta cada paso de un generador con una peticion HTTP a su propio
   servidor (`{server_path}api/predict`, via AsyncRequest). `server_path` se toma de
   la URL del primer websocket, o sea la URL PUBLICA del tunel. Cada paso sale a
   internet, pasa por Cloudflare y vuelve. Cuando esa peticion falla, Gradio deja
   la excepcion guardada y despues revienta con
   "'AsyncRequest' object has no attribute '_json_response_data'", y el generador
   que alimenta la UI se queda muerto: las imagenes se siguen generando en el
   worker, pero la UI ya no recibe nada. Aqui se fuerza `server_path` a
   http://127.0.0.1:PUERTO/, que no depende del tunel.

2. Latido de progreso.
   Tras el ultimo paso vienen la decodificacion VAE y el guardado (que incluye
   calcular el sha256 del checkpoint la primera vez, ~50 s), sin eventos del worker.
   Si una tarea lleva mas de HEARTBEAT_SECONDS sin eventos se reinyecta el ultimo
   progreso para que el canal no quede inactivo.

Ajustable con FOOOCUS_HEARTBEAT_SECONDS (0 desactiva el latido) y
FOOOCUS_QUEUE_LOOPBACK=0 (desactiva el arreglo 1).
"""

import os
import sys
import threading
import time
import weakref

HEARTBEAT_SECONDS = float(os.environ.get('FOOOCUS_HEARTBEAT_SECONDS', '8'))
QUEUE_LOOPBACK = os.environ.get('FOOOCUS_QUEUE_LOOPBACK', '1') != '0'

_tasks = weakref.WeakSet()
_installed = False


def _server_port():
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == '--port' and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                break
        if arg.startswith('--port='):
            try:
                return int(arg.split('=', 1)[1])
            except ValueError:
                break
    return 7865


class _TrackedList(list):
    """Lista de yields que recuerda cuando se escribio y cual fue el ultimo preview."""

    def __init__(self, *args):
        super().__init__(*args)
        self.last_append = time.monotonic()
        self.last_preview = None
        self.beats = 0

    def append(self, item):
        self.last_append = time.monotonic()
        try:
            if item[0] == 'preview':
                self.last_preview = item[1]
        except Exception:
            pass
        super().append(item)


def _hook_async_task(module):
    cls = module.AsyncTask
    if getattr(cls, '_fooocus_keepalive', False):
        return
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.yields = _TrackedList(self.yields)
        _tasks.add(self)

    cls.__init__ = __init__
    cls._fooocus_keepalive = True


def _patch_queue(module):
    queue_cls = getattr(module, 'Queue', None)
    if queue_cls is None:
        print('[Compat] gradio.queueing.Queue no encontrada; se omite el arreglo de loopback.',
              flush=True)
        return True
    if getattr(queue_cls, '_fooocus_loopback', False):
        return True
    if not hasattr(queue_cls, 'set_url'):
        print('[Compat] Queue.set_url no existe en esta version de Gradio; se omite el arreglo '
              'de loopback.', flush=True)
        return True

    local_url = f'http://127.0.0.1:{_server_port()}/'

    def set_url(self, url):
        self.server_path = local_url

    queue_cls.set_url = set_url
    queue_cls._fooocus_loopback = True
    print(f'[Compat] La cola de Gradio llamara a {local_url} en vez de a la URL del tunel.',
          flush=True)
    return True


def _loop():
    hooked = False
    queue_patched = not QUEUE_LOOPBACK
    while True:
        time.sleep(0.5 if not queue_patched else 1.0)

        if not queue_patched:
            module = sys.modules.get('gradio.queueing')
            if module is not None:
                try:
                    queue_patched = _patch_queue(module)
                except Exception as e:
                    print(f'[Compat] No se pudo aplicar el loopback de la cola: {e}', flush=True)
                    queue_patched = True

        if HEARTBEAT_SECONDS <= 0:
            if queue_patched:
                return
            continue

        module = sys.modules.get('modules.async_worker')
        if module is None or not hasattr(module, 'AsyncTask'):
            continue
        if not hooked:
            _hook_async_task(module)
            hooked = True
        now = time.monotonic()
        for task in list(_tasks):
            try:
                y = task.yields
                if not isinstance(y, _TrackedList) or not getattr(task, 'processing', False):
                    continue
                if len(y) > 0 or y.last_preview is None:
                    continue
                idle = now - y.last_append
                if idle < HEARTBEAT_SECONDS:
                    continue
                percentage, title, _ = y.last_preview
                y.beats += 1
                print(f'[Heartbeat] {idle:.0f}s sin eventos del worker, reenviando progreso: {title}',
                      flush=True)
                y.append(['preview', (percentage, f'{title} (sigue trabajando, {idle:.0f}s)', None)])
            except Exception as e:
                print(f'[Heartbeat] error: {e}', flush=True)


def install():
    global _installed
    if _installed:
        return
    _installed = True
    threading.Thread(target=_loop, name='fooocus-ui-keepalive', daemon=True).start()
    if HEARTBEAT_SECONDS > 0:
        print(f'[Compat] Latido de UI cada {HEARTBEAT_SECONDS:.0f}s cuando el worker no emite eventos.',
              flush=True)
