"""Latido de progreso para la UI de Gradio.

Sintoma: la barra llega al ultimo paso (26/26) y se queda congelada; las imagenes
si se generan y aparecen en outputs/historial minutos despues, pero la UI nunca
recibe el mensaje final. Tras el ultimo paso del sampler viene la decodificacion
VAE (en un T4 puede caer a modo tiled) y el guardado, y durante ese tiempo el
worker no manda ningun evento. `generate_clicked` solo escribe al navegador cuando
hay eventos, asi que el websocket queda inactivo y el tunel (Cloudflare o Gradio
share) lo cierra. El mensaje `finish` se pierde.

Solucion: mientras una tarea esta procesando y lleva mas de HEARTBEAT_SECONDS sin
eventos, se reinyecta el ultimo progreso conocido. generate_clicked lo reenvia al
navegador y el canal se mantiene vivo. No toca webui.py ni async_worker.py: se
engancha a AsyncTask desde fuera.

Ajustable con FOOOCUS_HEARTBEAT_SECONDS (0 lo desactiva).
"""

import os
import sys
import threading
import time
import weakref

HEARTBEAT_SECONDS = float(os.environ.get('FOOOCUS_HEARTBEAT_SECONDS', '8'))

_tasks = weakref.WeakSet()
_installed = False


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


def _loop():
    hooked = False
    while True:
        time.sleep(1.0)
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
    if _installed or HEARTBEAT_SECONDS <= 0:
        return
    _installed = True
    threading.Thread(target=_loop, name='fooocus-ui-keepalive', daemon=True).start()
    print(f'[Compat] Latido de UI cada {HEARTBEAT_SECONDS:.0f}s cuando el worker no emite eventos.',
          flush=True)
