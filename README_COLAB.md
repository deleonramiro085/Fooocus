# Fooocus Colab Edition - Auditoria, arreglos y guia de mantenimiento

Version actual: **2.6.1** (ver `fooocus_version.py`).
Runtime objetivo: Colab con **Python 3.12**, numpy 2.0.2, torch 2.11 + cu128,
Gradio **3.41.2** (congelado) sobre starlette/fastapi/pydantic degradados.

## Changelog

### 2.6.1 - UI congelada al final de cada imagen (Colab, oct 2026)

**Sintoma.** Se genera la imagen, el preview se congela hacia el paso 22 de 26,
los botones no responden y F5 no lo arregla. Las imagenes SI se guardan (carpeta
`outputs/` e historial) y el proceso sigue vivo. En la consola aparece, entre la
imagen 1 y la 2:

    'AsyncRequest' object has no attribute '_json_response_data'

y al final NO aparece `Total time: N seconds`.

**Causa raiz (verificada en el codigo de gradio 3.41.2, `gradio/queueing.py` y
`gradio/utils.py`).** La cola de Gradio 3 no invoca al generador directamente:
por CADA yield (cada preview) hace un POST HTTP a `{server_path}api/predict`
con un `httpx.AsyncClient` creado sin timeout explicito, es decir con el timeout
por defecto de httpx: **5 segundos**. Fooocus tiene pausas largas sin ningun
yield (VAE, guardar imagen, `Moving model(s)`, preparar la tarea 2/2). Si el
POST espera mas de 5 s, httpx lanza `ReadTimeout`; `AsyncRequest.__run` se traga
la excepcion y deja el objeto sin `_json_response_data`; el bucle
`while response.json.get('is_generating')` de `Queue.process_events` revienta
con el AttributeError de arriba, el `print(e)` lo muestra y la cola da el evento
por terminado. El worker (otro hilo) sigue, pero nadie le envia nada al navegador.

**Arreglos (todos en `modules/compat.py::_patch_gradio_queue`, se aplican al
arrancar y lo confirman tres lineas `[Compat]` en el log):**

1. `Queue.start` sustituye `queue_client` por `httpx.AsyncClient(timeout=None)`.
   **Este es el arreglo que resuelve el fallo.**
2. `Queue.set_url` fuerza `server_path` a `http://127.0.0.1:<puerto>/`. Sin esto,
   cada paso del sampler sale a internet y vuelve por la URL publica del tunel.
3. `Queue.send_message` sube su timeout de 1 s a 30 s. Con 1 s, un websocket
   lento hacia que Gradio diera al cliente por muerto.

**Cambios complementarios:**

- `colab_run.py`: cloudflared se lanza con `--protocol http2` (el QUIC/UDP por
  defecto se degrada en Colab).
- `colab_run.py`: `FOOOCUS_PREVIEW_MAX_SIDE` por defecto 512 px (antes 768).
- `colab_run.py`: clase `HealthProbe`. Cada 10 s consulta el servidor local y
  el tunel y escribe `[Salud HH:MM:SS] LOCAL|TUNEL sin respuesta` solo cuando
  cambia el estado. `LOCAL` caido = servidor bloqueado; solo `TUNEL` caido =
  problema de red/Cloudflare.
- `fooocus_colab.ipynb`: el `git reset` usa `FETCH_HEAD` tras `git fetch --depth 1
  <repo> <rama>`. Antes usaba `origin/<rama>`, que no existe en un clon shallow de
  otra rama (error `exit status 128` al cambiar `BRANCH`).

**Como verificar que funciona.** En el log debe verse, al arrancar:

    [Compat] Cola de Gradio apunta a http://127.0.0.1:7865/ (loopback), no al tunel.
    [Compat] Cola de Gradio sin timeout de 5 s en las peticiones del generador.

y tras generar 2 imagenes debe aparecer `Total time: N seconds` y NO el
AttributeError de `_json_response_data`.

### 2.6.0 - Fallo `returncode=-9` (OOM) y celda sin logs

Ver la seccion siguiente.

## Guia de diagnostico (si Colab cambia otra vez)

Empieza por el log de la celda. Mira esto, en este orden:

| Lo que ves | Significa | Donde mirar |
|---|---|---|
| La celda muere sin traceback, `returncode=-9` | OOM killer, falta RAM del sistema | `args_manager._autotune_memory_policy`, `colab_run.explain_exit` |
| `'AsyncRequest' object has no attribute '_json_response_data'` | Una peticion de la cola de Gradio fallo (timeout/red) | `modules/compat.py::_patch_gradio_queue`; busca la excepcion real envolviendo `AsyncRequest.__run` |
| Imagenes guardadas pero falta `Total time:` | El generador `generate_clicked` de `webui.py` no termino: la UI se desconecto | igual que arriba; el `Total time` solo se imprime cuando llega el evento `finish` |
| `[Salud] LOCAL sin respuesta` | Servidor bloqueado (GIL, hilo colgado) | hilos del worker en `modules/async_worker.py` |
| Solo `[Salud] TUNEL sin respuesta` | Tunel caido; probar `TUNEL = 'gradio'` | `colab_run.start_cloudflare_tunnel` |
| `AttributeError: module 'numpy' has no attribute ...` | API retirada en numpy 2 | `_patch_numpy` |
| Errores de `torch.cuda.amp`, `weights_only`, `get_autocast_gpu_dtype` | API retirada en torch nuevo | `_patch_torch` |
| `Image.ANTIALIAS` y similares | Pillow 10+ | `_patch_pillow` |
| Error al importar gradio, starlette, pydantic, httpx | Colab actualizo el stack web | `requirements_versions.txt`, `launch.py` |

Metodo general que funciono en este caso:

1. No adivinar: leer el codigo EXACTO de la version de la libreria instalada
   (gradio 3.41.2) y seguir la ruta del error hasta su origen. Un mensaje como
   `no attribute '_json_response_data'` es un sintoma; la excepcion real estaba
   silenciada dentro de `AsyncRequest.__run`.
2. Una senal "tiene que aparecer" en el log (`Total time`) es mejor prueba de
   que algo funciona que una impresion visual de la UI.
3. Los parches viven en el repo (`compat.py`, `colab_run.py`), nunca en la celda:
   un `git pull` propaga el arreglo sin repegar codigo.
4. Probar en una rama y pasar a `main` solo tras confirmar. En Colab, para cambiar
   de rama con un clon viejo: `!rm -rf /content/Fooocus` y volver a ejecutar la celda.

Si la capa web de Gradio 3 sigue dando problemas, la salida rapida es fijar el
runtime de Colab (Entorno de ejecucion > Cambiar tipo de entorno > Runtime Version
> **2025.07**, Python 3.11 + torch 2.6), mucho mas cercano al entorno original de
Fooocus 2.5.x. La salida de fondo es migrar a Gradio 4+, pero implica reescribir
`modules/gradio_hijack.py` y buena parte de `webui.py`.

## Fallo 2.6.0: `returncode=-9`

Sintoma: la celda arranca, imprime la URL del tunel, y muere. Sin traceback.
Lo unico visible era `CompletedProcess(..., returncode=-9)`. Las imagenes si
aparecian en `outputs/`.

`-9` es **SIGKILL**: lo **mato el kernel de Linux** por quedarse sin **RAM del
sistema** (OOM killer). Un OOM de VRAM (CUDA) siempre deja traceback con
`torch.cuda.OutOfMemoryError`.

### Causa raiz

En Colab la GPU tiene mas VRAM que RAM la maquina: un T4 son 15 GiB de VRAM
contra ~12.7 GiB de RAM. La configuracion por defecto usaba la RAM como zona de
descarga:

1. `args_manager.py` derivaba `always_offload_from_vram = not disable_offload_from_vram`
   (True por defecto).
2. `model_management.py` lo lee como `ALWAYS_VRAM_OFFLOAD = True`; con
   `NORMAL_VRAM`, `unet_inital_load_device()` y `unet_offload_device()` devuelven
   cpu, y `free_memory()` descarga todos los modelos, copiando los pesos a RAM.
3. Un SDXL de ~6.5 GiB mas page cache, UNet/CLIP/VAE y el modelo de expansion
   de prompt superan los 12.7 GiB. SIGKILL.

### Arreglo

`args_manager.py` autoajusta: si la VRAM es >= 90% de la RAM y nadie eligio
politica, activa `--always-high-vram` y `--disable-offload-from-vram`. Se anuncia
por consola y se desactiva con `FOOOCUS_DISABLE_VRAM_AUTOTUNE=1`. `colab_run.py`
pasa ademas los flags, respetando cualquier flag de VRAM de `ARGUMENTOS_EXTRA`.

### Celda sin logs

La celda antigua usaba `subprocess.run(cmd)` sin tuberia; el hijo heredaba el fd 1
real del kernel, no el stream de la celda. `colab_run.py` canaliza stdout+stderr,
reimprime linea a linea, vigila la RAM (avisa al 85% y 95%) y traduce el codigo
de salida.

### Otros

- `presets/default.json`: `default_steps` no existe; la clave real es
  `default_overwrite_step` (26).

## Formato de imagen WebP vs PNG

La galeria de Gradio 3 puede fallar al pintar WebP aunque el archivo se guarde
bien. Si la miniatura no aparece, cambia **Output Format** a `png` o `jpeg` en
*Settings*.
