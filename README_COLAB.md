# Fooocus Colab Edition 2.6 - Auditoria y arreglos

Auditoria de este fork contra el runtime **actual** de Google Colab
(Ubuntu 22.04, **Python 3.12**, numpy 2.0.2, torch 2.11 + cu128).

## El fallo principal: `returncode=-9`

Sintoma: la celda arranca, imprime la URL del tunel, y muere. Sin traceback.
Lo unico visible es `CompletedProcess(..., returncode=-9)`. Las imagenes si
aparecen en `outputs/` y en el historial.

`-9` es **SIGKILL**: el proceso no fallo, lo **mato el kernel de Linux** por
quedarse sin **RAM del sistema** (OOM killer). Esto es clave: un OOM de VRAM
(CUDA) siempre deja traceback con `torch.cuda.OutOfMemoryError`. Aqui no hay
ninguno, luego el problema nunca estuvo en la GPU.

### Causa raiz

En Colab la GPU tiene **mas VRAM que RAM tiene la maquina**: un T4 son 15 GiB
de VRAM contra ~12.7 GiB de RAM. Y la configuracion por defecto usaba la RAM
como zona de descarga de la VRAM:

1. `args_manager.py` derivaba
   `always_offload_from_vram = not disable_offload_from_vram`, es decir **True**
   salvo que se pase el flag.
2. `model_management.py` lo lee como `ALWAYS_VRAM_OFFLOAD = True`, y al no pasar
   ninguna politica de VRAM el estado se queda en `NORMAL_VRAM`. Con esa
   combinacion:
   - `unet_inital_load_device()` devuelve **cpu** de forma incondicional,
   - `unet_offload_device()` devuelve **cpu**,
   - `free_memory()` pierde su corto circuito
     (`if get_free_memory(device) > memory_required: break`), asi que descarga
     **todos** los modelos y `unpatch_model()` copia los pesos de vuelta a RAM
     tras cada pasada.
3. Un checkpoint SDXL de ~6.5 GiB, mas su page cache, mas UNet/CLIP/VAE
   materializados en RAM, mas el modelo de expansion de prompt, se pasan de los
   12.7 GiB. SIGKILL.

Esto explica el patron exacto: antes sobrevivia la primera carga y se congelaba
sobre el paso 30 (presion de RAM justo cuando toca descargar), un reinicio
soltaba el page cache y funcionaba una vez.

### Arreglo

`args_manager.py` autoajusta la politica de memoria: si la VRAM es >= 90% de la
RAM del sistema y nadie eligio politica a mano, activa `--always-high-vram` y
`--disable-offload-from-vram`, de modo que los pesos se quedan **residentes en
VRAM** en vez de ir y volver a RAM. Se anuncia por consola. Se desactiva con
`FOOOCUS_DISABLE_VRAM_AUTOTUNE=1`.

`colab_run.py` pasa ademas los flags de forma explicita, respetando cualquier
flag de VRAM que se ponga en `ARGUMENTOS_EXTRA` (son un grupo mutuamente
excluyente de argparse).

## El segundo fallo: la celda no mostraba logs

La celda antigua lanzaba Fooocus con `subprocess.run(cmd)` sin tuberia. El hijo
hereda el **descriptor 1 real** del kernel, que no es el stream que pinta la
celda de Colab (ipykernel solo sustituye `sys.stdout` dentro del proceso). Toda
la salida de Fooocus acababa en el log del runtime, invisible: de ahi que la
unica linea fuera el `repr` de `CompletedProcess`. Se estaba depurando a ciegas.

Ahora `colab_run.py` canaliza stdout+stderr y reimprime linea a linea, añade un
watchdog de RAM que avisa al 85% y al 95% y guarda el pico, y traduce el codigo
de salida a lenguaje humano (`-9` explica el OOM y que hacer).

## Otros arreglos

- `presets/default.json` traia `default_steps`, que **no existe** como clave de
  configuracion: `modules/config.py` nunca la lee, por eso seguian saliendo 30
  pasos. La clave real es `default_overwrite_step`, ya corregida a 26.
- La celda se reduce a actualizar el repo y llamar a `colab_run.main()`, para que
  los proximos arreglos lleguen por `git pull` sin repegar codigo en Colab.

## Nota sobre el stack web

La UI sigue en Gradio 3.41.2 (agosto 2023) sostenida por los parches de
`modules/compat.py`, y `requirements_versions.txt` degrada starlette, fastapi,
websockets, transformers, tokenizers y pydantic respecto a lo que trae Colab.
Funciona, pero es deuda tecnica: cada actualizacion de Colab puede romper el
siguiente eslabon. Si algun dia aparece un error nuevo en la capa web y no en la
generacion, la salida rapida es fijar el runtime en Colab
(Entorno de ejecucion > Cambiar tipo de entorno > Runtime Version > **2025.07**,
Python 3.11 + torch 2.6), mucho mas cercano al entorno original de Fooocus 2.5.x.

## Formato de imagen WebP vs PNG

La galeria de Gradio 3 puede fallar al pintar WebP aunque el archivo se guarde
bien. Si la miniatura no aparece, cambia **Output Format** a `png` o `jpeg` en
*Settings*.
