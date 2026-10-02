# Fooocus Colab Edition 2.6 - Auditoria y arreglos

Auditoria de este fork contra el runtime **actual** de Google Colab
(octubre de 2026): **Ubuntu 24.04**, **Python 3.13**, numpy 2.1, torch 2.11,
**CUDA 13**. Colab salto de Python 3.12 a 3.13 el 25 de agosto de 2026 y de
Ubuntu 22.04 a 24.04 el 4 de septiembre.

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
celdа de Colab. Ahora `colab_run.py` canaliza stdout+stderr, añade un watchdog
de RAM que avisa al 85% y al 95% y guarda el pico, y traduce el codigo de salida
a lenguaje humano (`-9` explica el OOM y que hacer).

## Revision de octubre de 2026

Arreglos encontrados al revisar el arranque linea a linea:

- **Descargas a medias tomadas por buenas.** aria2c preasigna el archivo a su
  tamaño final antes de bajar nada. Si la descarga se cortaba (celda detenida,
  runtime reciclado), el checkpoint "pesaba" 6.6 GiB, pasaba la comprobacion de
  tamaño y Fooocus cargaba un archivo lleno de ceros. Ahora `colab_run.py` y
  `modules/model_loader.py` miran el archivo de control `.aria2` y reanudan.
- **Cambiar `MODEL_URL` no cambiaba el modelo.** Con el mismo `MODEL_FILENAME`
  se reutilizaba el archivo viejo en silencio. Ahora se guarda la URL de origen
  en `<modelo>.source` y, si cambia, se vuelve a descargar.
- **`MODEL_FILENAME` distinto de `model.safetensors` rompia el arranque.** El
  preset fija `default_model`, asi que Fooocus buscaba un archivo inexistente y
  `default_pipeline` moria con `FileNotFoundError`. `colab_run.py` pasa el nombre
  real por la variable de entorno `default_model` (config.py la lee antes que el
  preset) y `launch.py` cae al primer checkpoint disponible si aun asi no cuadra.
- **Volver a ejecutar la celda dejaba dos Fooocus.** Tras detener la celda el
  proceso anterior podia seguir vivo con el puerto 7865 y la VRAM ocupados. Ahora
  se mata al arrancar, igual que el tunel viejo, y al detener la celda se espera
  a que Fooocus cierre de verdad.
- **Barras de progreso convertidas en cientos de lineas.** En modo texto Python
  traduce cada `\r` de tqdm a salto de linea. Ahora la salida se reenvia en bruto
  y las barras se repintan en su sitio.
- **Python 3.13 elimino `audioop`**, que pydub (importado por Gradio 3) necesita.
  Se añade `audioop-lts` con marcador de version y `requirements_met()` ahora
  evalua los marcadores para no relanzar pip en cada arranque.
- **Sin `--no-build-isolation`.** Si algun paquete no tiene wheel para Python
  3.13, la compilacion necesita su propio entorno de build.
- **Notebook:** el `git reset` usa `FETCH_HEAD` (el clon es de una sola rama y
  cambiar `BRANCH` rompia la celda); casilla `EXTRAS_OPCIONALES` que el README
  ya mencionaba; diagnostico sin importar torch (no deja un contexto CUDA
  ocupando VRAM en el kernel).
- **Drive:** activar la cache ya no borra un checkpoint descargado antes en el
  disco del runtime, lo mueve. Sobre Drive aria2c no preasigna (escribir 6 GiB de
  ceros sobre FUSE tarda minutos).
- **`HF_TOKEN`:** si esta en los Secretos de Colab o en el entorno, se usa para
  descargar modelos privados o restringidos de Hugging Face.

## Nota sobre el stack web

La UI sigue en Gradio 3.41.2 (agosto 2023) sostenida por los parches de
`modules/compat.py`, y `requirements_versions.txt` degrada starlette, fastapi,
websockets, transformers, tokenizers, huggingface_hub y pydantic respecto a lo
que trae Colab (que ya viene con Gradio 6 y transformers 5). Funciona, pero es
deuda tecnica: cada actualizacion de Colab puede romper el siguiente eslabon.

Si aparece un error nuevo en la capa web y no en la generacion, la salida rapida
es fijar el runtime en Colab: Entorno de ejecucion > Cambiar tipo de entorno >
Version del runtime > **2026.07** (Python 3.12, torch 2.11, Ubuntu 22.04), el
ultimo con Python 3.12. Los runtimes antiguos solo se mantienen un año.

## Formato de imagen WebP vs PNG

La galeria de Gradio 3 puede fallar al pintar WebP aunque el archivo se guarde
bien. Si la miniatura no aparece, cambia **Output Format** a `png` o `jpeg` en
*Settings*.
