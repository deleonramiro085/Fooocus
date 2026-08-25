# Fooocus 2.6 - Colab Edition

Auditoria y arreglo de este fork para que funcione en el runtime **actual** de Google Colab.

## El diagnostico corto

El repo estaba clavado en el mundo de 2024 (Python 3.10, numpy 1.26, torch 2.1) y el
runtime de Colab hoy es **Python 3.13 + torch >= 2.6 + numpy 2.x**. Casi ningun pin del
`requirements_versions.txt` tenia wheels para 3.13, asi que la instalacion moria antes de
empezar. Y aunque hubiera arrancado, el notebook clonaba **otro** repositorio.

## Los 10 fallos encontrados

| # | Fallo | Efecto | Arreglo |
|---|-------|--------|---------|
| 1 | `fooocus_colab.ipynb` clonaba `lllyasviel/Fooocus` | **ningun cambio de este fork se usaba jamas** (ni el preset ni los parches) | clona este repo y esta rama |
| 2 | `pip install pygit2==1.15.1` en el notebook | no hay wheel para 3.13 y compilarlo pide libgit2-dev: fallo inmediato | eliminado (Colab ya trae pygit2 1.20) |
| 3 | `torch==2.1.0 + cu121` forzado en `launch.py` | no existe para Python 3.13, y reinstalar torch rompe la CUDA del runtime | se usa el torch preinstalado; solo se instala si falta |
| 4 | `numpy==1.26.4` | ultima version soportada es 3.12: pip intenta compilar y muere | `numpy>=1.26,<3` + shims de numpy 2 |
| 5 | `pydantic==1.10.17` | gradio 3.41 **exige pydantic >= 2**: ese pin lo rompia, no lo arreglaba | `pydantic>=2.9,<2.10` |
| 6 | `cupy-cuda12x` | 500 MB que Fooocus no importa en ninguna linea | borrado |
| 7 | `gradio==3.41.2` dentro del `-r requirements` | su metadata pide `numpy~=1.0` y `pillow<11`: el resolver arrastraba numpy 1.x y reventaba | se instala con `--no-deps` y sus dependencias reales van listadas a mano |
| 8 | `presets/default.json` con `checkpoint_downloads: {}` | el modelo por defecto (cyberrealisticPony) nunca se descargaba: UI sin ningun checkpoint | mirror verificado en HF + aviso explicito si no hay checkpoints |
| 9 | `extras/inpaint_mask.py` importaba rembg / segment_anything / groundingdino arriba del modulo | `async_worker` lo importa al arrancar, asi que una dependencia opcional que faltara **tumbaba todo Fooocus** | imports perezosos con mensaje util |
| 10 | `entry_with_update.py` usaba las constantes `pygit2.GIT_*` | eliminadas en pygit2 >= 1.15: el update fallaba y encima imprimia "Update succeeded" | API `pygit2.enums` con fallback + `--skip-update` |

## Pines que parecen arbitrarios y no lo son

Estos cuatro son los que sostienen todo el stack web. Cambiarlos rompe la UI:

- **`starlette>=0.37.2,<0.38`**: es la ultima rama que acepta `TemplateResponse(name, context)`,
  la firma que usa gradio 3 y que Fooocus parchea en `modules/ui_gradio_extensions.py`.
  Colab trae starlette 1.6, donde eso ya lanza `TypeError` (pantalla en blanco).
- **`websockets>=13,<15`**: `gradio_client 0.5.0` hace `from websockets.legacy.protocol import ...`
  en la primera linea. websockets 15 elimino `legacy`, asi que `import gradio` fallaria.
- **`huggingface_hub<1.0`**: `gradio_client 0.5.0` importa `SpaceStage`, que ya no existe en la 1.x.
- **`python-multipart==0.0.9`**: es la ultima que expone el modulo `multipart` que importa
  starlette 0.37. Sin el, todas las subidas de imagen de la UI dan error.

Y uno mas: **`transformers>=4.49,<4.50`** es la ultima rama que conserva las APIs internas que
`modules/patch_clip.py` parchea (`modeling_utils.no_init_weights`, `CLIPTextModel`,
`CLIPVisionModelWithProjection`). Con transformers 5.x el encoder de texto no carga.

## Mejoras que no pediste pero te vas a agradecer

- **Cache de modelos en Google Drive** (casilla del notebook): los 9 GB se descargan una vez,
  no en cada sesion.
- **Reporte de entorno al arrancar**: python, torch, CUDA, GPU y las versiones criticas.
  Si algo falla, se ve en la primera pantalla y no hay que adivinar.
- **Descargas tolerantes a fallos**: un mirror caido ya no aborta el arranque.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: menos fragmentacion de VRAM en T4.
- **`requirements_optional.txt`**: rembg / SAM / GroundingDINO fuera del camino critico.
  GroundingDINO compila una extension CUDA al instalarse; que falle ya no es fatal.
- **`requirements_met()` reescrito**: entiende rangos, no solo `==`, y dice exactamente
  que falta en vez de reinstalar todo a ciegas.

## Como se usa

Abre `fooocus_colab.ipynb` en Colab, pon GPU y ejecuta la primera celda. Nada mas.

Para volver a la version anterior: `git checkout main`.
