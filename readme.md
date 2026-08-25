<div align="center">

# Fooocus Colab Edition 2.6

### Fooocus actualizado para Google Colab moderno

[![Versión](https://img.shields.io/badge/versión-2.6.0-7c3aed)](./fooocus_version.py)
[![Google Colab](https://img.shields.io/badge/Google_Colab-T4-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/deleonramiro085/Fooocus/blob/main/fooocus_colab.ipynb)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/licencia-GPL--3.0-blue)](./LICENSE)

**[Abrir Fooocus 2.6 en Google Colab](https://colab.research.google.com/github/deleonramiro085/Fooocus/blob/main/fooocus_colab.ipynb)**

</div>

## ¿Qué es esta edición?

**Fooocus Colab Edition 2.6** es un fork enfocado en recuperar una experiencia confiable de Fooocus sobre los runtimes actuales de Google Colab. Conserva la interfaz sencilla y las herramientas SDXL de Fooocus, pero moderniza el arranque, las dependencias, las descargas y el acceso remoto.

El objetivo es simple: ejecutar una celda, ver qué está ocurriendo y obtener una URL pública funcional, sin instalaciones silenciosas ni esperas que parezcan bloqueos.

> Esta es una edición comunitaria mantenida por [deleonramiro085](https://github.com/deleonramiro085). El proyecto original pertenece a [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus).

## Ventajas de la versión 2.6

- **Compatible con el Colab actual:** adapta Fooocus 2.5.x a Python 3.13, NumPy 2 y versiones modernas de PyTorch con CUDA.
- **Aprovecha el entorno GPU existente:** evita reinstalar Torch y romper la combinación de CUDA preparada por Colab.
- **Descargas rápidas y reanudables:** usa `aria2c` con hasta 16 conexiones, progreso periódico, reintentos y continuación de archivos incompletos.
- **Sin falsa sensación de bloqueo:** Python, `pip` y los procesos de arranque muestran salida continua sin búfer.
- **Modelo bajo control del usuario:** el notebook solicita una URL directa y valida que el checkpoint descargado tenga un tamaño razonable.
- **Túnel más estable:** utiliza Cloudflare Tunnel por defecto y recurre a Gradio Share cuando Cloudflare no responde.
- **Caché opcional en Google Drive:** permite conservar modelos entre sesiones y evita descargar varios gigabytes cada vez.
- **Actualización segura:** reconoce versiones modernas de `pygit2`, evita actualizaciones ambiguas y ofrece `--skip-update`.
- **Arranque más resistente:** las funciones opcionales ya no impiden abrir Fooocus cuando faltan paquetes secundarios.
- **Diagnóstico integrado:** incluye una celda para comprobar GPU, CUDA, Torch, Gradio y las dependencias críticas.
- **Pruebas específicas para Colab:** valida el notebook, la configuración y las rutas principales antes de publicar cambios.

## Inicio rápido

1. Abre el [notebook oficial de este fork](https://colab.research.google.com/github/deleonramiro085/Fooocus/blob/main/fooocus_colab.ipynb).
2. En Colab, selecciona **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4**.
3. Pega una URL directa de Hugging Face en `MODEL_URL`.
4. Si el archivo tiene otro nombre, ajusta `MODEL_FILENAME`. El preset predeterminado espera `model.safetensors`.
5. Ejecuta la celda y abre la dirección marcada como `URL PUBLICA`.

Mientras Fooocus esté encendido, la celda seguirá ejecutándose. **Eso es normal:** el proceso está manteniendo el servidor disponible, no está congelado.

## Mejoras técnicas

| Área | Fooocus heredado | Colab Edition 2.6 |
| --- | --- | --- |
| Runtime | Pines antiguos de Python y NumPy | Compatibilidad con Python 3.13 y NumPy 2 |
| PyTorch | Puede reinstalar una versión antigua | Reutiliza Torch y CUDA del runtime |
| Descargas | Flujo único y difícil de diagnosticar | `aria2c`, progreso, reintentos y reanudación |
| Modelo | Descarga implícita desde el preset | URL explícita y validación del archivo |
| Registros | Salida que puede quedar en búfer | Logs visibles línea por línea |
| Acceso web | Gradio Share únicamente | Cloudflare con fallback a Gradio |
| Persistencia | Modelos temporales | Caché opcional en Google Drive |
| Actualizador | Puede fallar con `pygit2` moderno | Compatibilidad nueva y modo `--skip-update` |
| Dependencias opcionales | Pueden bloquear el inicio | Se cargan solo cuando la función las necesita |

## Funciones de Fooocus conservadas

Esta edición mantiene las principales capacidades del proyecto original: generación SDXL, estilos y expansión de prompts, variaciones, upscale, inpaint y outpaint, Image Prompt, FaceSwap, ControlNet, LoRA, presets, múltiples relaciones de aspecto y ajustes avanzados de muestreo.

Algunas funciones avanzadas descargan modelos adicionales cuando se usan por primera vez. En el nivel gratuito de Colab, las operaciones de mayor consumo pueden agotar la RAM o provocar la desconexión de la sesión.

## Uso manual de la rama

```bash
git clone --depth 1 --branch main --single-branch \
  https://github.com/deleonramiro085/Fooocus.git
cd Fooocus
python -u entry_with_update.py --skip-update --share --preset default
```

Para Colab recomendamos usar el notebook incluido, porque automatiza la GPU, las descargas, el túnel, la caché y el diagnóstico.

## Modelo predeterminado

El preset `default` utiliza el nombre neutral `model.safetensors` y **no descarga automáticamente un checkpoint comercial o de terceros**. Esto evita enlaces rotos, límites de descarga inesperados y cambios silenciosos de modelo.

Usa una URL directa de Hugging Face compatible con SDXL. Comprueba siempre la licencia y las condiciones del modelo elegido.

## Solución rápida de problemas

**Parece detenido durante la descarga:** revisa las líneas de progreso de `aria2c`. Si la transferencia se interrumpe, vuelve a ejecutar; la descarga continuará cuando sea posible.

**No aparece una URL pública:** espera el fallback a Gradio o cambia `TUNEL` a `gradio` en el notebook.

**No detecta GPU:** activa una GPU T4 y reinicia la sesión de Colab.

**El checkpoint no carga:** confirma que sea un modelo SDXL completo, que el nombre coincida con `MODEL_FILENAME` y que la descarga supere 1 GB.

**Colab se desconecta:** desactiva funciones pesadas, reduce el número de imágenes y evita cargar varios checkpoints simultáneamente.

## Alcance de la versión 2.6

La versión 2.6 es una actualización mayor de **compatibilidad, instalación y confiabilidad en Colab**. No cambia la arquitectura generativa base de Fooocus ni promete soporte para arquitecturas distintas de SDXL.

## Créditos y licencia

Fooocus fue creado por [lllyasviel](https://github.com/lllyasviel/Fooocus) y su comunidad. Esta edición conserva la licencia [GNU GPL v3](./LICENSE) y reconoce el trabajo del proyecto original, Gradio, PyTorch, Hugging Face, Cloudflare y las demás dependencias abiertas que hacen posible el proyecto.

Consulta también la [auditoría y guía de Colab](./README_COLAB.md), el [registro histórico de cambios](./update_log.md) y la [guía de diagnóstico original](./troubleshoot.md).
