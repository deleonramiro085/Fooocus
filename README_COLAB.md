# Fooocus 2.5.6 para Google Colab

## Estado

La rama `fix/colab-functional-bootstrap` corrige el camino crítico de una sesión limpia:

1. valida que Colab tenga GPU NVIDIA;
2. instala `aria2c` y `cloudflared` sin tocar el PyTorch CUDA del runtime;
3. descarga Juggernaut XL v8 y el parche de inpaint con reanudación;
4. valida tamaño y cabecera del checkpoint antes de arrancar;
5. abre Fooocus en `127.0.0.1:7865`;
6. comprueba que la UI responde por HTTP;
7. publica la interfaz con un túnel Cloudflare estable.

El notebook anterior no podía funcionar en una sesión limpia: `checkpoint_downloads` estaba vacío y el archivo no descargaba el modelo; además seguía usando `--share`, es decir, el túnel `gradio.live` que ya había fallado por WebSocket.

## Cómo probar

1. Abre `fooocus_colab.ipynb` en Google Colab.
2. Selecciona una GPU T4.
3. Ejecuta la celda de arranque con todas las opciones desactivadas.
4. Abre la URL que aparece como `FOOOCUS LISTO`.
5. Genera una imagen con `a red fox astronaut, cinematic photo` y cantidad `1`.

La validación termina solo cuando la imagen aparece en la galería y la consola imprime `Total time`. Que la UI abra no basta.

## Decisiones de estabilidad

- No se usa `--share`: Cloudflare transporta la cola y el WebSocket.
- `MODO_ALTA_VRAM` queda apagado por defecto: en una T4 la estabilidad importa más que unos segundos.
- Los extras opcionales quedan fuera del primer arranque.
- Google Drive es opcional. Primero se valida en disco local para separar problemas de código y latencia de Drive.
- El lanzador mantiene los pines auditados de Gradio 3.41.2 y no inicia una migración de UI hasta confirmar generación real.

## Después de confirmar

La segunda fase será optimización: caché persistente, descargas integradas en `model_loader.py`, medición de VRAM/tiempos y perfiles T4/L4/A100. La tercera fase será potenciación, incluida la migración de Gradio y funciones nuevas.
