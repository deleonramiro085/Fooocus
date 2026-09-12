import os

import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

args_parser.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
args_parser.parser.add_argument("--disable-preset-selection", action='store_true',
                                help="Disables preset selection in Gradio.")

args_parser.parser.add_argument("--language", type=str, default='default',
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

# For example, https://github.com/lllyasviel/Fooocus/issues/849
args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")

args_parser.parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to the outputs folder.")

args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio.")

args_parser.parser.add_argument("--disable-metadata", action='store_true',
                                help="Disables saving metadata to images.")

args_parser.parser.add_argument("--disable-preset-download", action='store_true',
                                help="Disables downloading models for presets", default=False)

args_parser.parser.add_argument("--disable-enhance-output-sorting", action='store_true',
                                help="Disables enhance output sorting for final image gallery.")

args_parser.parser.add_argument("--enable-auto-describe-image", action='store_true',
                                help="Enables automatic description of uov and enhance image when prompt is empty", default=False)

args_parser.parser.add_argument("--always-download-new-model", action='store_true',
                                help="Always download newer models", default=False)

args_parser.parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

args_parser.args = args_parser.parser.parse_args()


def _vram_policy_chosen(parsed):
    """True si el usuario ya eligio a mano una politica de VRAM."""
    return bool(parsed.always_gpu or parsed.always_high_vram or parsed.always_normal_vram
                or parsed.always_low_vram or parsed.always_no_vram or parsed.always_cpu)


def _autotune_memory_policy(parsed):
    """Evita que la RAM del sistema se use como zona de descarga de la VRAM.

    En Colab la GPU suele tener MAS VRAM que RAM tiene la maquina (un T4 son
    15 GiB de VRAM contra ~12.7 GiB de RAM). Con la politica por defecto
    (NORMAL_VRAM y always_offload_from_vram activo) ldm_patched trata la CPU como
    almacen: unet_inital_load_device() devuelve cpu de forma incondicional,
    unet_offload_device() devuelve cpu, y free_memory() pierde su corto circuito
    y copia los pesos de vuelta a RAM tras cada pasada. Un checkpoint SDXL de
    ~6.5 GiB mas su page cache revienta el techo de RAM y el kernel mata el
    proceso con SIGKILL: en Colab solo se ve "returncode=-9", sin traceback.

    Si la VRAM alcanza para tener el modelo residente y nadie eligio politica a
    mano, se deja todo en la GPU. Desactivable con FOOOCUS_DISABLE_VRAM_AUTOTUNE=1.
    """
    if os.environ.get('FOOOCUS_DISABLE_VRAM_AUTOTUNE') == '1':
        return
    if _vram_policy_chosen(parsed):
        return

    try:
        import torch
        if not torch.cuda.is_available():
            return
        total_vram = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return

    try:
        import psutil
        total_ram = psutil.virtual_memory().total
    except Exception:
        return

    if total_vram >= total_ram * 0.9:
        parsed.always_high_vram = True
        parsed.disable_offload_from_vram = True
        print('[VRAM] VRAM {:.1f} GiB >= RAM {:.1f} GiB: se activan --always-high-vram '
              'y --disable-offload-from-vram para no descargar los pesos a la RAM '
              'del sistema (evita que el OOM killer mate el proceso con SIGKILL/-9).'
              .format(total_vram / 1024 ** 3, total_ram / 1024 ** 3), flush=True)


_autotune_memory_policy(args_parser.args)

# (Disable by default because of issues like https://github.com/lllyasviel/Fooocus/issues/724)
args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

if args_parser.args.disable_analytics:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

args = args_parser.args
