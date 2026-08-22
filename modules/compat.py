"""Capa de compatibilidad con runtimes modernos (Python 3.13 / numpy 2.x / torch >= 2.6).

Fooocus 2.5.x nacio con Python 3.10, numpy 1.26 y torch 2.1. El runtime actual de
Google Colab es Python 3.13 con numpy 2.x y torch >= 2.6, donde varias APIs que
usan Fooocus y sus dependencias (facexlib, rembg, groundingdino, torchsde) ya no
existen. Aqui se re-exponen las que se pueden emular sin cambiar el resultado
numerico. Se aplica una sola vez, al arrancar, desde launch.py.
"""

import functools
import platform
import sys
import types


def _patch_numpy():
    import numpy as np

    aliases = {
        'float_': 'float64', 'complex_': 'complex128', 'unicode_': 'str_', 'string_': 'bytes_',
        'bool8': 'bool_', 'int0': 'intp', 'uint0': 'uintp', 'object0': 'object_',
        'str0': 'str_', 'void0': 'void',
    }
    for old, new in aliases.items():
        if not hasattr(np, old) and hasattr(np, new):
            setattr(np, old, getattr(np, new))

    constants = (('NaN', float('nan')), ('NAN', float('nan')), ('Inf', float('inf')),
                 ('Infinity', float('inf')), ('PINF', float('inf')), ('NINF', float('-inf')))
    for name, value in constants:
        if not hasattr(np, name):
            setattr(np, name, value)

    renamed = {'alltrue': 'all', 'sometrue': 'any', 'product': 'prod', 'cumproduct': 'cumprod',
               'round_': 'round', 'row_stack': 'vstack', 'trapz': 'trapezoid'}
    for old, new in renamed.items():
        if not hasattr(np, old) and hasattr(np, new):
            setattr(np, old, getattr(np, new))

    if not hasattr(np, 'in1d'):
        np.in1d = lambda ar1, ar2, **kwargs: np.isin(np.asarray(ar1).ravel(), ar2, **kwargs)
    if not hasattr(np, 'asfarray'):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    return


def _patch_torch():
    import torch

    # torch >= 2.4 marco como obsoletas (y las ramas nuevas ya eliminan) estas APIs,
    # que ldm_patched y varias dependencias siguen llamando.
    if not hasattr(torch, 'get_autocast_gpu_dtype') and hasattr(torch, 'get_autocast_dtype'):
        torch.get_autocast_gpu_dtype = lambda: torch.get_autocast_dtype('cuda')

    amp = getattr(torch, 'amp', None)
    if amp is not None and hasattr(amp, 'autocast'):
        cuda_amp = getattr(torch.cuda, 'amp', None)
        if cuda_amp is None:
            cuda_amp = types.ModuleType('torch.cuda.amp')
            torch.cuda.amp = cuda_amp
            sys.modules['torch.cuda.amp'] = cuda_amp
        if not hasattr(cuda_amp, 'autocast'):
            cuda_amp.autocast = functools.partial(amp.autocast, 'cuda')
        for name in ('custom_fwd', 'custom_bwd'):
            if not hasattr(cuda_amp, name) and hasattr(amp, name):
                setattr(cuda_amp, name, getattr(amp, name))

    # torch >= 2.6 usa weights_only=True por defecto. Fooocus ya pasa el flag de forma
    # explicita en todas sus llamadas; esto solo habilita los tipos que necesitan los
    # checkpoints de terceros (GroundingDINO guarda un argparse.Namespace con los pesos).
    try:
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass
    return


def _patch_pillow():
    from PIL import Image

    for name, member in (('ANTIALIAS', 'LANCZOS'), ('LINEAR', 'BILINEAR'), ('CUBIC', 'BICUBIC')):
        if not hasattr(Image, name) and hasattr(Image, 'Resampling') and hasattr(Image.Resampling, member):
            setattr(Image, name, getattr(Image.Resampling, member))
    return


def apply_compatibility_patches():
    for patch in (_patch_numpy, _patch_torch, _patch_pillow):
        try:
            patch()
        except Exception as e:
            print(f'[Compat] {patch.__name__} no se pudo aplicar: {e}')
    return


def print_environment_report():
    from modules.launch_util import installed_version

    lines = [f'python {platform.python_version()}']
    try:
        import torch
        lines.append(f'torch {torch.__version__} (cuda {torch.version.cuda})')
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            lines.append(f'gpu {props.name}, {props.total_memory // (1024 ** 2)} MB, '
                         f'sm_{props.major}{props.minor}')
        else:
            lines.append('gpu NO DETECTADA -> en Colab: Entorno de ejecucion > '
                         'Cambiar tipo de entorno > GPU')
    except Exception as e:
        lines.append(f'torch no se pudo importar: {e}')

    for package in ('numpy', 'gradio', 'gradio_client', 'transformers', 'tokenizers',
                    'huggingface_hub', 'pydantic', 'fastapi', 'starlette', 'websockets'):
        lines.append(f'{package} {installed_version(package)}')

    print('-' * 72)
    for line in lines:
        print(f'[Env] {line}')
    print('-' * 72)
    return
