"""Capa de compatibilidad con runtimes modernos (Python 3.13 / numpy 2.x / torch >= 2.6).

Fooocus 2.5.x nacio con Python 3.10, numpy 1.26 y torch 2.1. El runtime actual de
Google Colab es Python 3.13 con numpy 2.x y torch >= 2.6, donde varias APIs que
usan Fooocus y sus dependencias (facexlib, rembg, groundingdino, torchsde) ya no
existen. Aqui se re-exponen las que se pueden emular sin cambiar el resultado
numerico. Se aplica una sola vez, al arrancar, desde launch.py.
"""

import functools
import os
import platform
import sys
import types

# Lado mayor al que se reescala el preview antes de mandarlo al navegador.
# 0 desactiva el reescalado.
PREVIEW_MAX_SIDE = int(os.environ.get('FOOOCUS_PREVIEW_MAX_SIDE', '768'))
PREVIEW_FORMAT = os.environ.get('FOOOCUS_PREVIEW_FORMAT', 'jpeg').lower()
PREVIEW_QUALITY = int(os.environ.get('FOOOCUS_PREVIEW_QUALITY', '82'))


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

    # numpy 2.0 retiro toda la familia sctype. La usa el codigo estilo scikit-image que
    # gradio 3.41 lleva vendorizado en processing_utils._convert.
    if not hasattr(np, 'obj2sctype'):
        def obj2sctype(rep, default=None):
            try:
                if isinstance(rep, type) and issubclass(rep, np.generic):
                    return rep
                return np.dtype(rep).type
            except Exception:
                return default

        np.obj2sctype = obj2sctype

    if not hasattr(np, 'issctype'):
        np.issctype = lambda rep: np.obj2sctype(rep) is not None
    if not hasattr(np, 'maximum_sctype'):
        np.maximum_sctype = lambda t: np.dtype(t).type
    if not hasattr(np, 'sctype2char'):
        np.sctype2char = lambda sctype: np.dtype(sctype).char
    if not hasattr(np, 'find_common_type'):
        np.find_common_type = lambda array_types, scalar_types: np.result_type(
            *(list(array_types) + list(scalar_types)))
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str, np.void],
        }
    return


def _to_uint8(array):
    import numpy as np

    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.dtype in (bool, np.bool_):
        return array.astype(np.uint8) * 255
    if array.dtype.kind == 'f':
        finite = array[np.isfinite(array)] if array.size else array
        peak = float(finite.max()) if finite.size else 0.0
        scale = 255.0 if peak <= 1.0 else 1.0
        return np.clip(np.rint(np.nan_to_num(array.astype(np.float32)) * scale), 0, 255).astype(np.uint8)
    if array.dtype.kind in 'iu':
        info = np.iinfo(array.dtype)
        if info.max > 255:
            return np.clip(array.astype(np.float32) * (255.0 / info.max), 0, 255).astype(np.uint8)
        return np.clip(array, 0, 255).astype(np.uint8)
    return np.clip(np.asarray(array, dtype=np.float32), 0, 255).astype(np.uint8)


def _patch_gradio():
    """Hace utilizable gradio 3.41 sobre numpy 2 y aligera los previews.

    Dos problemas distintos, mismo origen (gradio 3.41 es de 2023):

    1. `processing_utils._convert` es una copia de las conversiones de dtype de
       scikit-image y llama a APIs retiradas en numpy 2 (obj2sctype, find_common_type,
       sctypes). La usan `Image.postprocess` y `Gallery.img_array_to_temp_file`, o sea
       tanto el preview como la miniatura final.

    2. `encode_array_to_base64` serializa el array como PNG a resolucion completa. Con
       896x1152 son ~2 MB por paso de sampler; sobre un tunel el websocket se satura y
       el ultimo mensaje (`results`, el que pinta la miniatura) se pierde sin error.
       Reescalar a 768 px y usar JPEG deja el mensaje en ~50 KB.
    """
    import base64
    from io import BytesIO

    import numpy as np
    from PIL import Image as _PILImage
    from gradio import processing_utils

    if getattr(processing_utils, '_fooocus_patched', False):
        return

    def _convert(image, dtype, force_copy=False, uniform=False):
        del uniform
        array = np.asarray(image)
        target = np.dtype('float64') if dtype is np.floating else np.dtype(dtype)
        if array.dtype == target:
            return array.copy() if force_copy else array
        if target == np.dtype(np.uint8):
            return _to_uint8(array)
        if target.kind == 'f':
            source = _to_uint8(array) if array.dtype.kind in 'ui' and array.dtype != np.uint8 else array
            if source.dtype.kind in 'ui':
                info = np.iinfo(source.dtype)
                return (source.astype(target) / float(info.max))
            return source.astype(target)
        return array.astype(target)

    def _prepare_for_transport(array):
        prepared = _to_uint8(array)
        if prepared.ndim == 3 and prepared.shape[2] == 1:
            prepared = prepared[:, :, 0]
        image = _PILImage.fromarray(prepared)

        if PREVIEW_MAX_SIDE > 0:
            longest = max(image.size)
            if longest > PREVIEW_MAX_SIDE:
                ratio = PREVIEW_MAX_SIDE / float(longest)
                image = image.resize((max(1, int(image.width * ratio)),
                                      max(1, int(image.height * ratio))),
                                     _PILImage.BILINEAR)

        has_alpha = image.mode in ('RGBA', 'LA', 'P')
        if PREVIEW_FORMAT in ('jpeg', 'jpg') and not has_alpha:
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            return image, 'JPEG', 'image/jpeg', {'quality': PREVIEW_QUALITY, 'optimize': True}
        return image, 'PNG', 'image/png', {'compress_level': 6}

    def encode_array_to_base64(image_array):
        image, pil_format, mime, options = _prepare_for_transport(image_array)
        with BytesIO() as buffer:
            image.save(buffer, pil_format, **options)
            payload = base64.b64encode(buffer.getvalue())
        return f'data:{mime};base64,' + payload.decode('utf-8')

    processing_utils._convert = _convert
    processing_utils.encode_array_to_base64 = encode_array_to_base64
    processing_utils._fooocus_patched = True

    if PREVIEW_MAX_SIDE > 0:
        print(f'[Compat] Previews limitados a {PREVIEW_MAX_SIDE} px '
              f'({PREVIEW_FORMAT}) para no saturar el tunel.', flush=True)
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
    for patch in (_patch_numpy, _patch_torch, _patch_pillow, _patch_gradio):
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
