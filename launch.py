import os
import ssl
import sys

print('[System ARGV] ' + str(sys.argv), flush=True)
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ.setdefault('GRADIO_SERVER_PORT', '7865')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
ssl._create_default_https_context = ssl._create_unverified_context

import platform
import fooocus_version
from build_launcher import build_launcher
from modules.launch_util import (delete_folder_content, installed_version, is_installed, python,
                                 requirements_met, run, run_pip)
from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False
NO_DEPS_PACKAGES = {'gradio': '3.41.2', 'gradio_client': '0.5.0'}
OPTIONAL_REQUIREMENTS_FILE = 'requirements_optional.txt'
INSTALL_OPTIONAL_FLAG = '--install-optional'

# Colab actualiza sus paquetes con frecuencia. Preferir wheels evita compilaciones
# de minutos y --no-build-isolation reutiliza setuptools/wheel ya presentes.
PIP_STABLE_FLAGS = '--disable-pip-version-check --prefer-binary --no-build-isolation'


def prepare_environment():
    requirements_file = os.environ.get('REQS_FILE', 'requirements_versions.txt')
    print(f'Python {sys.version}', flush=True)
    print(f'Fooocus version: {fooocus_version.version}', flush=True)

    if REINSTALL_ALL or not is_installed('torch') or not is_installed('torchvision'):
        torch_index_url = os.environ.get('TORCH_INDEX_URL', 'https://download.pytorch.org/whl/cu128')
        torch_command = os.environ.get('TORCH_COMMAND',
                                       f'pip install torch torchvision --extra-index-url {torch_index_url}')
        run(f'"{python}" -m {torch_command}', 'Installing torch and torchvision',
            "Couldn't install torch", live=True)
    else:
        print(f'Using pre-installed torch {installed_version("torch")} and torchvision '
              f'{installed_version("torchvision")}.', flush=True)

    if TRY_INSTALL_XFORMERS and (REINSTALL_ALL or not is_installed('xformers')):
        if platform.system() == 'Linux':
            run_pip(f'install -U --no-deps {os.environ.get("XFORMERS_PACKAGE", "xformers")}',
                    'xformers', live=True)

    if REINSTALL_ALL or not requirements_met(requirements_file):
        print(f'[Env] Installing application requirements from {requirements_file}...', flush=True)
        run_pip(f'install {PIP_STABLE_FLAGS} -r "{requirements_file}"',
                'requirements (wheels preferidas, salida en vivo)', live=True)

    for package, version in NO_DEPS_PACKAGES.items():
        if REINSTALL_ALL or installed_version(package) != version:
            run_pip(f'install {PIP_STABLE_FLAGS} --no-deps --force-reinstall {package}=={version}',
                    f'{package}=={version} (--no-deps)', live=True)

    if not is_installed('cv2'):
        run_pip(f'install {PIP_STABLE_FLAGS} opencv-contrib-python-headless', 'opencv', live=True)

    if INSTALL_OPTIONAL_FLAG in sys.argv:
        sys.argv.remove(INSTALL_OPTIONAL_FLAG)
        if os.path.exists(OPTIONAL_REQUIREMENTS_FILE):
            run_pip(f'install {PIP_STABLE_FLAGS} -r "{OPTIONAL_REQUIREMENTS_FILE}"',
                    'extras opcionales', live=True)


def prepare_compatibility():
    from modules.compat import apply_compatibility_patches, print_environment_report
    apply_compatibility_patches()
    print_environment_report()


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')]


def ini_args():
    from args_manager import args
    return args


prepare_environment()
prepare_compatibility()
build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache
os.environ['U2NET_HOME'] = config.path_inpaint
os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}', flush=True)
    delete_folder_content(config.temp_path, '[Cleanup] ')


def download_or_warn(url, model_dir, file_name):
    try:
        return load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    except Exception as e:
        print(f'[Downloader] No se pudo descargar "{file_name}": {e}', flush=True)
        return None


def download_models(default_model, previous_default_models, checkpoint_downloads,
                    embeddings_downloads, lora_downloads, vae_downloads):
    from modules.util import get_file_from_folder_list
    for file_name, url in vae_approx_filenames:
        download_or_warn(url, config.path_vae_approx, file_name)
    download_or_warn('https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
                     config.path_fooocus_expansion, 'pytorch_model.bin')

    if args.disable_preset_download:
        print('[Models] Descarga automática desactivada por --disable-preset-download.', flush=True)
        return default_model, checkpoint_downloads
    if not args.always_download_new_model:
        if not os.path.isfile(get_file_from_folder_list(default_model, config.paths_checkpoints)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(get_file_from_folder_list(alternative_model_name, config.paths_checkpoints)):
                    default_model, checkpoint_downloads = alternative_model_name, {}
                    break
    for file_name, url in checkpoint_downloads.items():
        download_or_warn(url, os.path.dirname(get_file_from_folder_list(file_name, config.paths_checkpoints)), file_name)
    for file_name, url in embeddings_downloads.items():
        download_or_warn(url, config.path_embeddings, file_name)
    for file_name, url in lora_downloads.items():
        download_or_warn(url, os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras)), file_name)
    for file_name, url in vae_downloads.items():
        download_or_warn(url, config.path_vae, file_name)
    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads, config.vae_downloads)
config.update_files()
if not config.model_filenames:
    print('!' * 72, flush=True)
    print('[Fooocus] No hay checkpoint. Descárgalo aparte con aria2c en models/checkpoints.', flush=True)
    print(f'[Fooocus] Nombre esperado: {config.default_base_model_name}', flush=True)
    print('!' * 72, flush=True)
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)
from webui import *
