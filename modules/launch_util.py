import os
import importlib
import importlib.util
import shutil
import subprocess
import sys
import re
import logging
import importlib.metadata
import packaging.version
from packaging.requirements import Requirement

logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

re_requirement = re.compile(r"\s*([-\w]+)\s*(?:==\s*([-+.\w]+))?\s*")

python = sys.executable
default_command_live = (os.environ.get('LAUNCH_LIVE_OUTPUT') == "1")
index_url = os.environ.get('INDEX_URL', "")

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False
    except Exception:
        return False

    return spec is not None


def installed_version(package):
    """Version instalada de un paquete, o None. No importa el modulo (barato y sin efectos)."""
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def run_pip(command, desc=None, live=default_command_live):
    try:
        index_url_line = f' --index-url {index_url}' if index_url != '' else ''
        return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}",
                   errdesc=f"Couldn't install {desc}", live=live)
    except Exception as e:
        print(e)
        print(f'CMD Failed {desc}: {command}')
        return None


def requirements_met(requirements_file):
    """True si todas las lineas del archivo ya estan satisfechas.

    Soporta cualquier especificador (>=, <, ==, rangos), no solo ==, para poder
    trabajar contra un runtime como Colab donde casi todo viene preinstalado.
    """
    missing = []

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if line == "" or line.startswith('#'):
                continue

            try:
                requirement = Requirement(line)
            except Exception as e:
                print(f"Skipping unparsable requirement line: {line} ({e})")
                continue

            version = installed_version(requirement.name)
            if version is None:
                missing.append(f"{requirement.name} (no instalado)")
                continue

            try:
                if packaging.version.parse(version) not in requirement.specifier:
                    missing.append(f"{requirement.name} {version} no cumple {requirement.specifier}")
            except Exception as e:
                missing.append(f"{requirement.name} (no se pudo comparar: {e})")

    if len(missing) > 0:
        print(f"Requirements pendientes ({len(missing)}):")
        for item in missing:
            print(f"  - {item}")
        return False

    return True


def delete_folder_content(folder, prefix=None):
    result = True

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'{prefix}Failed to delete {file_path}. Reason: {e}')
            result = False

    return result
