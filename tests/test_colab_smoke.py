import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_is_valid_and_uses_isolated_venv():
    notebook = json.loads((ROOT / 'fooocus_colab.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    code = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells']
                      if cell.get('cell_type') == 'code')
    ast.parse(code)
    assert "'--system-site-packages'" in code
    assert "'-m','venv'" in code
    assert "'--skip-update'" in code
    assert "trycloudflare" in code


def test_colab_requirements_do_not_install_torch_or_legacy_numpy():
    requirements = (ROOT / 'requirements_versions.txt').read_text(encoding='utf-8')
    lines = [line.strip().lower() for line in requirements.splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    assert not any(line.startswith(('torch=', 'torch==', 'torchvision=', 'torchaudio='))
                   for line in lines)
    assert 'numpy>=2.0,<3' in lines
    assert 'transformers>=4.49.0,<4.50.0' in lines


def test_model_loader_has_safe_aria2_fallback():
    source = (ROOT / 'modules' / 'model_loader.py').read_text(encoding='utf-8')
    assert "'--check-integrity=true'" in source
    assert 'download_url_to_file' in source
    assert 'os.path.getsize' in source
