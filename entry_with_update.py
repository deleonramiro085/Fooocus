import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

SKIP_UPDATE_FLAG = '--skip-update'

skip_update = os.environ.get('FOOOCUS_SKIP_UPDATE', '0') == '1'
if SKIP_UPDATE_FLAG in sys.argv:
    sys.argv.remove(SKIP_UPDATE_FLAG)
    skip_update = True


def resolve(module, *paths):
    """Devuelve la primera constante que exista.

    pygit2 >= 1.15 movio las constantes GIT_* a pygit2.enums y elimino las viejas,
    por eso el update fallaba en silencio con el pygit2 que trae Colab (1.20).
    """
    for path in paths:
        current = module
        try:
            for part in path.split('.'):
                current = getattr(current, part)
            return current
        except AttributeError:
            continue
    return None


def update_repository():
    import pygit2

    owner_validation = resolve(pygit2, 'enums.Option.SET_OWNER_VALIDATION',
                               'GIT_OPT_SET_OWNER_VALIDATION')
    if owner_validation is not None:
        try:
            pygit2.option(owner_validation, 0)
        except Exception:
            pass

    repo = pygit2.Repository(root)
    if repo.head_is_detached:
        return 'HEAD esta detached, se omite la actualizacion.'

    branch_name = repo.head.shorthand
    repo.remotes['origin'].fetch()

    local_branch = repo.lookup_reference(f'refs/heads/{branch_name}')
    remote_commit = repo.revparse_single(f'refs/remotes/origin/{branch_name}')
    merge_result, _ = repo.merge_analysis(remote_commit.id)

    up_to_date = resolve(pygit2, 'enums.MergeAnalysis.UP_TO_DATE', 'GIT_MERGE_ANALYSIS_UP_TO_DATE')
    fast_forward = resolve(pygit2, 'enums.MergeAnalysis.FASTFORWARD', 'GIT_MERGE_ANALYSIS_FASTFORWARD')
    reset_hard = resolve(pygit2, 'enums.ResetMode.HARD', 'GIT_RESET_HARD')

    if up_to_date is not None and merge_result & up_to_date:
        return f'Ya estaba al dia ({branch_name}).'

    if fast_forward is not None and merge_result & fast_forward:
        local_branch.set_target(remote_commit.id)
        repo.head.set_target(remote_commit.id)
        repo.checkout_tree(repo.get(remote_commit.id))
        if reset_hard is not None:
            repo.reset(local_branch.target, reset_hard)
        return f'Actualizado a {str(remote_commit.id)[:8]} ({branch_name}).'

    return 'Hay cambios locales o ramas divergentes: no se actualizo nada.'


if skip_update:
    print('[Update] Omitido.')
else:
    try:
        print(f'[Update] {update_repository()}')
    except Exception as e:
        print(f'[Update] No se pudo comprobar actualizaciones: {e}')
        print('[Update] Se continua con la copia local (esto no impide arrancar Fooocus).')

from launch import *
