_list:
    @just --list

dev SCRIPT *ARGS:
    just watch run {{SCRIPT}} {{ARGS}}

watch CMD='check' *ARGS='':
    watchexec \
        --project-origin . \
        --shell none \
        --debounce 500ms \
        --wrap-process session \
        --watch mine \
        --restart --clear -- just {{CMD}} {{ARGS}}

python *ARGS:
    @just --quiet _ensure_venv_is_active
    python {{ARGS}}

pip *ARGS:
    @just python -m pip {{ARGS}}

pip-install-requirements:
    @just pip install -r requirements.txt

run SCRIPT *ARGS='':
    @just python {{SCRIPT}} {{ARGS}}

_ensure_venv_is_active:
    #!/usr/bin/env bash
    if [ ! -n "$VIRTUAL_ENV" ]; then
        echo "Please activate the virtual environment first.";
        echo '$ source .venv/bin/activate'
        exit 1;
    fi

pull-upstream BRANCH='main':
    git fetch upstream
    git merge upstream/{{BRANCH}}
