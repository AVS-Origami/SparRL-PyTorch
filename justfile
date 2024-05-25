train *ARGS:
    python3 code/main.py {{ARGS}} --workers 32

eval *ARGS:
    python3 code/main.py {{ARGS}} --spar_tree pruned.tree --tree --load --eval