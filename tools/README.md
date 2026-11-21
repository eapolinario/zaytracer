# tools/

Analysis scripts for debugging renders. They are Python, not Zig, because none
of them belong in the renderer: they inspect its inputs and outputs from the
outside.

Each script declares its own dependencies inline ([PEP 723][pep723]), so `uv`
resolves them per script and nothing needs a project-wide virtualenv:

```sh
uv run tools/mesh_check.py models/xyzrgb_dragon.obj
uv run tools/ppm_inspect.py image.ppm --crop 760,170,1010,430 --out /tmp/tail.png
```

The Python version is pinned in `.python-version` at the repository root; `uv`
downloads it if it is missing.

[pep723]: https://peps.python.org/pep-0723/
