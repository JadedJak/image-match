"""Docker CMD entrypoint -- picks the UI engine per config.UI_ENGINE (see
app/config.py), so switching between them is a .env change + container
recreate, not a rebuild. Both app/ui.py (Gradio) and app/ui_nicegui.py
(NiceGUI, the default) expose a run() function that starts the server;
this just calls whichever one is configured."""
from . import config

if config.UI_ENGINE == "gradio":
    from .ui import run
else:
    from .ui_nicegui import run

if __name__ == "__main__":
    run()
