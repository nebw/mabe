import os
import pathlib

API_KEY = None
ROOT_PATH = pathlib.Path(os.getenv("MABE_ROOT_PATH", "/srv/data/benwild/mabe"))
