import json
from importlib.resources import files
import numpy as np
def load_palette(): return json.loads(files('facadelinenet').joinpath('palette.json').read_text(encoding='utf-8'))
def colorize(labels): return np.asarray([c['rgb'] for c in load_palette()['classes']],dtype=np.uint8)[labels]
