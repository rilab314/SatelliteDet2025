import os
import sys


data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(data_root)
if project_root not in sys.path:
    sys.path.append(project_root)
