import os
import sys


detr_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(detr_root)
if detr_root not in sys.path:
    sys.path.append(detr_root)
if project_root not in sys.path:
    sys.path.append(project_root)
