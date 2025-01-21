import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
detr_path = os.path.join(project_root, 'defm_detr')
if project_root not in sys.path:
    sys.path.append(project_root)
if detr_path not in sys.path:
    sys.path.append(detr_path)
