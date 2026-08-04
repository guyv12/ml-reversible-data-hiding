from pathlib import Path

from frontend.config import STYLES_DIR

def load_stylesheet(target, file_name: Path):
    file_path = STYLES_DIR / file_name
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            target.setStyleSheet(f.read())
    except FileNotFoundError:
        print(f"Cant find {file_path.name} styles file")