from pathlib import Path

from PySide6.QtWidgets import QApplication

def load_stylesheet(target, filename: Path):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            target.setStyleSheet(f.read())
    except FileNotFoundError:
        print(f"Cant find {filename.name} styles file")