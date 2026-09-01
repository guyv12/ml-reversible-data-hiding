from pathlib import Path
from typing import Final
from PySide6.QtCore import QSize

APP_VERSION: Final[str] = "1.0"
APP_SHELL_SIZE: Final[QSize] = QSize(720, 540)
ABOUT_DIALOG_SIZE: Final[QSize] = QSize(300, 200)
MENU_WIDTH: Final[int] = 320
SECTIONS_LABEL_HEIGHT: Final[int] = 20
BORDER_PADDING: Final[int] = 5
IMAGE_HEADER_MARGIN: Final[tuple[int, int, int, int]] = (BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, 0)
PHOTO_DISPLAY_MARGIN: Final[tuple[int, int, int, int]] = (10, 10, 10, 10)
HISTOGRAM_MARGIN: Final[tuple[int, int, int, int]] = (BORDER_PADDING, 10, 10, BORDER_PADDING)
ZERO_MARGINS: Final[tuple[int, int, int, int]] = (0, 0, 0, 0)
EMPTY_LAYOUT_SPACING: Final[int] = 8
ICON_SIZE: Final[tuple[int, int]] = (48, 48)
LABEL_BUTTON_SIZE: Final[tuple[int, int]] = (20, 20)

FRONTEND_DIR = Path(__file__).resolve().parent

ASSETS_DIR = FRONTEND_DIR / "assets"
STYLES_DIR = ASSETS_DIR / "styles"

STARTING_IMAGE_PATH = str(ASSETS_DIR / "start.webp")
PROCESSING_VIEW_IMAGE_PATH = str(ASSETS_DIR / "processing.webp")
ABOUT_DIALOG_IMAGE_PATH = str(ASSETS_DIR / "about.webp")
