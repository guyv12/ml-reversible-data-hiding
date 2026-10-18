    # -*- mode: python ; coding: utf-8 -*-

EXCLUDES = [
    'segmentation_models_pytorch', 'timm', 'torchvision',
    'huggingface_hub', 'hf_xet', 'safetensors',
    'matplotlib', 'contourpy', 'kiwisolver', 'dateutil',
    'httpx', 'httpcore', 'click', 'yaml',
    'skimage',
    'PySide6.QtNetwork', 'PySide6.QtQml', 'PySide6.QtQuick',
    'PySide6.QtPdf', 'PySide6.QtVirtualKeyboard',
    'PySide6.QtTest',
    'torch._inductor', 'torch._dynamo', 'torch.onnx', 'torch.utils.tensorboard',
    'experiments', 'backend.data', 'backend.models.train',
    'backend.predictor.test', 'backend.compressor.test',
    'backend.receiver.test', 'backend.data.test',
    'tkinter',
]

DROP_FILES = {
    'PySide6/opengl32sw.dll',
    'PySide6/Qt6Pdf.dll', 'PySide6/Qt6Network.dll', 'PySide6/Qt6VirtualKeyboard.dll',
    'PySide6/Qt6Qml.dll', 'PySide6/Qt6QmlMeta.dll', 'PySide6/Qt6QmlModels.dll',
    'PySide6/Qt6QmlWorkerScript.dll', 'PySide6/Qt6Quick.dll',
    'cv2/opencv_videoio_ffmpeg500_64.dll',
}

DROP_DIRS = (
    'PySide6/translations/',
    'PySide6/plugins/platforminputcontexts/',
    'PySide6/plugins/generic/',
    'PySide6/plugins/networkinformation/',
    'PySide6/plugins/tls/',
    'PySide6/plugins/styles/',
)

KEEP_PLUGINS = {
    'PySide6/plugins/platforms/': {'qwindows.dll'},
    'PySide6/plugins/imageformats/': {'qwebp.dll', 'qsvg.dll', 'qico.dll'},
}

def _keep(entry):
    name = entry[0].replace('\\', '/')
    if name in DROP_FILES or name.startswith(DROP_DIRS):
        return False
    for folder, allowed in KEEP_PLUGINS.items():
        if name.startswith(folder):
            return name[len(folder):] in allowed
    return True

a = Analysis(
    ['frontend/app.py'],
    pathex=['.'],
    binaries=[],
    datas=[('frontend/assets', 'frontend/assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RDHEI Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='frontend/assets/icons/app.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RDHEI Studio',
)
