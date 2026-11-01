
from pathlib import Path
import numpy as np
import pydicom
import argparse

parser = argparse.ArgumentParser(description="Check DICOM pixel values or prefix failing filenames.")
parser.add_argument("--prefix",action="store_true",help="Prefix failing DICOM filenames with a dot.")
parser.add_argument("--check",action="store_true",help="Check only; do not rename files (default).")
args = parser.parse_args()

ROOT = Path("datasets/dcm/test")

checked = 0
passed = 0
failed = 0
errors = 0
renamed = 0

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    # Skip files already prefixed with a dot in prefix mode.
    if args.prefix and path.name.startswith("."):
        continue

    try:
        ds = pydicom.dcmread(path)

        if "PixelData" not in ds:
            continue

        pixels = ds.pixel_array

        # 1. Set negative pixel values to zero.
        pixels = pixels.clip(min=0)

        # 2. Convert to int16 AFTER clipping.
        pixels = pixels.astype(np.int16)

        # 3. Check whether the four most significant bits are zero.
        invalid_mask = (pixels & np.uint16(0xF000)) != 0
        invalid_count = int(np.count_nonzero(invalid_mask))

        checked += 1

        if invalid_count == 0:
            passed += 1
            print(f"PASS {path.name}")

        else:
            failed += 1
            examples = np.unique(pixels[invalid_mask])[:10]

            print(
                f"FAIL {path.name}: {invalid_count}/{pixels.size} "
                f"pixels have nonzero top 4 bits; "
                f"example values: {examples.tolist()}"
            )

            # 4. Prefix failing filenames when requested.
            if args.prefix:
                new_path = path.with_name("." + path.name)

                if new_path.exists():
                    print(f"WARNING: Target already exists: {new_path}")
                else:
                    path.rename(new_path)
                    renamed += 1
                    print(f"  RENAMED -> {new_path.name}")

    except Exception as exc:
        errors += 1
        print(f"ERROR {path.name}: {exc}")

print("\n=== SUMMARY ===")
print(f"Mode: {'PREFIX' if args.prefix else 'CHECK'}")
print(f"Images checked: {checked}")
print(f"PASS (four MSBs zero after clipping): {passed}")
print(f"FAIL (four MSBs nonzero after clipping): {failed}")
print(f"ERROR: {errors}")

if args.prefix:
    print(f"Files renamed: {renamed}")
