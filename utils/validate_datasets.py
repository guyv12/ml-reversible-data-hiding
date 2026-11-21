from pathlib import Path
from collections import defaultdict
import hashlib

DATASET_ROOT = Path("datasets/pgm") # Change to dcm to check dicoms
SPLITS = {
    "train": DATASET_ROOT / "train",
    "val": DATASET_ROOT / "val",
    "test": DATASET_ROOT / "test",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Ignore .gitkeep
files_by_split = {
    name: sorted(p for p in directory.iterdir() if not p.name.startswith("."))
    for name, directory in SPLITS.items()
}

for name, files in files_by_split.items():
    print(f"{name}: {len(files)} files")

print("\n=== 1. Filename overlap ===")
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    names_a = {p.name for p in files_by_split[a]}
    names_b = {p.name for p in files_by_split[b]}
    overlap = sorted(names_a & names_b)
    print(f"{a} vs {b}: {len(overlap)} shared filenames")
    for name in overlap[:20]:
        print(f"  {name}")

print("\n=== 2. Identical file contents (SHA-256) ===")
hashes_by_split = {}
for name, files in files_by_split.items():
    hashes = defaultdict(list)
    for path in files:
        hashes[sha256_file(path)].append(path)
    hashes_by_split[name] = hashes

pairs = [("train", "val"), ("train", "test"), ("val", "test")]
leak_found = False

for a, b in pairs:
    hashes_a = hashes_by_split[a]
    hashes_b = hashes_by_split[b]
    shared_hashes = set(hashes_a) & set(hashes_b)

    print(f"\n{a} vs {b}: {len(shared_hashes)} shared content hashes")
    for digest in sorted(shared_hashes):
        leak_found = True
        for pa in hashes_a[digest]:
            for pb in hashes_b[digest]:
                print(f"  MATCH: {pa} == {pb}")

print("\n=== 3. Exact duplicate files within each split ===")
for name, hashes in hashes_by_split.items():
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    print(f"{name}: {sum(len(paths) - 1 for paths in duplicates.values())} duplicate copies")
    for paths in duplicates.values():
        print("  " + " | ".join(str(p) for p in paths))

print("\n=== RESULT ===")
if leak_found:
    print("POTENTIAL LEAKAGE: identical file contents occur across splits.")
else:
    print("No cross-split byte-identical files found.")