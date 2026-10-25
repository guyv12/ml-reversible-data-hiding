import shutil
from pathlib import Path


files = sorted(Path("CAR+TCIA").glob("*"))
car = []
tcia = []

for f in files:
    if f.name.endswith(".dcm"):
        tcia.append(f)
    else:
        car.append(f)

car_dest = Path("CAR97").mkdir(parents=True, exist_ok=True)
tcia_dest = Path("TCIA").mkdir(parents=True, exist_ok=True)

for f in car:
    shutil.copy2(f, car_dest)
for f in tcia:
    shutil.copy2(f, tcia_dest)
