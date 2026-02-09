import shutil
from pathlib import Path

RAW = Path("data/raw")
RAW_ZONE = Path("raw_zone")
RAW_ZONE.mkdir(exist_ok=True)

def ingest():
    for f in RAW.glob("*.csv"):
        dst = RAW_ZONE / f.name
        shutil.copy(f, dst)
        print(f"Ingested {f.name} → {dst}")

if __name__ == "__main__":
    ingest()