from fastapi import FastAPI, UploadFile, File
from pathlib import Path

app = FastAPI()

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    dest = RAW_DATA_DIR / "telco_churn.csv"
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"status": "success", "saved_to": str(dest)}