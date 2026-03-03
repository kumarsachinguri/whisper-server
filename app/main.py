import whisper
import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# Load model once at startup
model = whisper.load_model("base")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "Whisper API is running"}


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Create unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Transcribe
        result = model.transcribe(file_path)

        # Cleanup
        os.remove(file_path)

        return JSONResponse(content={
            "text": result["text"]
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )