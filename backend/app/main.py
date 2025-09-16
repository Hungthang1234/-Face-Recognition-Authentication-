from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, cv2
from .deps import load_models, verify_with_liveness, enroll_images

app = FastAPI()

class EnrollRequest(BaseModel):
    user_id: str

@app.on_event("startup")
def _startup():
    load_models()

@app.post("/enroll")
async def enroll(user: EnrollRequest, files: list[UploadFile] = File(...)):
    imgs = []
    for f in files:
        b = await f.read()
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        imgs.append(img)
    ok = enroll_images(user.user_id, imgs)
    return {"ok": ok}

@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    b = await file.read()
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    res = verify_with_liveness(img)
    return res

@app.get("/health")
def health():
    return {"ok": True}
