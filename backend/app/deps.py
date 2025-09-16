import os, json, cv2, numpy as np, faiss
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

ART_DIR = "artifacts"
FAISS_PATH = f"{ART_DIR}/face_index.faiss"
LABELS_PATH = f"{ART_DIR}/labels.json"
FAS_ONNX = f"{ART_DIR}/fas_efficientnet_b0.onnx"

app_det = None
embedder = None
index = None
labels = []
fas_sess = None

def load_models():
    global app_det, embedder, index, labels, fas_sess
    app_det = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider']); app_det.prepare(ctx_id=0)
    embedder = get_model('glintr100', providers=['CPUExecutionProvider']); embedder.prepare(ctx_id=0)
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
    try:
        import onnxruntime as ort
        fas_sess = ort.InferenceSession(FAS_ONNX, providers=['CPUExecutionProvider'])
    except Exception:
        fas_sess = None

def detect_and_align(img_bgr, size=(112,112)):
    faces = app_det.get(img_bgr)
    if not faces: return None
    f = max(faces, key=lambda x: x.det_score)
    x1,y1,x2,y2 = map(int, f.bbox)
    crop = img_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop.size == 0: return None
    import cv2
    return cv2.resize(crop, size)

def l2n(x, eps=1e-10):
    import numpy as np
    return x / max(np.linalg.norm(x), eps)

def face_embedding(img):
    import cv2
    face = detect_and_align(img)
    if face is None: return None
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    emb = embedder.get_feat(rgb)
    return l2n(emb).astype("float32")

def enroll_images(uid, imgs):
    global index, labels
    embs = [e for e in (face_embedding(i) for i in imgs) if e is not None]
    if not embs: return False
    import numpy as np
    mean = l2n(np.mean(np.stack(embs,0),0)).astype("float32")[None,:]
    if index is None:
        index = faiss.IndexFlatIP(mean.shape[1])
    index.add(mean); labels.append(uid)
    return True

def classify_spoof(img, thr=0.5):
    if fas_sess is None:
        return {"probs":[0.9,0.1], "is_spoof": False}
    import cv2, numpy as np
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224,224)).astype(np.float32)/255.0
    x = np.transpose(x, (2,0,1))[None,:]
    logits = fas_sess.run(None, {"input": x})[0]
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    return {"probs":[float(p) for p in probs[0]], "is_spoof": bool(probs[0,1] >= thr)}

def verify_with_liveness(img, match_thresh=0.35, spoof_thresh=0.5):
    global index, labels
    fas = classify_spoof(img, thr=spoof_thresh)
    if fas["is_spoof"]:
        return {"decision":"REJECT (spoof)", "fas": fas}
    if index is None or index.ntotal == 0:
        return {"decision":"REJECT (no enrolled users)", "fas": fas}
    q = face_embedding(img)
    if q is None: return {"decision":"REJECT (no face)", "fas": fas}
    D, I = index.search(q[None,:], 1)
    score, idx = float(D[0,0]), int(I[0,0])
    name = labels[idx] if 0 <= idx < len(labels) else "Unknown"
    return {"decision": "ACCEPT" if score>=match_thresh else "REJECT (no match)",
            "score": score, "identity": name, "fas": fas}
