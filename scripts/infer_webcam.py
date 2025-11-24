# scripts/infer_webcam.py
import cv2, torch, numpy as np, mediapipe as mp

# ---------- load checkpoint ----------
ckpt = torch.load("models/mini_lstm_best.pth", map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt.get("model"))
classes    = list(ckpt["classes"])
F          = int(ckpt.get("feat_dim", 111))

# ---------- tiny model (must match training) ----------
class TinyLSTM(torch.nn.Module):
    def __init__(self, in_dim, num_classes, hidden=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc   = torch.nn.Sequential(torch.nn.LayerNorm(hidden*2), torch.nn.Linear(hidden*2, num_classes))
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.fc(y[:, -1, :])

net = TinyLSTM(F, len(classes))
net.load_state_dict(state_dict)
net.eval()

# ---------- feature extraction: fixed 111 dims ----------
POSE_IDXS   = [0,11,12,13,14,15,16,23,24]  # 9 pose joints
POSE_DIM    = 9 * 3                        # x,y,vis
HAND_JOINTS = 21
HAND_DIM    = HAND_JOINTS * 2              # x,y only
FIXED_FEAT  = POSE_DIM + HAND_DIM + HAND_DIM  # 111
assert F == FIXED_FEAT, f"Model expects F={F}, but code outputs {FIXED_FEAT}."

def to_fixed_vec(res) -> np.ndarray:
    v = np.zeros(FIXED_FEAT, dtype=np.float32); off = 0
    pl = getattr(res, "pose_landmarks", None)
    if pl and pl.landmark:
        for k,i in enumerate(POSE_IDXS):
            lm = pl.landmark[i]
            v[off+3*k+0] = lm.x; v[off+3*k+1] = lm.y; v[off+3*k+2] = getattr(lm,"visibility",1.0)
    off += POSE_DIM
    lh = getattr(res, "left_hand_landmarks", None)
    if lh and lh.landmark:
        for k in range(min(HAND_JOINTS, len(lh.landmark))):
            lm = lh.landmark[k]
            v[off+2*k+0] = lm.x; v[off+2*k+1] = lm.y
    off += HAND_DIM
    rh = getattr(res, "right_hand_landmarks", None)
    if rh and rh.landmark:
        for k in range(min(HAND_JOINTS, len(rh.landmark))):
            lm = rh.landmark[k]
            v[off+2*k+0] = lm.x; v[off+2*k+1] = lm.y
    return v

# ---------- text with readable background ----------
def put_label(img, text, org, font_scale=1.0, text_color=(30,30,30), box_color=(235,235,235), thickness=2, alpha=0.65, pad=8):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x,y = org; x2,y2 = x+tw+2*pad, y+th+2*pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y-th-pad), (x2, y2-th), box_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x+pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

# ---------- motion tracking helpers ----------
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# Softer colors / thicker lines for visibility
DRAW_SPEC_LAND = mp_drawing.DrawingSpec(color=(20, 220, 20), thickness=2, circle_radius=2)
DRAW_SPEC_CONN = mp_drawing.DrawingSpec(color=(30, 160, 255), thickness=2, circle_radius=2)

def draw_tracking(img, res):
    # Hands
    if res.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            DRAW_SPEC_LAND, DRAW_SPEC_CONN
        )
    if res.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            DRAW_SPEC_LAND, DRAW_SPEC_CONN
        )
    # Face contours (clearer than full tessellation)
    if res.face_landmarks and hasattr(mp.solutions.face_mesh, "FACEMESH_CONTOURS"):
        mp_drawing.draw_landmarks(
            img, res.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
            DRAW_SPEC_LAND, DRAW_SPEC_CONN
        )
    # Pose (upper body)
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            DRAW_SPEC_LAND, DRAW_SPEC_CONN
        )

def lms_to_bbox(lms, w, h, margin=0.04):
    """Compute pixel bbox from normalized landmarks; returns (x1,y1,x2,y2) or None."""
    if not lms or not lms.landmark:
        return None
    xs = [int(pt.x * w) for pt in lms.landmark]
    ys = [int(pt.y * h) for pt in lms.landmark]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    dx = int((x2 - x1) * margin); dy = int((y2 - y1) * margin)
    return max(0, x1 - dx), max(0, y1 - dy), min(w-1, x2 + dx), min(h-1, y2 + dy)

def smooth_bbox(prev, curr, alpha=0.6):
    """Exponential moving average for bbox."""
    if prev is None or curr is None:
        return curr
    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = curr
    sx1 = int(alpha*px1 + (1-alpha)*cx1)
    sy1 = int(alpha*py1 + (1-alpha)*cy1)
    sx2 = int(alpha*px2 + (1-alpha)*cx2)
    sy2 = int(alpha*py2 + (1-alpha)*cy2)
    return (sx1, sy1, sx2, sy2)

def draw_bbox(img, box, color=(255, 210, 70), thickness=2):
    if box is None: return
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness, cv2.LINE_AA)

# ---------- words & thresholds ----------
SEQ_LEN    = 32
SMOOTH_K   = 5
THRESH     = 0.60

TARGET_WORDS = ["HELLO","GOODBYE","PLEASE","THANK YOU","YES","NO"]
NONE_NAMES   = {"NONE","None","idle","IDLE"}

def _norm(s: str) -> str:
    return s.replace("_"," ").strip().upper()

TARGET_NORM = {_norm(w) for w in TARGET_WORDS}
NONE_NORM   = {_norm(w) for w in NONE_NAMES}

# toggles
SHOW_TRACK = True   # 't'
SHOW_BBOX  = True   # 'b'

mp_hol = mp.solutions.holistic
buf, logits_hist = [], []

def draw_hud(img, thr):
    words_line = "Words: " + " | ".join(TARGET_WORDS)
    put_label(img, words_line, (12, img.shape[0]-40), font_scale=0.65, thickness=1)
    put_label(img, f"[T]rack:{'ON' if SHOW_TRACK else 'OFF'}  [B]ox:{'ON' if SHOW_BBOX else 'OFF'}  Threshold: {thr:.2f}  ([ / ] adjust, ESC quit)",
              (12, img.shape[0]-12), font_scale=0.65, thickness=1)

# ---------- camera ----------
DESIRED_W, DESIRED_H = 1280, 720
win_name = "Sign word detector (silent until confident)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, DESIRED_W, DESIRED_H)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DESIRED_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)

# smoothed bboxes
bb_face = bb_l = bb_r = None

with mp_hol.Holistic(model_complexity=1, smooth_landmarks=True) as hol:
    while True:
        ok, fr = cap.read()
        if not ok: break

        h, w = fr.shape[:2]
        rgb  = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res  = hol.process(rgb)

        disp = fr.copy()

        # motion tracking overlays
        if SHOW_TRACK:
            draw_tracking(disp, res)

        if SHOW_BBOX:
            cur_face = lms_to_bbox(res.face_landmarks, w, h, margin=0.08)
            cur_l    = lms_to_bbox(res.left_hand_landmarks,  w, h, margin=0.12)
            cur_r    = lms_to_bbox(res.right_hand_landmarks, w, h, margin=0.12)
            bb_face  = smooth_bbox(bb_face, cur_face, alpha=0.6)
            bb_l     = smooth_bbox(bb_l,    cur_l,    alpha=0.6)
            bb_r     = smooth_bbox(bb_r,    cur_r,    alpha=0.6)
            draw_bbox(disp, bb_face, (120,190,255), 2)  # face (light blue)
            draw_bbox(disp, bb_l,    (60,230,120),  2)  # left hand (green)
            draw_bbox(disp, bb_r,    (60,230,120),  2)  # right hand (green)

        # inference
        vec = to_fixed_vec(res)
        buf.append(vec)
        if len(buf) > SEQ_LEN: buf = buf[-SEQ_LEN:]

        if len(buf) == SEQ_LEN:
            seq = np.stack(buf, 0)
            mu  = seq.mean(0, keepdims=True)
            sd  = seq.std(0, keepdims=True) + 1e-6
            seq = (seq - mu) / sd
            x   = torch.from_numpy(seq[None, ...]).float()
            with torch.no_grad():
                logit = net(x).numpy()[0]
            logits_hist.append(logit); logits_hist = logits_hist[-SMOOTH_K:]
            sm = np.mean(logits_hist, 0)

            p = np.exp(sm - sm.max()); p /= p.sum()
            k = int(p.argmax()); conf = float(p[k])
            pred_raw = str(classes[k])
            pred = _norm(pred_raw)

            if pred in TARGET_NORM and pred not in NONE_NORM and conf >= THRESH:
                put_label(disp, f"{pred_raw}  {conf*100:4.1f}%", (20, 50), font_scale=1.1)

        draw_hud(disp, THRESH)
        cv2.imshow(win_name, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break               # ESC
        elif key == ord(']'): THRESH = min(0.95, THRESH + 0.02)
        elif key == ord('['): THRESH = max(0.05, THRESH - 0.02)
        elif key in (ord('t'), ord('T')): SHOW_TRACK = not SHOW_TRACK
        elif key in (ord('b'), ord('B')): SHOW_BBOX  = not SHOW_BBOX

cap.release()
cv2.destroyAllWindows()
