# scripts/extract_keypoints.py
import os, glob, cv2, numpy as np
import mediapipe as mp
from tqdm import tqdm

RAW = "data_raw"
OUT = "data_proc/dataset_keypoints.npz"
SEQ_LEN = 32  # frames per sample
os.makedirs("data_proc", exist_ok=True)

# fixed dims
POSE_IDXS = [0,11,12,13,14,15,16,23,24]   # 9 joints (x,y,vis) -> 27
POSE_DIM  = 9 * 3
HAND_JOINTS = 21                           # x,y only -> 42 per hand
HAND_DIM = HAND_JOINTS * 2
F = POSE_DIM + HAND_DIM + HAND_DIM         # 111

mp_hol = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# nice thick drawing style for clarity
DRAW_SPEC_LAND = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
DRAW_SPEC_CONN = mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2)

def to_fixed_vec(res) -> np.ndarray:
    """Always returns a (111,) float32 vector: [pose(27), left(42), right(42)]."""
    v = np.zeros(F, dtype=np.float32)
    off = 0

    # pose (x,y,visibility)
    pl = getattr(res, "pose_landmarks", None)
    if pl and pl.landmark:
        for k, i in enumerate(POSE_IDXS):
            lm = pl.landmark[i]
            v[off + 3*k + 0] = lm.x
            v[off + 3*k + 1] = lm.y
            v[off + 3*k + 2] = getattr(lm, "visibility", 1.0)
    off += POSE_DIM

    # left hand
    lh = getattr(res, "left_hand_landmarks", None)
    if lh and lh.landmark:
        for k in range(min(HAND_JOINTS, len(lh.landmark))):
            lm = lh.landmark[k]
            v[off + 2*k + 0] = lm.x
            v[off + 2*k + 1] = lm.y
    off += HAND_DIM

    # right hand
    rh = getattr(res, "right_hand_landmarks", None)
    if rh and rh.landmark:
        for k in range(min(HAND_JOINTS, len(rh.landmark))):
            lm = rh.landmark[k]
            v[off + 2*k + 0] = lm.x
            v[off + 2*k + 1] = lm.y

    return v

def pad_trim(seq, T):
    """Center-crop or pad (repeat last) to exactly T frames."""
    if len(seq) >= T:
        s = (len(seq) - T) // 2
        return np.stack(seq[s:s+T], 0)
    last = seq[-1] if seq else np.zeros(F, np.float32)
    return np.stack(seq + [last] * (T - len(seq)), 0)

# small text helper
def put_label(img, text, org, font_scale=0.8, color=(255,255,255), bg=(0,0,0), alpha=0.5, pad=4):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - th - pad), (x + tw + 2*pad, y + pad), bg, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

classes = [d for d in sorted(os.listdir(RAW)) if os.path.isdir(os.path.join(RAW, d))]
X, y = [], []

# visual window
cv2.namedWindow("extract", cv2.WINDOW_NORMAL)
cv2.resizeWindow("extract", 960, 540)

with mp_hol.Holistic(model_complexity=1, smooth_landmarks=True) as hol:
    for ci, c in enumerate(classes):
        vids = sorted(glob.glob(os.path.join(RAW, c, "*.mp4")))
        for vpath in tqdm(vids, desc=c):
            cap = cv2.VideoCapture(vpath)
            seq = []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fn = os.path.basename(vpath)
            while True:
                ok, fr = cap.read()
                if not ok: break
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                res = hol.process(rgb)
                vec = to_fixed_vec(res)
                seq.append(vec)

                # draw landmarks for visual verification
                disp = fr.copy()
                if res.pose_landmarks:
                    mp_drawing.draw_landmarks(disp, res.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS, DRAW_SPEC_LAND, DRAW_SPEC_CONN)
                if res.left_hand_landmarks:
                    mp_drawing.draw_landmarks(disp, res.left_hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS, DRAW_SPEC_LAND, DRAW_SPEC_CONN)
                if res.right_hand_landmarks:
                    mp_drawing.draw_landmarks(disp, res.right_hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS, DRAW_SPEC_LAND, DRAW_SPEC_CONN)
                if res.face_landmarks and hasattr(mp.solutions.face_mesh, "FACEMESH_CONTOURS"):
                    mp_drawing.draw_landmarks(disp, res.face_landmarks,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS, DRAW_SPEC_LAND, DRAW_SPEC_CONN)

                # overlay progress text
                fnum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                put_label(disp, f"{c}  {fn}  frame {fnum}/{total}", (10, 35))
                cv2.imshow("extract", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break  # ESC to abort this clip

            cap.release()
            if not seq:
                continue
            seq = pad_trim(seq, SEQ_LEN)

            mu = seq.mean(0, keepdims=True)
            sd = seq.std(0, keepdims=True) + 1e-6
            seq = (seq - mu) / sd

            X.append(seq); y.append(ci)

cv2.destroyAllWindows()

X = np.stack(X, 0) if X else np.zeros((0, SEQ_LEN, F), np.float32)
y = np.array(y, np.int64)
np.savez_compressed(OUT, X=X, y=y, classes=np.array(classes))
print("Saved", OUT, X.shape, y.shape, classes)
