# scripts/record_clips.py
import cv2, time, os, argparse, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--class_name", required=True)
ap.add_argument("--save_dir", default="data_raw")
ap.add_argument("--fps", type=int, default=30)
args = ap.parse_args()

def put_label(img, text, org, font_scale=0.9,
              text_color=(30,30,30), box_color=(235,235,235),
              thickness=2, alpha=0.65, pad=8):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x,y = org; x2,y2 = x+tw+2*pad, y+th+2*pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y-th-pad), (x2, y2-th), box_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x+pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, thickness, cv2.LINE_AA)

outdir = pathlib.Path(args.save_dir)/args.class_name
outdir.mkdir(parents=True, exist_ok=True)
idx = len(list(outdir.glob("*.mp4")))

DESIRED_W, DESIRED_H = 1280, 720
win = f"Record: {args.class_name}"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DESIRED_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)
cv2.resizeWindow(win, DESIRED_W, DESIRED_H)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
print("Press R to start, S to stop/save, Q to quit.")

recording = False
out = None
t_start = None

while True:
    ok, frame = cap.read()
    if not ok: break
    if recording:
        elapsed = time.time() - t_start
        out.write(frame)
        put_label(frame, f"REC {elapsed:4.1f}s  [S]=stop", (10,40),
                  font_scale=1.0, box_color=(255,200,120))
    else:
        put_label(frame, f"{args.class_name}  [R]=record  [Q]=quit",
                  (10,40), font_scale=1.0)
    cv2.imshow(win, frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), ord('Q')):
        break
    elif k in (ord('r'), ord('R')) and not recording:
        path = outdir / f"{args.class_name}_{idx:03d}.mp4"
        out = cv2.VideoWriter(str(path), fourcc, args.fps, (DESIRED_W, DESIRED_H))
        t_start = time.time()
        recording = True
        print(f"Recording to {path} ...")
    elif k in (ord('s'), ord('S')) and recording:
        recording = False
        out.release()
        duration = time.time() - t_start
        print(f"Saved: {path} ({duration:.1f}s)")
        idx += 1

cap.release()
cv2.destroyAllWindows()
