# scripts/record_clips.py
import cv2, time, os, argparse, pathlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--class_name", required=True)
ap.add_argument("--save_dir", default="data_raw")
ap.add_argument("--fps", type=int, default=30)
args = ap.parse_args()

# Setup directories
outdir = pathlib.Path(args.save_dir)/args.class_name
outdir.mkdir(parents=True, exist_ok=True)
idx = len(list(outdir.glob("*.mp4")))

# Window setup
DESIRED_W, DESIRED_H = 1280, 720
win_name = f"Record: {args.class_name}"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, DESIRED_W, DESIRED_H)

# Global state
state = {
    "recording": False,
    "quit": False,
    "start_time": 0,
    "out_writer": None,
    "idx": idx
}

# Button definitions
def get_buttons():
    # Dynamic label/color for Record button
    rec_label = "STOP (S)" if state["recording"] else "RECORD (R)"
    rec_color = (0, 0, 200) if state["recording"] else (0, 200, 0) # Red if rec, Green if idle
    
    return [
        {"id": "rec",  "x": 50,   "y": 620, "w": 200, "h": 60, "label": rec_label, "color": rec_color, "text_color": (255,255,255)},
        {"id": "quit", "x": 1150, "y": 30,  "w": 100, "h": 40, "label": "QUIT (Q)", "color": (50, 50, 50), "text_color": (200,200,200)}
    ]

def draw_buttons(img, buttons):
    for b in buttons:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # Draw shadow
        cv2.rectangle(img, (x+4, y+4), (x+w+4, y+h+4), (0,0,0), -1)
        # Draw button
        cv2.rectangle(img, (x, y), (x+w, y+h), b["color"], -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (200,200,200), 2) # Border
        
        # Text centering
        (tw, th), _ = cv2.getTextSize(b["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, b["label"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, b["text_color"], 2, cv2.LINE_AA)

def toggle_recording():
    if not state["recording"]:
        # Start recording
        fname = outdir / f"{args.class_name}_{state['idx']:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        state["out_writer"] = cv2.VideoWriter(str(fname), fourcc, args.fps, (DESIRED_W, DESIRED_H))
        state["start_time"] = time.time()
        state["recording"] = True
        print(f"Started recording: {fname}")
    else:
        # Stop recording
        state["recording"] = False
        if state["out_writer"]:
            state["out_writer"].release()
            state["out_writer"] = None
        duration = time.time() - state["start_time"]
        print(f"Saved clip {state['idx']} ({duration:.1f}s)")
        state["idx"] += 1

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        buttons = get_buttons()
        for b in buttons:
            if b["x"] <= x <= b["x"]+b["w"] and b["y"] <= y <= b["y"]+b["h"]:
                if b["id"] == "rec":
                    toggle_recording()
                elif b["id"] == "quit":
                    state["quit"] = True

cv2.setMouseCallback(win_name, on_mouse)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DESIRED_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)

print(f"Mouse control enabled. Class: {args.class_name}")

while not state["quit"]:
    ok, frame = cap.read()
    if not ok: break

    # Handle recording write
    if state["recording"] and state["out_writer"]:
        state["out_writer"].write(frame)

    # Draw UI
    disp = frame.copy()
    
    # Status text
    if state["recording"]:
        elapsed = time.time() - state["start_time"]
        cv2.circle(disp, (30, 30), 10, (0, 0, 255), -1) # Red dot
        cv2.putText(disp, f"REC {elapsed:.1f}s", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(disp, f"Ready: {args.class_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    draw_buttons(disp, get_buttons())
    
    cv2.imshow(win_name, disp)
    
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        state["quit"] = True
    elif key in (ord('r'), ord('R')) and not state["recording"]:
        toggle_recording()
    elif key in (ord('s'), ord('S')) and state["recording"]:
        toggle_recording()

# Cleanup
if state["recording"] and state["out_writer"]:
    state["out_writer"].release()
cap.release()
cv2.destroyAllWindows()
