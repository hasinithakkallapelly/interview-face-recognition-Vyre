# Interview Proctoring System

A real-time computer-vision prototype that monitors an interview session for face presence, positioning, identity changes, and prohibited objects.

## Features

- Detects zero, one, or multiple faces using YOLO
- Checks whether the detected face stays near the center of the frame
- Compares the live face against pre-session candidate reference images
- Detects phones, books, and laptops using a COCO-trained YOLO model
- Counts only violations that persist for a configured duration
- Applies cooldowns so one continuous event is not counted repeatedly
- Terminates the session after three confirmed infractions

## How identity checking works

Before the monitored session begins, the application loads one or more reference images of the candidate. It detects the single face in each image and trains an OpenCV LBPH recognizer using those samples.

During the session, the live face is compared with the reference samples. Several consecutive mismatches are required before an identity violation is raised.

This verifies that the person remains consistent with the supplied reference images. It is a prototype, not a production biometric-authentication system.

## Requirements

- Python 3.10–3.12
- A webcam
- macOS, Linux, or Windows
- Internet access on the first run so Ultralytics can download `yolov8n.pt`
- One or more clear candidate reference images

The reference images should:

- Contain exactly one visible face
- Be well lit
- Show the face clearly without heavy obstruction
- Preferably include two or three slightly different angles

## Setup

### macOS or Linux

```bash
git clone https://github.com/hasinithakkallapelly/interview-face-recognition-Vyre.git
cd interview-face-recognition-Vyre

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
git clone https://github.com/hasinithakkallapelly/interview-face-recognition-Vyre.git
cd interview-face-recognition-Vyre

py -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Create a local folder for reference images. It is ignored by Git and should not contain images you intend to publish.

```bash
mkdir reference_images
```

Place one or more candidate photographs in that folder, then run:

```bash
python main.py \
  --reference-image reference_images/front.jpg \
  --reference-image reference_images/angle.jpg
```

For a single reference image:

```bash
python main.py --reference-image reference_images/front.jpg
```

Press `Esc` to stop the session.

## Optional arguments

```text
--camera NUMBER          Camera index; default: 0
--object-every NUMBER    Run object detection every N frames; default: 5
--face-model PATH        Path to the face-detection model
--device-model PATH      Local model path or Ultralytics model name
```

Example using a different camera:

```bash
python main.py --reference-image reference_images/front.jpg --camera 1
```

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The automated tests cover:

- Multiple reference-image enrollment
- Invalid face crops
- Consecutive identity mismatches
- Transient violations
- Violation-duration thresholds
- Cooldown behavior
- Three-infraction termination

Real-world recognition accuracy still needs testing with actual faces, lighting conditions, and camera angles.

## Project structure

```text
main.py                     Application entry point
identity_verifier.py        LBPH enrollment and live comparison
device_detector.py          Prohibited-object detection
infraction_counter.py       Duration, cooldown, and termination logic
utils.py                    Alert and cancellation output
models/yolov8n-face.pt      Face-detection model
test_identity_verifier.py   Identity-verification tests
test_infraction_counter.py  Infraction-policy tests
```

## Configuration

Important values can be adjusted in the code:

- LBPH mismatch threshold: `IdentityVerifier(mismatch_threshold=...)`
- Consecutive mismatches: `consecutive_required`
- Violation duration: `minimum_duration`
- Repeat cooldown: `cooldown`
- Maximum infractions: `max_infractions`

Thresholds should be calibrated using representative reference and webcam images. A value that works for one camera or lighting environment may be unreliable in another.

## Troubleshooting

### `AttributeError: module 'cv2' has no attribute 'face'`

The standard OpenCV package does not include the LBPH face module. Remove conflicting OpenCV packages and reinstall the contrib build:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-contrib-python
```

### Camera does not open

Try another camera index:

```bash
python main.py --reference-image reference_images/front.jpg --camera 1
```

On macOS, allow camera access for Terminal or your IDE under:

`System Settings → Privacy & Security → Camera`

### Reference image is rejected

The application requires exactly one detectable face in every reference image. Use a clearer, well-lit image with only the candidate visible.

### Object model download fails

Download `yolov8n.pt` manually and pass its path:

```bash
python main.py \
  --reference-image reference_images/front.jpg \
  --device-model /path/to/yolov8n.pt
```

## Limitations

- LBPH is sensitive to lighting, pose, image quality, and camera distance
- This is not liveness detection and does not prevent photograph or video replay attacks
- Object-detection accuracy depends on the camera view and model confidence
- All processing is local; session events are not persisted
- The system requires real-world calibration before use in consequential decisions
