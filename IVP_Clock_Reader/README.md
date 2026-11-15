
# IVP 2026 Analogue Clock Reader

BCS2710A group project. The goal is to automatically read the time from analogue clock images using classical image processing.

Part 1 is complete. Parts 2 and 3 still need to be implemented.

## Setup

Python 3.10 or newer is recommended.

Run these commands from the project folder:

```powershell
py -m pip install -r requirements.txt
py -m clock_reader path/to/clock.jpg --out preview
py -m unittest discover -s tests -v
```

The preview folder contains the processed images and detection information.

## Part 1: Clock Detection and Preprocessing

Implemented by Mehmet Levent Koca in `clock_reader/preprocessing.py`.

1. Convert the input image to grayscale and improve contrast.
2. Detect the clock face using Hough circles and contours.
3. Select the most likely clock face and estimate its center and radius.
4. Automatically crop and resize the clock to 512 x 512 pixels.
5. Generate grayscale, enhanced, edge and mask images.
6. Return the processed images and clock geometry for Part 2.

This part does not detect clock hands or calculate the time.

## Available Functions

| Function | Input | Output |
|---|---|---|
| `prepare_clock_from_path(path)` | Image file path | `ClockRegion` |
| `prepare_clock(image)` | OpenCV BGR image | `ClockRegion` |
| `load_image(path)` | Image file path | BGR image |
| `draw_detection(image, clock)` | Original image and `ClockRegion` | Debug image |

## ClockRegion Outputs

Both `prepare_clock_from_path()` and `prepare_clock()` return a `ClockRegion` object.

| Field | Type | Description |
|---|---|---|
| `image_bgr` | uint8 (512,512,3) | Normalized color image |
| `gray` | uint8 (512,512) | Grayscale clock image |
| `enhanced` | uint8 (512,512) | Contrast-enhanced image |
| `edges` | uint8 (512,512) | Canny edge image |
| `face_mask` | uint8 (512,512) | Clock face mask, 255 inside |
| `center_xy` | tuple (x,y) | Clock center in normalized image |
| `radius_px` | float | Clock radius in normalized image |
| `source_center_xy` | tuple (x,y) | Clock center in original image |
| `source_radius_px` | float | Clock radius in original image |
| `source_bbox_xyxy` | tuple | Bounding box in original image |
| `found` | bool | Whether clock detection succeeded |
| `confidence` | float | Detection confidence score |
| `method` | str | Detection method used |
| `metadata()` | dict | Detection information |

All processed images use the same 512 x 512 coordinate system.

## Example Usage

```python
from clock_reader import prepare_clock_from_path

clock = prepare_clock_from_path("data/clock.jpg")

image = clock.enhanced
edges = clock.edges
mask = clock.face_mask

center = clock.center_xy
radius = clock.radius_px

if not clock.found:
    print("Clock detection failed, fallback used")
```

You can also use `prepare_clock(image)` when the image is already loaded with OpenCV.

## Part 2: Hand Detection

Create `clock_reader/hand_detection.py`.

Use the `ClockRegion` returned by Part 1.

Tasks:

1. Detect possible clock hands using image processing.
2. Filter unwanted lines from numbers, markings and backgrounds.
3. Determine the position, angle and length of each candidate hand.
4. Return the detected candidates for Part 3.

Expected function:

```python
from clock_reader.preprocessing import ClockRegion

def detect_hands(clock: ClockRegion):
    # Return candidate hands with angles, lengths and scores
    pass
```

Person 2 should agree on the returned data structure with Person 3.

## Part 3: Time Calculation and Evaluation

Create `clock_reader/time_reader.py`.

Tasks:

1. Receive the candidate hands from Part 2.
2. Identify the hour and minute hands.
3. Convert their angles into a 12-hour time.
4. Return the predicted time as `hh:mm`.
5. Evaluate predictions against the training data.
6. Generate the Kaggle submission CSV.

Expected function:

```python
def read_time(hand_candidates):
    # Return predicted time as "hh:mm"
    pass
```

## Project Structure

```text
clock_reader/
    __init__.py
    __main__.py
    preprocessing.py
    hand_detection.py       (Part 2)
    time_reader.py          (Part 3)

tests/
    test_preprocessing.py

report/
    template/

README.md
requirements.txt
.gitignore
```

## Collaboration

Each person should work on their own module and make regular GitHub commits.

Part 1 is implemented in `preprocessing.py`.

Person 2 should use the Part 1 functions without modifying the preprocessing module unless changes are agreed upon.

Person 3 should connect the modules and handle evaluation and submission.

All processing must be automatic. Machine learning, OCR and manual cropping are not allowed.

## Deadline

Code, report, presentation and Kaggle submission: November 30, 2026 at 22:00.

Presentation: December 1, 2026.
