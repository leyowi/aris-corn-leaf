# ARIS

**ARIS** � Design and Development of a Computer Vision-Based Monitoring System
for Zeamays (Corn) Leaf Disease Detection.

An AI-assisted autonomous agricultural rover that detects corn leaf diseases
(Corn Rust, Northern Corn Leaf Blight, Gray Leaf Spot, Healthy) in the field,
geo-tags detections via GPS, and reports them through a web dashboard.

## Pipeline

1. **Leaf detection** - locate each corn leaf in the frame.
2. **Leaf segmentation** - precise mask per detected leaf.
3. **Disease classification** - classify each
   segmented leaf crop.
4. **Baseline** - compared against the cascaded pipeline 
   above for the capstone paper.

## Setup

See `requirements.txt` (dev machine) and `requirements-pi5.txt`
(Raspberry Pi 5 onboard inference).