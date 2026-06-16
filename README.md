# SolarVision AI

SolarVision AI is a lightweight, edge-ready web application designed to automate the physical inspection of solar panels. Developed as a Final Year Project at the University of Malaysia Sarawak (UNIMAS), this system transitions solar maintenance from an expensive, industrial luxury into an accessible, low-cost tool.

While traditional SCADA systems track electrical output and are blind to physical damage, SolarVision AI utilizes computer vision to instantly detect and localize structural issues. 

### Core Technology
* **AI Engine:** YOLOv11 Nano (YOLOv11n) for high-speed, accurate object detection and bounding box localization without the need for expensive GPUs.
* **Web Framework:** Streamlit, utilizing `@st.cache_resource` to lock the model into active RAM for millisecond response times.
* **Dataset:** Trained on a rigorously augmented dataset of 6,303 images, optimized to combat environmental glare and highly reflective glass surfaces.
* **Capabilities:** Classifies panels as **Clean**, **Dusty**, or **Damaged** via static image upload or live camera inference.
