import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# 1. Load the trained AI model
@st.cache_resource
def load_model():
    return YOLO("weights.pt")

model = load_model()

# 2. UI Setup
st.set_page_config(page_title="SolarVision AI", layout="centered")

st.title("SolarVision AI")
st.write("Upload an image or use your camera to detect solar panel conditions.")

input_method = st.radio(
    "Select Input Method:",
    ("Upload Image", "Open Camera")
)

image = None

# 3. Image Input Handling
if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a solar panel image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif input_method == "Open Camera":

    camera_photo = st.camera_input(
        "Take a picture of the solar panel"
    )

    if camera_photo is not None:
        image = Image.open(camera_photo)

# 4. AI Inference and Display Results
if image is not None:

    st.divider()
    st.subheader("Results")

    # Display original image
    st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Analyzing panel condition..."):

        # Run YOLO prediction
        results = model(image, conf=0.10)

        # Get first result
        result = results[0]

        # Draw bounding boxes
        annotated_image = result.plot()

        # Convert BGR to RGB
        annotated_image_rgb = annotated_image[:, :, ::-1]

        # Display processed image
        st.image(
            annotated_image_rgb,
            caption="Processed Image",
            use_container_width=True
        )

        # Check detections
        if len(result.boxes) > 0:

            # Get highest confidence detection
            confidences = result.boxes.conf.cpu().numpy()
            best_index = np.argmax(confidences)

            best_box = result.boxes[best_index]

            # Extract prediction info
            class_id = int(best_box.cls.item())
            class_name = model.names[class_id].upper()

            confidence = float(best_box.conf.item()) * 100

            # Dynamic status color
            if class_name == "CLEAN":
                color = "green"

            elif class_name == "DUSTY":
                color = "orange"

            else:
                color = "red"

            # Display status
            st.markdown(
                f"### Status: <span style='color:{color}'>{class_name}</span>",
                unsafe_allow_html=True
            )

            # Display confidence
            st.markdown(
                f"### Confidence: {confidence:.2f}%"
            )

        else:
            st.warning("No panel or defect detected.")

        # Convert image for download
        success, buffer = cv2.imencode(".png", annotated_image)

        if success:

            # Download button BELOW status/confidence
            st.download_button(
                label="Save Processed Image",
                data=buffer.tobytes(),
                file_name="solarvision_result.png",
                mime="image/png"
            )