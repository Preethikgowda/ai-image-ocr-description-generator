
import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from transformers import pipeline
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="GenAI OCR Pipeline",
    page_icon="🔍",
    layout="wide"
)

# Initialize image captioning pipeline (load only once)
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Function to extract text from image
def extract_text_from_image(img):
    try:
        if img.mode != 'L':
            img = img.convert('L')  # Convert to grayscale

        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error in text extraction: {str(e)}")
        return ""

# Function to extract text from video
def extract_text_from_video(video_path, frames_to_skip=30):
    try:
        cap = cv2.VideoCapture(video_path)
        all_text = []
        frame_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            progress = frame_count / total_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {frame_count} of ~{total_frames}...")

            if frame_count % frames_to_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                text = pytesseract.image_to_string(pil_img)
                if text.strip():
                    all_text.append(text)

            frame_count += 1

        cap.release()
        progress_bar.empty()
        status_text.empty()
        return "\n".join(all_text)
    except Exception as e:
        st.error(f"Error in video processing: {str(e)}")
        return ""

# ✅ Updated Function to generate image description
def generate_image_description(img):
    try:
        captioner = load_caption_model()
        # 🚨 Fix: Pass PIL image directly (not bytes)
        result = captioner(img)
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Error in generating description: {str(e)}")
        return ""

# Main function to process input
def process_input(file_path, input_type, get_description=False):
    try:
        if input_type == 'image':
            img = Image.open(file_path)

            st.image(img, caption="Uploaded Image", use_column_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Extracted Text")
                text = extract_text_from_image(img)
                st.text_area("OCR Results", text, height=300)

            description = None
            if get_description:
                with col2:
                    st.subheader("Image Description")
                    with st.spinner("Generating description..."):
                        description = generate_image_description(img)
                    st.success("Description generated!")
                    st.text_area("Description", description, height=300)

            return text, description

        elif input_type == 'video':
            st.warning("Video processing may take some time depending on length...")
            video_file = open(file_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            text = extract_text_from_video(file_path)
            st.subheader("Extracted Text from Video")
            st.text_area("OCR Results", text, height=400)

            return text, None
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return "", ""

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    st.subheader("OCR Configuration")

    languages = {
        "English": "eng",
        "Spanish": "spa",
        "French": "fra",
        "German": "deu",
        "Chinese": "chi_sim",
        "Japanese": "jpn",
        "Russian": "rus"
    }

    selected_lang = st.selectbox("Select Language", list(languages.keys()), index=0)

    # ✅ Update Tesseract paths here
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

    st.subheader("Video Processing")
    frames_to_skip = st.slider("Process every N frames", 1, 60, 30)

# Main UI
st.title("🔍 GenAI OCR Pipeline")
st.markdown("Upload an image or video to extract text using OCR, or get an AI-generated description of images.")

uploaded_file = st.file_uploader(
    "Choose an image or video file",
    type=["png", "jpg", "jpeg", "bmp", "mp4", "mov", "avi"],
    accept_multiple_files=False
)

get_description = st.checkbox(
    "Generate image description (for images only)",
    value=False,
    help="Uses AI to describe the content of the image"
)

if uploaded_file is not None:
    file_type = 'image' if uploaded_file.type.startswith('image') else 'video'

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        # Optional: pass config to tesseract (advanced)
        custom_config = f'--tessdata-dir "{os.environ.get("TESSDATA_PREFIX", "")}" -l {languages[selected_lang]}'
        pytesseract.pytesseract.tesseract_cmd = pytesseract.pytesseract.tesseract_cmd

        with st.spinner(f"Processing {file_type}..."):
            extracted_text, description = process_input(
                temp_path,
                file_type,
                get_description=get_description and file_type == 'image'
            )

        if extracted_text:
            st.download_button(
                label="Download Extracted Text",
                data=extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

        if description:
            st.download_button(
                label="Download Image Description",
                data=description,
                file_name="image_description.txt",
                mime="text/plain"
            )
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass
