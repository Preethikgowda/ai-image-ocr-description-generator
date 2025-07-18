# ai-image-ocr-description-generator


This project is a GenAI-powered application that extracts text from images and generates intelligent descriptions of the image content. It combines Optical Character Recognition (OCR) using Tesseract with image captioning models to enhance document understanding, accessibility, and data extraction capabilities.

---

## 📌 Features

- 🔍 **OCR**: Extracts readable text from uploaded image files using Tesseract.
- 🧠 **AI-Powered Description**: Uses Hugging Face's `transformers` pipeline to generate human-like image descriptions.
- 🖼️ **Image Upload Interface**: Easy-to-use interface built with Streamlit.
- 💬 **Dual Output**: Displays both raw text and descriptive summary side-by-side.

---

## 🛠️ Tech Stack

| Component        | Technology Used                      |
|------------------|--------------------------------------|
| Frontend         | Streamlit (Python-based Web UI)      |
| OCR Engine       | Tesseract OCR                        |
| Image Handling   | OpenCV, PIL                          |
| Image Captioning | Hugging Face Transformers (Vision Encoder Decoder Pipeline) |
| Environment      | Python 3.10+                         |

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Preethikgowda/ai-image-ocr-description-generator.git
cd ai-image-ocr-description-generator
2. Set Up Virtual Environment
python -m venv venv
venv\Scripts\activate     # Windows
3. Install Dependencies
pip install -r requirements.txt
4. Run the App
streamlit run genai_ocr_app.py

📝 Requirements
 requirements.txt

Essential libraries include:
streamlit
transformers
pytesseract
opencv-python
Pillow
torch
sentencepiece
numpy

📂 Project Structure

├── genai_ocr_app.py         # Main Streamlit app
├── venv/                    # Virtual environment (excluded from Git)
├── .gitignore               # Files/folders to ignore
├── requirements.txt         # Project dependencies
└── README.md                # Project overview

📷 Sample Output
Extracted Text: Displays printed or handwritten text from image.
Image Description: AI-generated summary like "A group of people standing near a whiteboard."

🧠 Use Cases
Digitizing physical documents

Assisting visually impaired users

Enhancing data capture workflows

Automating document analysis

🙋‍♀️ Author
Preethi T K
preethikgowda26@gmail.com

📄 License
This project is licensed under the MIT License - feel free to use and modify.
