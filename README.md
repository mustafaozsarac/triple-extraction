# triple-extraction

This script extracts the triples, stores them in triples, prints out the triples, and then creates the graph. You can zoom in and out on the graph to see the relations in detail.

## Features

1. **OpenAI-Powered Data Extraction Using Langchain** (`llmcall.py`):
   - Integrates with OpenAI API (via LangChain) for structured data extraction.
   - Encodes images in base64 format for seamless processing.
   - Uses a pipeline approach to process and visualize relationships using `networkx` and `matplotlib`.
   - Works good.

2. **OCR and Text Extraction** (`tesseract-ocr.py`):
   - Uses Tesseract OCR for extracting text and bounding box information from images.
   - Processes images with OpenCV for optimal text detection.
   - Doesn't work very well.

## Installation
* For tesseract-ocr.py script, you'll need to install the following libraries if they’re not already available:
```bash
pip install opencv-python pytesseract
```
* You will also need to have Tesseract OCR installed on your machine. Here’s the Tesseract OCR installation guide.
* For the llmcall.py script, you need to have these libraries:
```bash
pip install langchain-core langchain-openai pydantic networkx matplotlib
```
* Also, you need to provide an OpenAI API Key.

## Dependencies

- Python 3.8+
- OpenCV
- Tesseract OCR
- LangChain
- NetworkX
- Matplotlib