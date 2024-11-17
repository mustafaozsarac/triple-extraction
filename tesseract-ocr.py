import cv2
import pytesseract
import re
import numpy as np

# Set up Tesseract path (Uncomment and adjust path)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

def extract_text(image_path):
    """Extracts text and their positions from the image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Applying binary threshold to separate text
    _, binary_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Detect text boxes
    data = pytesseract.image_to_data(binary_img, output_type=pytesseract.Output.DICT)
    
    text_positions = []
    for i, word in enumerate(data['text']):
        if word.strip():
            # Store text and its bounding box position
            text_positions.append({
                'text': word.strip(),
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
    return text_positions

def detect_relations(text_positions, threshold=50):
    """Identify relations based on text proximity and direction."""
    triples = []
    for i, subj in enumerate(text_positions):
        for j, obj in enumerate(text_positions):
            if i != j:
                # Calculate vertical and horizontal distance
                x_distance = abs(subj['left'] - obj['left'])
                y_distance = abs(subj['top'] - obj['top'])
                
                if y_distance < threshold:  # Likely horizontal relation
                    if subj['left'] < obj['left']:
                        triples.append((subj['text'], "has", obj['text']))
                elif x_distance < threshold:  # Likely vertical relation
                    if subj['top'] < obj['top']:
                        triples.append((subj['text'], "has", obj['text']))
    return triples

def process_image(image_path):
    """Processes the image to extract and format triples."""
    text_positions = extract_text(image_path)
    triples = detect_relations(text_positions)
    return triples

# Example usage
if __name__ == "__main__":
    image_path = 'test1.png'  # replace with your image file
    triples = process_image(image_path)
    print("Extracted triples:")
    for triple in triples:
        print(f"{triple[0]} {triple[1]} {triple[2]}")
