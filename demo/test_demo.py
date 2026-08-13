import os
import cv2

def run_demo(model_choice):
    """
    Run the function to start the demo
    """
    processor, model, reader = load_model_processor_and_reader()
    
    instructions = [
                "You are an OCR engine. You do not chat, explain, or comment.",
                "Locate all text in the image and transcribe it exactly as it appears.",
                "Output ONLY the raw transcribed text. Nothing else.",
                "Do NOT add any introduction (e.g. 'Here is the text...').",
                "Do NOT add any conclusion, offer, or question (e.g. 'Let me know if...').",
                "Do NOT use Markdown formatting: no bullet points, no bold (**), no headers (#), no asterisks.",
                "Do NOT interpret, label, or reformat the content — just transcribe what is visually present, line by line.",
                "Your entire response must be the transcription and nothing else.",
            ]
    
    predictions = {}
    
    for image_name in os.listdir("resources/images"):
        if image_name.endswith(".png") or image_name.endswith(".jpg"):
            image_path = os.path.join("resources/images", image_name)
            image = cv2.imread(image_path)
            
            if model_choice == "1":
                result = infer_with_glm(processor, model, image, instructions)
                list_of_words = extract_content(result)
                predictions[image_name] = list_of_words
                print(f"Predictions for {image_name} using GLM-OCR:")
                print(result)
                print("\n")
            elif model_choice == "2":
                result = infer_with_easyocr(reader, image)
                results_list = easyocr_to_word_list(result)
                list_of_words_with_conf = [(word, conf) for word, conf in results_list]
                predictions[image_name] = list_of_words_with_conf
                print(f"Predictions for {image_name} using EasyOCR:")
                for word, conf in list_of_words_with_conf:
                    print(f"Text: {word}, Confidence: {conf}%")
                print("\n")

def extract_content(glm_result):
    """
    Extracts the content from the GLM result.
    """
    words = []
    for line in glm_result.splitlines():
        line = line.strip()
        if not line:
            continue
        words.extend(line.split())
    return words
            
def easyocr_to_word_list(easyocr_result):
    """
    Convertit le résultat d'EasyOCR en une liste de mots avec leurs coordonnées et leur confiance.
    """
    word_list = []
    for (bbox, text, prob) in easyocr_result:
        conf = round(float(prob)*100, 4)
        text = str(text).strip()
        word_list.append((text, conf))
    return word_list

def load_model_processor_and_reader():
    """
    Charge le modèle, le processeur et le lecteur OCR.
    """
    processor, model, reader = None, None, None
    if model_choice == "1":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch
        # Regarde si le dossier "resources/models/GLM-OCR" existe
        model_dir = "resources/models/GLM-OCR"
        if not os.path.exists(model_dir):
            # On doit télécharger le modèle depuis Hugging Face
            print("Downloading GLM-OCR model from Hugging Face...")
            model = AutoModelForImageTextToText.from_pretrained("zai-org/GLM-OCR")
            processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")
            print("Saving GLM-OCR model to resources/models/GLM-OCR...")
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
        else:
            print("Loading GLM-OCR model from resources/models/GLM-OCR...")
            model = AutoModelForImageTextToText.from_pretrained(model_dir)
            processor = AutoProcessor.from_pretrained(model_dir)
        reader = None  # Pas nécessaire pour GLM-OCR
    elif model_choice == "2":
        import easyocr
        processor = None  # Pas nécessaire pour EasyOCR
        model = None  # Pas nécessaire pour EasyOCR
        reader = easyocr.Reader(['en'], gpu=False)  # Initialisation du lecteur EasyOCR
    
    return processor, model, reader

def infer_with_glm(processor, model, image, instructions):
    messages = [{"role": "user","content": [{"type": "image", "image": image},{"type": "text", "text": "".join(instructions)},]}]
    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=256)
    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True)

def infer_with_easyocr(reader, image):
    return reader.readtext(image, detail=1, paragraph=False)

if __name__ == "__main__":
    model_choice = input("Choose the model to run the demo (1 for GLM-OCR, 2 for EasyOCR): ")
    if model_choice == "1":
        print("Running demo with GLM-OCR...")
    elif model_choice == "2":
        print("Running demo with EasyOCR...")
    run_demo(model_choice)