import os
import io
import base64
import tempfile
import torch
import numpy as np
import streamlit as st
import ollama
import fitz  # PyMuPDF
from PIL import Image
import clip
from sentence_transformers import SentenceTransformer

class EnhancedDocumentQASystem:
    def __init__(self, max_pages=150, dpi=300):
        self.image_embeddings = []
        self.text_embeddings = []
        self.images = []
        self.texts = []
        self.text_model = None
        self.image_model = None
        self.image_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.uploaded_files = []
        self.max_pages = max_pages
        self.dpi = dpi

    def image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def validate_pdf(self, file):
        try:
            file.seek(0)
            file_size = len(file.read())
            file.seek(0)  # Reset file pointer for further reading

            if file_size == 0:
                st.warning(f"PDF is empty: {file.name}")
                return False

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            pdf_doc = fitz.open(temp_file_path)
            if pdf_doc.page_count == 0:
                st.warning(f"PDF has no readable pages: {file.name}")
                return False
            pdf_doc.close()
            os.unlink(temp_file_path)
            return True
        except Exception as e:
            st.error(f"Error validating PDF {file.name}: {e}")
            return False

    def process_file_input(self, files):
        st.info("Processing files...")
        self.uploaded_files = []
        
        if not files:
            st.error("No files uploaded. Please upload PDF files.")
            return []
        
        valid_files = []
        for file in files:
            st.text(f"Checking file: {file.name}")
            
            if self.validate_pdf(file):
                valid_files.append(file)
                self.uploaded_files.append(file)
            else:
                st.warning(f"Skipping invalid or non-PDF file: {file.name}")
        
        if not valid_files:
            st.error("No valid PDF files found. Please upload valid PDF files.")
            return []
        
        st.success(f"Valid files: {', '.join([f.name for f in valid_files])}")
        return valid_files

    def convert_files(self, files):
        images = []
        texts = []
        for file in files:
            try:
                # Check if file has content
                file.seek(0)
                content = file.read()
                if not content:
                    st.warning(f"File {file.name} is empty. Skipping.")
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                pdf_doc = fitz.open(temp_file_path)
                for page_number in range(min(pdf_doc.page_count, self.max_pages)):
                    page = pdf_doc[page_number]
                    # Extract image
                    pix = page.get_pixmap(dpi=self.dpi)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    images.append(img)
                    # Extract text
                    texts.append(page.get_text())
                
                pdf_doc.close()
                os.unlink(temp_file_path)
            except Exception as e:
                st.error(f"Error converting PDF {file.name}: {e}")
                st.warning(f"Skipping corrupted or unreadable file: {file.name}")
                continue

        if len(images) > self.max_pages:
            images = images[:self.max_pages]
            texts = texts[:self.max_pages]
            st.warning(f"Truncated to first {self.max_pages} pages")
        
        return images, texts

    def load_models(self):
        try:
            # Load ViT-B/32 for image processing
            self.image_model, self.image_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Load sentence-transformers model for text processing
            self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.text_model = self.text_model.to(self.device)
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            raise

    def index_documents(self, files):
        try:
            self.image_embeddings = []
            self.text_embeddings = []
            self.images = []
            self.texts = []

            if not files:
                return "No files provided for indexing."

            self.images, self.texts = self.convert_files(files)
            
            if not self.images or not self.texts:
                return "No valid pages found. Please upload proper PDF files."

            if self.image_model is None or self.text_model is None:
                self.load_models()

            progress_bar = st.progress(0)
            total_pages = len(self.images)

            # Process images
            for i, img in enumerate(self.images):
                with torch.no_grad():
                    image = self.image_preprocess(img).unsqueeze(0).to(self.device)
                    image_features = self.image_model.encode_image(image)
                    self.image_embeddings.append(image_features.cpu().numpy())
                progress_bar.progress((i + 1) / (total_pages * 2))

            # Process texts
            for i, text in enumerate(self.texts):
                text_features = self.text_model.encode(text)
                self.text_embeddings.append(text_features)
                progress_bar.progress((total_pages + i + 1) / (total_pages * 2))

            torch.cuda.empty_cache()

            return f"Uploaded and processed {len(self.images)} pages"
        except Exception as e:
            torch.cuda.empty_cache()
            return f"Error indexing documents: {str(e)}"

    def search_documents(self, query, k=1):
        try:
            if not self.image_embeddings or not self.text_embeddings:
                return "No documents indexed. Please upload PDFs first."

            # Process query
            with torch.no_grad():
                text_features = self.text_model.encode(query)
                image_features = self.image_model.encode_text(clip.tokenize([query]).to(self.device))

            # Normalize embeddings
            text_features = text_features / np.linalg.norm(text_features)
            image_features = image_features.cpu().numpy() / np.linalg.norm(image_features.cpu().numpy())

            # Calculate similarities
            text_similarities = np.dot(self.text_embeddings, text_features)
            image_similarities = np.dot(self.image_embeddings, image_features.T).flatten()

            # Combine similarities (you can adjust the weights if needed)
            combined_similarities = 0.5 * text_similarities + 0.5 * image_similarities

            top_k_indices = combined_similarities.argsort()[-k:][::-1]

            results = [self.images[idx] for idx in top_k_indices]
            return results
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    def get_answer_with_vision(self, images, query):
        try:
            if not images:
                return "No images found. Please index documents first."

            formatted_images = [
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{self.image_to_base64(image)}"
                    }
                } for image in images
            ]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        *formatted_images
                    ]
                }
            ]

            response = ollama.chat(
                model='llama3.1-vision', 
                messages=messages,
                options={
                    'temperature': 0.7,
                    'max_tokens': 512
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error in Ollama inference: {str(e)}"

def display_most_similar_document(query):
    if not st.session_state.qa_system.image_embeddings or not st.session_state.qa_system.text_embeddings:
        st.error("No documents indexed. Please upload and index PDFs first.")
        return

    # Generate text features for the query
    with torch.no_grad():
        text_features = st.session_state.qa_system.text_model.encode(query)

    # Normalize embeddings
    text_features = text_features / np.linalg.norm(text_features)

    # Calculate text similarities
    text_similarities = np.dot(st.session_state.qa_system.text_embeddings, text_features)

    # Find the most similar document
    most_similar_idx = text_similarities.argmax()
    most_similar_score = text_similarities[most_similar_idx]

    # Display the most similar document
    st.write(f"Most similar document (score: {most_similar_score:.4f}):")
    
    # Get the matched image (PIL Image)
    matched_image = st.session_state.qa_system.images[most_similar_idx]

    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    matched_image.save(buffered, format="PNG")
    matched_image_bytes = buffered.getvalue()

    # Encode the image to base64
    encoded_image = base64.b64encode(matched_image_bytes).decode('utf-8')
    
    # Display the matched image
    st.image(matched_image_bytes, use_column_width=True)
    
    # Display the associated text
    st.write(f"Associated text: {st.session_state.qa_system.texts[most_similar_idx]}")

    # Get AI response using Ollama
    st.write("AI Response:")
    response_container = st.empty()

    try:
        # Send the query to Ollama's vision model and stream the response
        with st.spinner('⏳ Generating Response...'):
            stream = ollama.chat(
                model="llama3.2-vision",
                messages=[
                    {
                        'role': 'user',
                        'content': f"Please answer the following question based on the image: {query}",
                        'images': [encoded_image]  # Pass the base64 string directly here
                    }
                ],
                stream=True
            )

            collected_chunks = []
            for chunk in stream:
                chunk_content = chunk['message']['content']
                collected_chunks.append(chunk_content)
                complete_response = ''.join(collected_chunks)
                response_container.markdown(complete_response)
                
    except Exception as e:
        st.error(f"Error generating AI response: {e}")

def main():
    st.set_page_config(page_title="Enhanced Document QA System", page_icon="📚", layout="wide")
    st.title("📚 Enhanced Automated Document QA System")

    # Initialize session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = EnhancedDocumentQASystem()

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("🔄 Index Documents"):
            with st.spinner("Processing and indexing documents..."):
                valid_files = st.session_state.qa_system.process_file_input(uploaded_files)
                if valid_files:
                    result = st.session_state.qa_system.index_documents(valid_files)
                    st.success(result)
                else:
                    st.error("No valid files to index.")

    # Use a form for query input to prevent page refresh on enter
    with st.form(key='query_form'):
        query = st.text_input("Enter your query")
        k = st.slider("Number of results", min_value=1, max_value=10, value=1)
        submit_button = st.form_submit_button(label="🔍 Search & Answer")

    if submit_button:
        if not st.session_state.qa_system.image_embeddings or not st.session_state.qa_system.text_embeddings:
            st.error("No documents indexed. Please upload and index PDFs first.")
        elif not query:
            st.error("Please enter a query.")
        else:
            with st.spinner("Searching documents and generating answer..."):
                results = st.session_state.qa_system.search_documents(query, k)
                display_most_similar_document(query)

if __name__ == "__main__":
    main()
