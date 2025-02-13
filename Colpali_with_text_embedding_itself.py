import streamlit as st
import torch
import numpy as np
import base64
import gc
import io
import fitz  # PyMuPDF
import io
import tempfile
from PIL import Image
from pdf2image import convert_from_bytes
from colpali_engine.models import ColQwen2, ColQwen2Processor
import ollama
from transformers.utils.import_utils import is_flash_attn_2_available

# Initialize session state
if 'images' not in st.session_state:
    st.session_state.images = []
    st.session_state.texts = []
    st.session_state.image_embeddings = []
    st.session_state.text_embeddings = []

# Function to get device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device_map = get_device()

# Load ColQwen2 model
@st.cache_resource
def load_model():
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
    return model, processor

model, processor = load_model()

# Function to process images
def process_image(image):
    batch_images = processor.process_images([image]).to(model.device)
    with torch.no_grad():
        image_embedding = model(**batch_images)
    return image_embedding[0].cpu().to(torch.float32).numpy()

# Function to process text
def process_text(text):
    batch_text = processor.process_queries([text]).to(model.device)
    with torch.no_grad():
        text_embedding = model(**batch_text)
    return text_embedding[0].cpu().to(torch.float32).numpy()

# Function to clear GPU cache
def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

# Extract images and text from PDFs
def extract_images_and_text(pdf_file):
    images = []
    texts = []

    # Reset pointer to the beginning of the file
    pdf_file.seek(0)
    content = pdf_file.read()
    if not content:
        st.warning("Uploaded PDF file is empty. Skipping.")
        return [], []

    # Use a temporary file to handle the PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    # Open the PDF with PyMuPDF
    pdf_doc = fitz.open(temp_file_path)

    # Extract images and text from the PDF pages
    for page_number in range(min(pdf_doc.page_count, 5)):  # You can change this limit
        page = pdf_doc[page_number]
        
        try:
            # Check if page has embedded images
            img_list = page.get_images(full=True)
            if img_list:
                for img_index, img in enumerate(img_list):
                    xref = img[0]
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(image_bytes))
                    images.append(img)
                    st.success(f"Extracted image from page {page_number + 1} (image {img_index + 1})")
            
            else:
                # If no embedded images, create a pixmap of the page
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
                st.warning(f"No embedded images on page {page_number + 1}. Used page rendering instead.")
            
            # Extract text
            text = page.get_text()
            texts.append(text)
        
        except Exception as e:
            # Log the issue with image extraction
            st.warning(f"Could not extract image or process page {page_number + 1}: {e}")
            continue

    return images, texts

# Function to pad/truncate embeddings to fixed length
def adjust_embedding_length(embedding, fixed_length=128):
    seq_len, emb_dim = embedding.shape
    if seq_len < fixed_length:
        padding = np.zeros((fixed_length - seq_len, emb_dim), dtype=embedding.dtype)
        adjusted_embedding = np.concatenate([embedding, padding], axis=0)
    elif seq_len > fixed_length:
        adjusted_embedding = embedding[:fixed_length, :]
    else:
        adjusted_embedding = embedding
    return np.mean(adjusted_embedding, axis=0)  # Convert to single vector

# Main Streamlit App
def main():
    st.title("📷 Multimodal RAG System (Hybrid Search)")

    st.header("Upload PDFs for Indexing")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                images, texts = extract_images_and_text(uploaded_file)
                st.success(f"Processed PDF: {uploaded_file.name}")

                for i, (image, text) in enumerate(zip(images, texts)):
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                    image_embedding = process_image(image)
                    text_embedding = process_text(text)

                    # Adjust embeddings to fixed length
                    image_embedding_fixed = adjust_embedding_length(image_embedding)
                    text_embedding_fixed = adjust_embedding_length(text_embedding)

                    st.session_state.images.append(img_str)
                    st.session_state.texts.append(text)
                    st.session_state.image_embeddings.append(image_embedding_fixed)
                    st.session_state.text_embeddings.append(text_embedding_fixed)

                    st.success(f"Indexed Page {i+1}")

            except Exception as e:
                st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")

        clear_cache()

    # Display Indexed Images
    if st.session_state.images:
        st.subheader("Indexed Pages")
        cols = st.columns(4)
        for i, img_str in enumerate(st.session_state.images):
            with cols[i % 4]:
                st.image(base64.b64decode(img_str), use_column_width=True, caption=f"Page {i+1}")

    # Query Section
    st.subheader("Search PDF Contents")
    query = st.text_input("Enter your query")

    if query and st.session_state.images:
        query_text_embedding = process_text(query)
        query_image_embedding = process_image(Image.new('RGB', (1, 1)))  # Placeholder image

        # Adjust query embeddings
        query_text_embedding_fixed = adjust_embedding_length(query_text_embedding)
        query_image_embedding_fixed = adjust_embedding_length(query_image_embedding)

        # Convert embeddings to numpy arrays
        image_embeddings_array = np.array(st.session_state.image_embeddings)
        text_embeddings_array = np.array(st.session_state.text_embeddings)

        # Compute cosine similarities
        image_similarities = np.dot(image_embeddings_array, query_image_embedding_fixed)
        text_similarities = np.dot(text_embeddings_array, query_text_embedding_fixed)

        # Normalize similarities
        image_similarities = (image_similarities - np.min(image_similarities)) / (np.max(image_similarities) - np.min(image_similarities))
        text_similarities = (text_similarities - np.min(text_similarities)) / (np.max(text_similarities) - np.min(text_similarities))

        # Hybrid Search: Combine similarities (50% weight for both)
        combined_similarities = 0.5 * image_similarities + 0.5 * text_similarities

        # Retrieve best match
        best_match_idx = np.argmax(combined_similarities)
        best_match_score = combined_similarities[best_match_idx]

        st.write(f"Most relevant page (score: {best_match_score:.4f}):")
        best_match_image = base64.b64decode(st.session_state.images[best_match_idx])
        st.image(best_match_image)
        st.write(f"Extracted text: {st.session_state.texts[best_match_idx]}")

        # Use Ollama for AI-generated response
        st.write("AI Response:")
        response_container = st.empty()

        try:
            with st.spinner('⏳ Generating Response...'):
                stream = ollama.chat(
                    model="llama3.2-vision",
                    messages=[
                        {
                            'role': 'user',
                            'content': f"Please answer the following question based on the provided image. {query}",
                            'images': [best_match_image]
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
            st.error(f"Error getting AI response: {str(e)}")

        clear_cache()

    elif not st.session_state.images:
        st.warning("No PDFs indexed. Please upload a document first.")

if __name__ == "__main__":
    main()
