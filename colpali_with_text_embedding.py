import streamlit as st
import fitz  # PyMuPDF
import io
import tempfile
from PIL import Image
import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from sentence_transformers import SentenceTransformer
import base64
import gc
import ollama

# Initialize session state
if 'images' not in st.session_state:
    st.session_state.images = []
    st.session_state.image_embeddings = []
    st.session_state.text_embeddings = []
    st.session_state.texts = []

# Function to get device type (cuda, mps, or cpu)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device_map = get_device()

# Load models
@st.cache_resource
def load_models():
    # Load text and image models here
    image_model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    image_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    text_model = text_model.to(device_map)
    return image_model, image_processor, text_model

# Process image to get embeddings
def process_image(image, processor, model):
    batch_images = processor.process_images([image]).to(model.device)
    with torch.no_grad():
        image_embeddings = model(**batch_images)
    return image_embeddings[0].cpu().to(torch.float32).numpy()

# Process text to get embeddings
def process_text(text, model):
    return model.encode(text)

# Function to extract images and text from PDF using PyMuPDF
def extract_images_and_text_from_pdf(pdf_file):
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
        # Extract image (convert the page to pixmap and then image)
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

        # Extract text
        text = page.get_text()
        texts.append(text)

    return images, texts

# Function to clear cache
def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

# Main function for the Streamlit app
def main():
    st.title("📷 Image and Text RAG System")

    image_model, image_processor, text_model = load_models()

    st.header("Add and Query PDFs")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == 'application/pdf':
                try:
                    # Extract images and text from the PDF
                    images, texts = extract_images_and_text_from_pdf(uploaded_file)

                    st.success(f"Successfully processed PDF: {uploaded_file.name}")

                    for i, (img, text) in enumerate(zip(images, texts)):
                        # Process the image to get embeddings
                        image_embedding = process_image(img, image_processor, image_model)

                        # Process the text to get embeddings
                        text_embedding = process_text(text, text_model)

                        # Store data in session state
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        st.session_state.images.append(img_str)
                        st.session_state.image_embeddings.append(image_embedding)
                        st.session_state.text_embeddings.append(text_embedding)
                        st.session_state.texts.append(text)

                        st.success(f"Processed page {i+1}")

                except Exception as e:
                    st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")

        # Clear cache after processing
        clear_cache()

    # Display indexed images
    if st.session_state.images:
        st.subheader("Indexed Images")
        cols = st.columns(4)
        for i, img_str in enumerate(st.session_state.images):
            with cols[i % 4]:
                st.image(base64.b64decode(img_str), use_column_width=True, caption=f"Image {i+1}")

    # Query section
    st.subheader("Query Images and Text")
    query = st.text_input("Enter your query")

    if query and st.session_state.images:
        # Embed query image (dummy image for query)
        image_query_embedding = process_image(Image.new('RGB', (1, 1)), image_processor, image_model)
        
        # Embed query text
        text_query_embedding = process_text(query, text_model)

        # Adjust dimensions for image embeddings
        image_embeddings_adjusted = []
        for embedding in st.session_state.image_embeddings:
            embedding_mean = np.mean(embedding, axis=0)
            image_embeddings_adjusted.append(embedding_mean)

        image_embeddings_adjusted = np.array(image_embeddings_adjusted)
        text_embeddings = np.array(st.session_state.text_embeddings)

        # Ensure query embeddings have the same shape
        image_query_embedding = np.mean(image_query_embedding, axis=0)

        # Compute similarity scores
        image_similarities = np.dot(image_embeddings_adjusted, image_query_embedding)
        text_similarities = np.dot(text_embeddings, text_query_embedding)

        # Normalize similarities
        image_similarities = (image_similarities - np.min(image_similarities)) / (np.max(image_similarities) - np.min(image_similarities))
        text_similarities = (text_similarities - np.min(text_similarities)) / (np.max(text_similarities) - np.min(text_similarities))

        # Combine similarities
        combined_similarities = 0.5 * image_similarities + 0.5 * text_similarities

        # Get the most similar result
        most_similar_idx = np.argmax(combined_similarities)
        most_similar_score = combined_similarities[most_similar_idx]

        st.write(f"Most similar document (score: {most_similar_score:.4f}):")
        matched_image = base64.b64decode(st.session_state.images[most_similar_idx])
        st.image(matched_image)
        st.write(f"Associated text: {st.session_state.texts[most_similar_idx]}")

        # Get AI response using ollama (optional)
        st.write("AI Response:")
        response_container = st.empty()

        try:
            with st.spinner('⏳ Generating Response...'):
                stream = ollama.chat(
                    model="llama3.2-vision",
                    messages=[
                        {
                            'role': 'user',
                            'content': f"Please answer the following question based on the image. {query}",
                            'images': [matched_image]
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
        st.warning("No images in the index. Please add images first.")

if __name__ == "__main__":
    main()
