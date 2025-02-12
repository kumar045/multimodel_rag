import streamlit as st
import torch
from PIL import Image
import io
import base64
import numpy as np
from colpali_engine.models import ColQwen2, ColQwen2Processor
import gc
import ollama
from pdf2image import convert_from_bytes
from io import BytesIO

# Initialize session state
if 'images' not in st.session_state:
    st.session_state.images = []
    st.session_state.embeddings = []

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device_map = get_device()

@st.cache_resource
def load_model():
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
    return model, processor

def process_image(image, processor, model):
    batch_images = processor.process_images([image]).to(model.device)
    with torch.no_grad():
        image_embeddings = model(**batch_images)
    return image_embeddings[0].cpu().to(torch.float32).numpy()

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def main():
    st.title("📷 Image RAG System (ColQwen2 + Llama Vision)")

    model, processor = load_model()

    st.header("Add and Query Images")

    # Image upload section
    uploaded_files = st.file_uploader("Upload Images or PDFs", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == 'application/pdf':
                # Convert PDF to images
                try:
                    pdf_images = convert_from_bytes(uploaded_file.read())
                    st.success(f"Successfully converted PDF: {uploaded_file.name} ({len(pdf_images)} pages)")
                    
                    # Process each page of the PDF
                    for i, image in enumerate(pdf_images):
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        embedding = process_image(image, processor, model)
                        
                        st.session_state.images.append(img_str)
                        st.session_state.embeddings.append(embedding)
                        st.success(f"Processed page {i+1} of {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
            else:
                # Process regular image files
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    
                    embedding = process_image(image, processor, model)
                    
                    st.session_state.images.append(img_str)
                    st.session_state.embeddings.append(embedding)
                    st.success(f"Processed image: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing image {uploaded_file.name}: {str(e)}")
        
        clear_cache()

    # Display indexed images
    if st.session_state.images:
        st.subheader("Indexed Images")
        cols = st.columns(4)
        for i, img_str in enumerate(st.session_state.images):
            with cols[i % 4]:
                st.image(base64.b64decode(img_str), use_column_width=True, caption=f"Image {i+1}")

    # Query section
    st.subheader("Query Images")
    query = st.text_input("Enter your query")
    
    if query and st.session_state.images:
        with torch.no_grad():
            batch_query = processor.process_queries([query]).to(model.device)
            query_embedding = model(**batch_query)
        query_embedding_cpu = query_embedding.cpu().to(torch.float32).numpy()[0]

        # Set fixed sequence length
        fixed_seq_len = 620  # Adjust based on your embeddings

        image_embeddings_list = []
        for embedding in st.session_state.embeddings:
            seq_len, embedding_dim = embedding.shape
            if seq_len < fixed_seq_len:
                padding = np.zeros((fixed_seq_len - seq_len, embedding_dim), dtype=embedding.dtype)
                embedding_fixed = np.concatenate([embedding, padding], axis=0)
            elif seq_len > fixed_seq_len:
                embedding_fixed = embedding[:fixed_seq_len, :]
            else:
                embedding_fixed = embedding
            image_embeddings_list.append(embedding_fixed)

        retrieved_image_embeddings = np.stack(image_embeddings_list)

        # Adjust query embedding
        seq_len_q, embedding_dim_q = query_embedding_cpu.shape
        if seq_len_q < fixed_seq_len:
            padding = np.zeros((fixed_seq_len - seq_len_q, embedding_dim_q), dtype=query_embedding_cpu.dtype)
            query_embedding_fixed = np.concatenate([query_embedding_cpu, padding], axis=0)
        elif seq_len_q > fixed_seq_len:
            query_embedding_fixed = query_embedding_cpu[:fixed_seq_len, :]
        else:
            query_embedding_fixed = query_embedding_cpu

        # Convert to tensors
        query_embedding_tensor = torch.from_numpy(query_embedding_fixed).to(model.device).unsqueeze(0)
        retrieved_image_embeddings_tensor = torch.from_numpy(retrieved_image_embeddings).to(model.device)

        # Compute similarity scores
        with torch.no_grad():
            scores = processor.score_multi_vector(query_embedding_tensor, retrieved_image_embeddings_tensor)
        scores_np = scores.cpu().numpy().flatten()

        most_similar_idx = np.argmax(scores_np)
        most_similar_score = scores_np[most_similar_idx]
        
        st.write(f"Most similar image (score: {most_similar_score:.4f}):")
        matched_image = base64.b64decode(st.session_state.images[most_similar_idx])
        st.image(matched_image)

        # Get AI response using ollama
        st.write("AI Response:")
        response_container = st.empty()
        
        try:
            with st.spinner('⏳ Generating Response...'):
                stream = ollama.chat(
                    model="llama3.2-vision",
                    messages=[
                        {
                            'role': 'user',
                            'content': "Please answer the following question using only the information visible in the provided image" 
                            " Do not use any of your own knowledge, training data, or external sources."
                            " Base your response solely on the content depicted within the image."
                            " If there is no relation with question and image," 
                            f" you can respond with 'Question is not related to image'.\nHere is the question: {query}",
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