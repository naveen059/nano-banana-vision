import os
import mimetypes
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types
from io import BytesIO
import base64

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI Image & Text Generator", layout="wide")
st.title("🤖 Generate and edit images")

prompt = st.text_area("Enter your prompt:", "Generate image of people gathered around for bonfire")
uploaded_file = st.file_uploader("Upload an image to reference (optional, will convert to base64)", type=["png", "jpg", "jpeg"])

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    elif not API_KEY:
        st.error("API key not found. Check your .env file.")
    else:
        client = genai.Client(api_key=API_KEY)
        images = []
        collected_text = []
        file_index = 0

        parts = [types.Part(text=prompt)]

        if uploaded_file:
            image_bytes = uploaded_file.read()
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=uploaded_file.type,
                        data=image_bytes
                    )
                )
            )

        contents = [types.Content(parts=parts)]
        config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=config
        ):
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            content = candidate.content
            if not content or not content.parts:
                continue
            part = content.parts[0]

            if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                data_buffer = part.inline_data.data
                extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                images.append((f"image_{file_index}{extension}", data_buffer))
                file_index += 1
            elif hasattr(part, "text") and part.text:
                collected_text.append(part.text)

        if collected_text:
            st.subheader("Generated Text:")
            st.write(" ".join(collected_text))

        if images:
            st.subheader("Generated Images:")
            for name, data in images:
                st.image(BytesIO(data), caption=name)
                st.download_button("Download " + name, data, file_name=name, mime="image/png")

        if not images and not collected_text:
            st.warning("No output generated. Try a different prompt or check your API key.")
