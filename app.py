import os
import mimetypes
import sqlite3
import base64
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types
from io import BytesIO

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DB_FILE = "chat_history.db"

st.set_page_config(page_title="AI Image & Text Generator", layout="wide")

# --- Database setup ---
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response_text TEXT,
    images TEXT
)
''')
conn.commit()

# --- Database helper functions ---
def save_chat(prompt, response_text, images):
    images_serialized = base64.b64encode("\n".join(images).encode("utf-8")).decode("utf-8") if images else None
    c.execute("INSERT INTO chat_history (prompt, response_text, images) VALUES (?, ?, ?)",
              (prompt, response_text, images_serialized))
    conn.commit()

def load_chats(search_term=None):
    if search_term:
        c.execute("SELECT * FROM chat_history WHERE prompt LIKE ? ORDER BY id DESC", (f"%{search_term}%",))
    else:
        c.execute("SELECT * FROM chat_history ORDER BY id DESC")
    rows = c.fetchall()
    chats = []
    for row in rows:
        images = []
        if row[3]:
            decoded = base64.b64decode(row[3]).decode("utf-8")
            images = decoded.split("\n")
        chats.append({"id": row[0], "prompt": row[1], "text": row[2], "images": images})
    return chats

def clear_history():
    c.execute("DELETE FROM chat_history")
    conn.commit()

# --- Image helpers ---
def convert_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def convert_base64_to_bytes(data):
    return base64.b64decode(data)

# --- Generate AI content ---
def generate_content(prompt_text, image_bytes=None, mime_type=None):
    client = genai.Client(api_key=API_KEY)
    parts = [types.Part(text=prompt_text)]
    if image_bytes:
        parts.append(types.Part(inline_data=types.Blob(mime_type=mime_type or "image/png", data=image_bytes)))

    contents = [types.Content(parts=parts)]
    config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    images = []
    collected_text = []

    for chunk in client.models.generate_content_stream(model="gemini-2.0-flash-exp", contents=contents, config=config):
        if not chunk.candidates:
            continue
        candidate = chunk.candidates[0]
        content = candidate.content
        if not content or not content.parts:
            continue
        part = content.parts[0]

        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            data_buffer = part.inline_data.data
            images.append(convert_image_to_base64(data_buffer))
        elif hasattr(part, "text") and part.text:
            collected_text.append(part.text)

    return collected_text, images

# --- Sidebar ---
st.sidebar.header("Chat History")
search_term = st.sidebar.text_input("Search chats")

if st.sidebar.button("Clear Chat History"):
    clear_history()
    st.session_state.selected_chat_id = None
    st.session_state.active_tab = 0  # Go to Generate tab

# Load chats
chats = load_chats(search_term)
chat_map = {chat["id"]: chat for chat in chats}

# --- Session state defaults ---
if "selected_chat_id" not in st.session_state:
    st.session_state.selected_chat_id = chats[0]["id"] if chats else None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # default to Generate tab

# --- Sidebar: select chat ---
if chats:
    selected_label = st.sidebar.radio(
        "Select a chat",
        [f"{chat['id']}: {chat['prompt'][:50]}..." if len(chat['prompt']) > 50 else f"{chat['id']}: {chat['prompt']}" for chat in chats],
        index=0 if st.session_state.selected_chat_id is None else next(
            (i for i, c in enumerate(chats) if c["id"] == st.session_state.selected_chat_id), 0)
    )
    st.session_state.selected_chat_id = int(selected_label.split(":")[0])
    st.session_state.active_tab = 3  # Switch to Chat Detail tab

# --- Main tabs ---
tab_titles = ["Generate", "Caption Image", "Image Variations", "Chat Detail"]
tabs = st.tabs(tab_titles)
active_tab = st.session_state.active_tab

# --- Tab 0: Generate ---
with tabs[0]:
    st.subheader("Generate AI Image & Text")
    prompt = st.text_area("Enter prompt:", "Generate image of people gathered around for bonfire")
    uploaded_file = st.file_uploader("Upload reference image (optional)", type=["png", "jpg", "jpeg"])

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("Please enter a prompt!")
        elif not API_KEY:
            st.error("API key not found. Check your .env file.")
        else:
            image_bytes = uploaded_file.read() if uploaded_file else None
            mime_type = uploaded_file.type if uploaded_file else None
            collected_text, images = generate_content(prompt, image_bytes, mime_type)

            if collected_text:
                st.subheader("Generated Text:")
                st.write(" ".join(collected_text))

            if images:
                st.subheader("Generated Images:")
                for i, data in enumerate(images):
                    st.image(BytesIO(convert_base64_to_bytes(data)), caption=f"image_{i}.png")
                    st.download_button("Download image", convert_base64_to_bytes(data),
                                       file_name=f"image_{i}.png", mime="image/png")

            save_chat(prompt, " ".join(collected_text) if collected_text else None, images)
            st.success("Chat saved!")

# --- Tab 1: Image Captioning ---
with tabs[1]:
    st.subheader("Image Captioning")
    caption_file = st.file_uploader("Upload image to caption", type=["png", "jpg", "jpeg"], key="caption_file")
    if caption_file and st.button("Generate Caption"):
        image_bytes = caption_file.read()
        mime_type = caption_file.type if caption_file else "image/png"
        caption_text, _ = generate_content("Describe this image", image_bytes, mime_type)
        if caption_text:
            st.write("**Caption:**", " ".join(caption_text))
        else:
            st.warning("Could not generate caption.")

# --- Tab 2: Image Variations ---
with tabs[2]:
    st.subheader("Image Variations")
    variation_file = st.file_uploader("Upload image to create variations", type=["png", "jpg", "jpeg"], key="variation_file")
    if variation_file and st.button("Generate Variations"):
        image_bytes = variation_file.read()
        mime_type = variation_file.type if variation_file else "image/png"
        variation_text, variation_images = generate_content("Create variations of this image", image_bytes, mime_type)
        if variation_images:
            st.subheader("Variations:")
            for i, data in enumerate(variation_images):
                st.image(BytesIO(convert_base64_to_bytes(data)), caption=f"variation_{i}.png")
                st.download_button(f"Download variation_{i}", convert_base64_to_bytes(data),
                                   file_name=f"variation_{i}.png", mime="image/png")
        else:
            st.warning("No variations generated.")

# --- Tab 3: Chat Detail ---
with tabs[3]:
    st.subheader("Chat Detail")
    selected_id = st.session_state.selected_chat_id
    chat = chat_map.get(selected_id)

    if chat:
        st.markdown(f"**Prompt:** {chat['prompt']}")
        if chat["text"]:
            st.markdown(f"**Response:** {chat['text']}")

        if chat["images"]:
            st.subheader("Images:")
            cols = st.columns(3)  # 3 images per row
            for idx, data in enumerate(chat["images"]):
                with cols[idx % 3]:
                    st.image(BytesIO(convert_base64_to_bytes(data)), caption=f"image_{idx}.png")
                    st.download_button(f"Download image_{idx}", convert_base64_to_bytes(data),
                                       file_name=f"chat_{selected_id}_image_{idx}.png", mime="image/png")

        # Regenerate button
        if st.button("Regenerate Images/Text"):
            image_bytes_list = [convert_base64_to_bytes(img) for img in chat["images"]] if chat["images"] else [None]
            regenerated_text = []
            regenerated_images = []

            for img_bytes in image_bytes_list:
                text, images = generate_content(chat["prompt"], img_bytes)
                if text:
                    regenerated_text.extend(text)
                if images:
                    regenerated_images.extend(images)

            st.success("Regeneration complete!")
            if regenerated_text:
                st.write(" ".join(regenerated_text))

            if regenerated_images:
                cols = st.columns(3)
                for idx, img_data in enumerate(regenerated_images):
                    with cols[idx % 3]:
                        st.image(BytesIO(convert_base64_to_bytes(img_data)), caption=f"regenerated_{idx}.png")
                        st.download_button(f"Download regenerated_{idx}", convert_base64_to_bytes(img_data),
                                           file_name=f"chat_{selected_id}_regenerated_{idx}.png", mime="image/png")

            save_chat(chat["prompt"], " ".join(regenerated_text) if regenerated_text else None, regenerated_images)
    else:
        st.info("Select a chat from sidebar to view details.")
