import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from dotenv import load_dotenv
from openai import OpenAI
import io
import base64
import json

st.set_page_config(page_title="Deteksi Sampah Makanan - CNN + OpenCV Style", page_icon="♻️", layout="wide")

load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
if GROK_API_KEY:
    client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

st.title("♻️ SI-RAMA Apps")

st.markdown("""
**Food Waste Classification App**

Real-time detection using CNN to classify food waste:

- **Suitable for Composting** (yellow box)
- **Suitable for Digestion** (blue box)
- **Suitable for MFC** (green box)

Capture image → get instant result with bounding box and scientific explanation.
""")
# ===================== TRAINING / LOAD CNN =====================
MODEL_PATH = "waste_classifier.h5"


@st.cache_resource
def load_or_train_cnn():
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model CNN sudah siap")
        return tf.keras.models.load_model(MODEL_PATH)

    st.warning("🚀 Model Has been traied")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        "data",
        target_size=(224, 224),
        batch_size=8,
        subset="training",
        class_mode="categorical"
    )
    val_gen = datagen.flow_from_directory(
        "data",
        target_size=(224, 224),
        batch_size=8,
        subset="validation",
        class_mode="categorical"
    )

    base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, epochs=12, validation_data=val_gen, verbose=1)
    model.save(MODEL_PATH)
    st.success("✅ CNN was been trained!")
    return model


model = load_or_train_cnn()

# Mapping class (urutannya sesuai folder alphabetical: composting, digestion, mfc)
class_names = ["Composting", "Digestion", "MFC"]
colors = [(255, 215, 0), (0, 100, 255), (0, 220, 0)]  # kuning, biru, hijau

# ===================== CAMERA + DETEKSI =====================
camera_image = st.camera_input("📷 Put your camera on the food and take caputre **Capture**", key="cnn_camera")

if camera_image is not None:
    original_img = Image.open(camera_image).convert("RGB")

    # Preprocessing untuk CNN
    img_resized = original_img.resize((224, 224))
    img_array = np.array(img_resized).astype("float32")
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    # Prediksi CNN (real-time)
    with st.spinner("CNN during classification..."):
        prediction = model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx] * 100
        label = class_names[class_idx]
        display_label = f"Layak {label}"
        box_color = colors[class_idx]

    # Gambar kotak + label (seperti contoh gambar kamu)
    annotated = original_img.copy()
    draw = ImageDraw.Draw(annotated)

    w, h = annotated.size
    draw.rectangle([(20, 20), (w - 20, h - 20)], outline=box_color, width=18)

    try:
        font = ImageFont.truetype("arial.ttf", 65)
    except:
        font = ImageFont.load_default()

    draw.text((60, 60), display_label, fill=box_color, font=font, stroke_width=6, stroke_fill=(0, 0, 0))

    # Tampilan
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(annotated, caption=f"✅ {display_label} (Confidence: {confidence:.1f}%)", use_column_width=True)
    with col2:
        if label == "MFC":
            st.success(f"**{display_label}**")
        elif label == "Digestion":
            st.info(f"**{display_label}**")
        else:
            st.warning(f"**{display_label}**")

        st.markdown("**Color-coded bounding box appears automatically** (green = MFC, blue = Digestion, yellow = Composting)")

    # ===================== PENJELASAN DARI GROK (paling canggih) =====================
    if GROK_API_KEY:
        with st.spinner("Analyzing and explaining results..."):
            try:
                buffered = io.BytesIO()
                annotated.save(buffered, format="JPEG")
                b64 = base64.b64encode(buffered.getvalue()).decode()

                response = client.chat.completions.create(
                    model="grok-4-1-fast-reasoning",  # Model paling canggih xAI (support vision)
                    messages=[
                        {"role": "system",
                         "content": "Berikan penjelasan ilmiah + rumus matematis singkat dalam bahasa Indonesia."},
                        {"role": "user", "content": [
                            {"type": "text",
                             "text": f"Gambar ini diklasifikasikan sebagai {label}. Jelaskan kenapa cocok untuk {label} (MFC / Digestion / Composting). Sertakan rumus seperti Power = V×I, biogas yield, atau decomposition rate."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]}
                    ],
                    temperature=0.2,
                    max_tokens=600
                )
                penjelasan = response.choices[0].message.content.strip()
                st.markdown("### 📖 Scientific & Mathematical Explanation ")
                st.write(penjelasan)
            except Exception as e:
                st.info("Powered by local AI model • Optional scientific insights via API.")
    else:
        st.info("Your API not valid")

st.caption(
    "✅ Powered by CNN • Instant Detection • One-time Training")