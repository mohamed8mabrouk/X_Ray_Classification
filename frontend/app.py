import streamlit as st
import requests

st.set_page_config(
    page_title="Bone Fracture AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000/predict"

st.markdown('''
<style>
.stApp { background:#f5f7fb; }
[data-testid="stSidebar"] { background:#111827; }
[data-testid="stSidebar"] * { color:white !important; }

.hero {
    background:linear-gradient(135deg,#0f172a,#1e3a8a);
    padding:42px;
    border-radius:22px;
    margin-bottom:28px;
    color:white;
}
.hero h1 { font-size:42px; margin:0 0 12px 0; font-weight:800; }
.hero p { font-size:18px; margin:0; line-height:1.6; opacity:.9; }

.card {
    background:white;
    padding:28px;
    border-radius:18px;
    border:1px solid #e5e7eb;
    min-height:150px;
    box-shadow:0 5px 20px rgba(0,0,0,.05);
}
.card h3 { color:#111827; }
.card p { color:#6b7280; line-height:1.6; }

.result-fractured {
    background:#fef2f2;
    border:2px solid #ef4444;
    color:#991b1b;
    padding:25px;
    border-radius:18px;
    text-align:center;
    font-size:30px;
    font-weight:800;
}
.result-normal {
    background:#ecfdf5;
    border:2px solid #10b981;
    color:#065f46;
    padding:25px;
    border-radius:18px;
    text-align:center;
    font-size:30px;
    font-weight:800;
}
.warning {
    background:#fff7ed;
    border:1px solid #fed7aa;
    color:#9a3412;
    padding:16px;
    border-radius:12px;
    margin-top:25px;
}
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🦴 Bone Fracture AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Home", "X-Ray Analysis", "About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### TensorFlow CNN")
    st.caption("AI-powered X-Ray classification system")

if page == "Home":
    st.markdown('''
    <div class="hero">
        <h1>Bone Fracture AI</h1>
        <p>
            An AI-powered system that analyzes X-Ray images using
            a Convolutional Neural Network to classify bone fractures.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('''
        <div class="card">
            <h3>📤 Upload</h3>
            <p>Upload an X-Ray image directly using the supported image formats.</p>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown('''
        <div class="card">
            <h3>🤖 AI Analysis</h3>
            <p>The image is sent to the trained CNN through the FastAPI backend.</p>
        </div>
        ''', unsafe_allow_html=True)

    with c3:
        st.markdown('''
        <div class="card">
            <h3>📋 Prediction</h3>
            <p>The system returns the classification without displaying confidence.</p>
        </div>
        ''', unsafe_allow_html=True)

  

elif page == "X-Ray Analysis":
    st.markdown('''
    <div class="hero">
        <h1>🩻 X-Ray Analysis</h1>
        <p>Upload an X-Ray image and let the trained CNN classify it.</p>
    </div>
    ''', unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Upload X-Ray")
        uploaded_file = st.file_uploader(
            "Choose an X-Ray image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True
            )

    with right:
        st.markdown("### Analysis")

        if uploaded_file is None:
            st.info("Please upload an X-Ray image first.")
        else:
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                try:
                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                        )
                    }

                    with st.spinner("Analyzing X-Ray..."):
                        response = requests.post(
                            API_URL,
                            files=files,
                            timeout=120
                        )

                    if response.status_code == 200:
                        result = response.json()
                        label = result.get("label", "Unknown")

                        if label.lower() == "fractured":
                            st.markdown(
                                '<div class="result-fractured">🦴 Fractured</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="result-normal">✅ Not Fractured</div>',
                                unsafe_allow_html=True
                            )

                        processing_time = result.get("processing_time_ms")
                        if processing_time is not None:
                            st.success(
                                f"⏱️ Processing Time: {processing_time:.2f} ms"
                            )
                    else:
                        st.error(
                            f"Backend error {response.status_code}: {response.text}"
                        )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to FastAPI. "
                        "Make sure the backend is running on port 8000."
                    )
                except requests.exceptions.Timeout:
                    st.error("❌ The request took too long. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

else:
    st.markdown('''
    <div class="hero">
        <h1>About Bone Fracture AI</h1>
        <p>A deep-learning demonstration project for X-Ray classification.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="card">
        <h3>How it works</h3>
        <p>
            1. Upload an X-Ray image.<br><br>
            2. Streamlit sends the image to the FastAPI backend.<br><br>
            3. FastAPI sends the image to the TensorFlow CNN model.<br><br>
            4. The prediction is returned to the frontend.<br><br>
            5. The classification result is displayed.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    
