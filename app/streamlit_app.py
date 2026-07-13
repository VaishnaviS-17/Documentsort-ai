import streamlit as st
import torch
from PIL import Image
import sys
import os
from datetime import datetime
from fpdf import FPDF

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_custom import DocumentCNN
from model_pretrained import get_resnet18_model
from dataset import eval_transform
from routing import predict_with_confidence_routing
from gradcam import GradCAM, overlay_heatmap

CLASS_NAMES = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="DocuSort AI", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
section[data-testid="stSidebar"] {display: none;}

.stApp { background-color: #0f1115; }
html, body, [class*="css"] { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; }

.block-container { padding: 1.5rem 3rem 3rem 3rem; max-width: 100%; }

.navbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 0 1.2rem 0; margin-bottom: 3rem;
}
.navbar-logo { font-size: 1.8rem; font-weight: 700; color: #f1f3f7; letter-spacing: -0.02em; }
.navbar-links a {
    color: #97a1b3; font-size: 1.3rem; text-decoration: none; margin-left: 1.8rem;
    transition: color 0.15s ease;
}
.navbar-links a:hover { color: #2dd4bf; }

.hero-title { font-size: 2.6rem; font-weight: 800; color: #f1f3f7; letter-spacing: -0.03em; margin-bottom: 0.7rem; text-align: center; }
.hero-subtitle { font-size: 1.2rem; color: #8891a3; max-width: 660px; margin: 0 auto; line-height: 1.6; text-align: center; }

.stat-row { display: flex; justify-content: center; gap: 3rem; margin-top: 2.2rem; }
.stat-item { text-align: center; }
.stat-num { font-size: 1.8rem; font-weight: 700; color: #2dd4bf; }
.stat-label { font-size: 0.98rem; color: #6b7284; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

.how-section { padding: 3rem 0; margin: 1rem 0; }
.how-heading { font-size: 1.2rem; font-weight: 700; color: #6b7284; text-transform: uppercase; letter-spacing: 0.08em; text-align: center; margin-bottom: 1.8rem; }
.step-card { background-color: #171a21; border: 1px solid #232733; border-radius: 10px; padding: 1.3rem; text-align: center; height: 100%; }
.step-num { font-size: 0.98rem; font-weight: 700; color: #2dd4bf; margin-bottom: 0.5rem; }
.step-title { font-size: 1.1rem; font-weight: 600; color: #e5e8ef; margin-bottom: 0.4rem; } 
.step-desc { font-size: 0.92rem; color: #7d8598; line-height: 1.5; }

.section-heading { font-size: 0.8rem; font-weight: 700; color: #7d8598; margin: 0 0 0.8rem 0; text-transform: uppercase; letter-spacing: 0.08em; }

.toolbar { background-color: #171a21; border: 1px solid #232733; border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem; }

div[role="radiogroup"] { flex-direction: row !important; gap: 0.6rem; }
div[role="radiogroup"] label { background-color: #1c1f28; border: 1px solid #2c303c; border-radius: 6px; padding: 6px 14px; }

section[data-testid="stFileUploaderDropzone"] { background-color: #171a21; border: 1.5px dashed #2c303c; border-radius: 10px; }
section[data-testid="stFileUploaderDropzone"]:hover { border-color: #2dd4bf; }
section[data-testid="stFileUploaderDropzone"] button { background-color: #1c1f28; color: #e5e8ef; border: 1px solid #2c303c; border-radius: 6px; }
section[data-testid="stFileUploaderDropzone"] small { color: #6b7284 !important; }

.result-card { background-color: #171a21; border: 1px solid #232733; border-radius: 10px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.status-ok { color: #4ade80; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em; }
.status-review { color: #fbbf24; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em; }
.predicted-label { font-size: 1.8rem; font-weight: 700; color: #f1f3f7; margin: 0.4rem 0; }
.meta-line { color: #565e70; font-size: 0.8rem; margin-top: 1rem; padding-top: 0.8rem; }

.stDownloadButton button { background-color: #2dd4bf; color: #0f1115; border: none; border-radius: 6px; font-weight: 600; width: 100%; }
.stDownloadButton button:hover { background-color: #5eead4; color: #0f1115; }

.app-footer { text-align: center; color: #4b5163; font-size: 0.78rem; padding: 5rem 0 0rem 0; }
</style> 
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_custom_cnn():
    model = DocumentCNN(num_classes=10)
    model.load_state_dict(torch.load('../saved_models/best_custom_cnn.pth', map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_resnet18():
    model = get_resnet18_model(num_classes=10, freeze_backbone=True)
    model.load_state_dict(torch.load('../saved_models/best_resnet18.pth', map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def get_target_layer(model, model_choice):
    if model_choice == "Custom CNN":
        return model.block4.conv
    return model.layer4[-1].conv2


def generate_pdf_report(image_pil, overlayed_gradcam, result, model_choice, threshold):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "DocuSort AI - Classification Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Model used: {model_choice}", ln=True)
    pdf.cell(0, 7, f"Predicted class: {result['predicted_class']}", ln=True)
    pdf.cell(0, 7, f"Confidence: {result['confidence']:.1%}", ln=True)
    status = "Flagged for manual review (low confidence)" if result["needs_manual_review"] else "High confidence"
    pdf.cell(0, 7, f"Status: {status}  (threshold: {threshold:.0%})", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Top 3 Predictions:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for cls, prob in result["top3_predictions"]:
        pdf.cell(0, 6, f"  - {cls}: {prob:.1%}", ln=True)
    pdf.ln(4)

    tmp_orig = "_tmp_orig.jpg"
    tmp_cam = "_tmp_cam.jpg"
    image_pil.convert("RGB").save(tmp_orig)
    Image.fromarray(overlayed_gradcam).save(tmp_cam)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Visual Reference", ln=True)
    img_w = 85
    pdf.image(tmp_orig, x=10, y=pdf.get_y(), w=img_w)
    pdf.image(tmp_cam, x=105, y=pdf.get_y(), w=img_w)
    pdf.ln(img_w * 1.25)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Left: original document   |   Right: Grad-CAM attention overlay", ln=True)

    os.remove(tmp_orig)
    os.remove(tmp_cam)
    return bytes(pdf.output())


def build_top3_html(top3_predictions):
    parts = []
    for cls, prob in top3_predictions:
        parts.append(
            '<div style="display:flex; justify-content:space-between; margin:4px 0; font-size:0.88rem; color:#d5d9e2;">'
            f'<span>{cls}</span><span>{prob:.1%}</span></div>'
            '<div style="background:#2c303c; border-radius:4px; height:6px; margin-bottom:8px;">'
            f'<div style="background:#2dd4bf; width:{prob*100:.1f}%; height:6px; border-radius:4px;"></div></div>'
        )
    return "".join(parts)


def build_result_card_html(result, model_choice):
    status_html = (
        '<div class="status-review">Flagged for manual review</div>'
        if result["needs_manual_review"]
        else '<div class="status-ok">High confidence</div>'
    )
    top3_html = build_top3_html(result["top3_predictions"])
    return (
        '<div class="result-card">' + status_html
        + f'<div class="predicted-label">{result["predicted_class"]}</div>'
        + f'<div style="color:#8891a3; font-size:0.9rem; margin-bottom:1rem;">{result["confidence"]:.1%} confidence</div>'
        + top3_html
        + f'<div class="meta-line">Model: {model_choice} &nbsp;|&nbsp; Device: {DEVICE}</div>'
        + '</div>'
    )


st.markdown(
    '<div class="navbar">'
    '<div class="navbar-logo">DocuSort AI</div>'
    '<div class="navbar-links">'
    '<a href="#model-section">Model</a>'
    '<a href="#how-it-works">How it Works</a>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="hero-title">Automated Document Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload a scanned document and get an instant classification, '
    'backed by a custom-trained CNN and a fine-tuned ResNet18, with visual model interpretability '
    'and confidence-based routing for uncertain cases.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="stat-row">'
    '<div class="stat-item"><div class="stat-num">10</div><div class="stat-label">Document Classes</div></div>'
    '<div class="stat-item"><div class="stat-num">2</div><div class="stat-label">Model Architectures</div></div>'
    '<div class="stat-item"><div class="stat-num">3,482</div><div class="stat-label">Training Images</div></div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="how-section" id="how-it-works">', unsafe_allow_html=True)
st.markdown('<div class="how-heading">How it works</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown(
        '<div class="step-card"><div class="step-num">STEP 1</div>'
        '<div class="step-title">Upload</div>'
        '<div class="step-desc">Drop in a scanned document image (JPG or PNG)</div></div>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        '<div class="step-card"><div class="step-num">STEP 2</div>'
        '<div class="step-title">Classify</div>'
        '<div class="step-desc">The model predicts the document type with a confidence score</div></div>',
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        '<div class="step-card"><div class="step-num">STEP 3</div>'
        '<div class="step-title">Review</div>'
        '<div class="step-desc">See what the model focused on, and export a full report</div></div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading" id="model-section">Configuration</div>', unsafe_allow_html=True)
st.markdown('<div class="toolbar">', unsafe_allow_html=True)
tc1, tc2, tc3 = st.columns([1.3, 1, 1])
with tc1:
    model_choice = st.radio("Model", ["Custom CNN", "ResNet18 (Transfer Learning)"], horizontal=True)
with tc2:
    confidence_threshold = st.slider("Review threshold", 0.3, 0.9, 0.6, 0.05)
with tc3:
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Upload Document</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop a file here, or click to browse",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('L')

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="section-heading">Document Preview</div>', unsafe_allow_html=True)
        st.image(image_pil, use_container_width=True)

    model = load_custom_cnn() if model_choice == "Custom CNN" else load_resnet18()

    input_tensor = eval_transform(image_pil).unsqueeze(0).to(DEVICE)
    result = predict_with_confidence_routing(
        model, input_tensor, CLASS_NAMES, confidence_threshold=confidence_threshold
    )

    target_layer = get_target_layer(model, model_choice)
    gradcam = GradCAM(model, target_layer)
    heatmap, predicted_idx = gradcam.generate(input_tensor.clone())
    overlayed = overlay_heatmap(image_pil, heatmap)

    with col2:
        st.markdown('<div class="section-heading">Result</div>', unsafe_allow_html=True)
        st.markdown(build_result_card_html(result, model_choice), unsafe_allow_html=True)

        pdf_bytes = generate_pdf_report(image_pil, overlayed, result, model_choice, confidence_threshold)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"docusort_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

    if show_gradcam:
        st.markdown('<div class="section-heading">Model Attention (Grad-CAM)</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.image(image_pil, caption="Original", use_container_width=True)
        with col4:
            st.image(overlayed, caption=f"Attention for: {CLASS_NAMES[predicted_idx]}", use_container_width=True)

else:
    st.markdown(
        '<div style="text-align:center; padding: 2.5rem 0; color:#7d8598;">'
        'Upload a scanned document image above to get started.</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="app-footer">DocuSort AI &middot; built with PyTorch, TorchVision, and Streamlit '
    '&middot; trained on the Tobacco3482 dataset</div>',
    unsafe_allow_html=True
)