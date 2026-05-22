import os
import tempfile
import base64
import cv2
import gdown
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# =========================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Dashboard Deteksi Helm YOLOv10s",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. PATH FILE MODEL DAN FOTO PROFIL
# =========================================================
MODEL_PATH = "best.pt"
PROFILE_IMAGE_PATH = "profile.jpeg"

GOOGLE_DRIVE_FILE_ID = "1JhjScUvcVOKWdSOXYeloc2cCA-BXcJXp"

PROFILE_NAME = "Naufal Daffa Abdu Al Hafidl"
PROFILE_CAMPUS = "Universitas Gunadarma"


# =========================================================
# 3. FUNGSI FOTO PROFIL BASE64
# =========================================================
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# =========================================================
# 4. CSS / STYLE UI
# =========================================================
st.markdown("""
<style>

/* =========================================================
   JANGAN HIDE HEADER
   Karena tombol sidebar Streamlit ada di header
========================================================= */
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}

header {
    background: transparent !important;
    height: 3rem !important;
}

button[kind="header"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
/* HEADER TETAP ADA */
header {
    background: transparent !important;
    height: 3rem !important;
}

.stApp {
    background: #eef3f8;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.3rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #f3f7fb;
    border-right: 1px solid #dbe3ee;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #172033;
    line-height: 1.25;
    margin-bottom: 10px;
}

.main-description {
    color: #66758a;
    font-size: 16px;
    line-height: 1.7;
    max-width: 900px;
    margin-bottom: 42px;
}

.profile-wrapper {
    text-align: center;
    margin-top: 8px;
    margin-bottom: 34px;
}

.profile-img {
    width: 158px;
    height: 158px;
    border-radius: 50%;
    background-size: 116%;
    background-position: center 18%;
    background-repeat: no-repeat;
    border: 4px solid #ffffff;
    box-shadow: 0px 10px 28px rgba(43, 65, 94, 0.18);
    margin: 0 auto 16px auto;
}

.profile-placeholder {
    width: 158px;
    height: 158px;
    border-radius: 50%;
    background: #2f6df6;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px auto;
    font-size: 40px;
    font-weight: 900;
    border: 4px solid #ffffff;
}

.profile-name {
    font-size: 16px;
    font-weight: 800;
    color: #172033;
    margin-bottom: 5px;
}

.profile-campus {
    font-size: 14px;
    color: #69788d;
    font-weight: 700;
}

.upload-card {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 12px 30px rgba(43, 65, 94, 0.07);
    margin-bottom: 28px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.92);
    min-height: 138px;
    padding: 22px 18px;
    border-radius: 18px;
    box-shadow: 0px 12px 30px rgba(43, 65, 94, 0.08);
    text-align: center;
    border: 1px solid #e2e8f0;
    position: relative;
}

.metric-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 22px;
    bottom: 22px;
    width: 5px;
    border-radius: 999px;
    background: #2f6df6;
}

.metric-title {
    min-height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    color: #66758a;
    font-weight: 700;
    line-height: 1.45;
}

.metric-value {
    font-size: 40px;
    font-weight: 900;
    color: #111827;
    margin-top: 8px;
}

.section-space {
    height: 28px;
}

.image-card {
    background: white;
    border-radius: 18px;
    padding: 0;
    overflow: hidden;
    box-shadow: 0px 12px 30px rgba(43, 65, 94, 0.08);
    border: 1px solid #e2e8f0;
}

.image-title {
    text-align: center;
    color: #66758a;
    font-weight: 700;
    padding: 13px 0 14px 0;
    font-size: 15px;
    background: white;
}

.result-card {
    background: #dcecff;
    color: #0757b8;
    border-radius: 18px;
    padding: 22px;
    font-size: 15.5px;
    line-height: 1.7;
    font-weight: 500;
}

.conclusion-card {
    background: #dcfce7;
    color: #166534;
    border-radius: 18px;
    padding: 22px;
    font-size: 15.5px;
    line-height: 1.7;
    font-weight: 600;
    margin-top: 18px;
}

.chart-card {
    background: rgba(255,255,255,0.92);
    border-radius: 20px;
    padding: 18px 12px 12px 12px;
    box-shadow: 0px 10px 24px rgba(15, 23, 42, 0.06);
    border: 1px solid #e2e8f0;
    height: 100%;
}

[data-testid="stImage"] {
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stImage"] img {
    display: block;
    border-radius: 0 !important;
}

[data-testid="stFileUploader"] {
    background: #f8fafc;
    border-radius: 14px;
    padding: 12px;
}

.sidebar-info {
    background: #dcecff;
    color: #0757b8;
    border-radius: 16px;
    padding: 18px;
    font-size: 14.5px;
    line-height: 1.65;
    font-weight: 600;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. LOAD MODEL YOLO
# =========================================================
def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

        with st.spinner("Sedang mengunduh model best.pt dari Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    return os.path.exists(MODEL_PATH)


@st.cache_resource
def load_model():
    if not download_model_from_drive():
        return None

    return YOLO(MODEL_PATH)


model = load_model()

if model is None:
    st.error("Model best.pt gagal diunduh dari Google Drive. Pastikan link Google Drive sudah Anyone with the link.")
    st.stop()


# =========================================================
# 6. SIDEBAR PROFIL DAN PENGATURAN
# =========================================================
with st.sidebar:
    st.markdown('<div class="profile-wrapper">', unsafe_allow_html=True)

    if os.path.exists(PROFILE_IMAGE_PATH):
        img_base64 = get_image_base64(PROFILE_IMAGE_PATH)

        st.markdown(
            f"""
<div
    class="profile-img"
    style="background-image: url('data:image/jpeg;base64,{img_base64}');">
</div>
""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="profile-placeholder">ND</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
<div class="profile-name">{PROFILE_NAME}</div>
<div class="profile-campus">{PROFILE_CAMPUS}</div>
""",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.title("Pengaturan")

    menu = st.radio(
        "Pilih Jenis Deteksi",
        ["Deteksi Gambar", "Deteksi Video"]
    )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="sidebar-info">
    Class helm/helmet dihitung sebagai pengendara tertib,
    sedangkan non_helm/non_helmet dihitung sebagai
    pengendara tidak tertib.
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# 7. HEADER DASHBOARD
# =========================================================
st.markdown(
    '<div class="main-title">Dashboard Deteksi Penggunaan Helm<br>Menggunakan YOLOv10s</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">Sistem ini digunakan untuk mendeteksi dan menghitung pengendara tertib serta tidak tertib berdasarkan penggunaan helm.</div>',
    unsafe_allow_html=True
)


# =========================================================
# 8. FUNGSI CLASS
# =========================================================
def is_helmet_class(class_name):
    class_name = class_name.lower().strip()
    return class_name in ["helm", "helmet"]


def is_non_helmet_class(class_name):
    class_name = class_name.lower().strip()
    return class_name in [
        "non_helm",
        "non_helmet",
        "non helm",
        "non-helm",
        "no_helmet",
        "no helmet",
        "no-helmet",
        "tanpa_helm",
        "tanpa helm",
        "tanpa-helm"
    ]


# =========================================================
# 9. METRIC CARD
# =========================================================
def show_metrics(helmet_count, non_helmet_count):
    total = helmet_count + non_helmet_count

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(f"""
<div class="metric-card">
    <div class="metric-title">Pengendara Menggunakan<br>Helm</div>
    <div class="metric-value">{helmet_count}</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="metric-card">
    <div class="metric-title">Pengendara Tidak<br>Menggunakan Helm</div>
    <div class="metric-value">{non_helmet_count}</div>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
<div class="metric-card">
    <div class="metric-title">Total Terdeteksi</div>
    <div class="metric-value">{total}</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 10. HITUNG DETEKSI GAMBAR
# =========================================================
def count_image_results(results):
    helmet_count = 0
    non_helmet_count = 0

    names = results[0].names
    boxes = results[0].boxes

    if boxes is not None and boxes.cls is not None:
        for cls in boxes.cls:
            class_name = names[int(cls)]

            if is_helmet_class(class_name):
                helmet_count += 1
            elif is_non_helmet_class(class_name):
                non_helmet_count += 1

    return helmet_count, non_helmet_count


# =========================================================
# 11. HALAMAN DETEKSI GAMBAR
# =========================================================
if menu == "Deteksi Gambar":
    st.subheader("Deteksi Menggunakan Gambar")

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

        with st.spinner("Sedang melakukan deteksi gambar..."):
            results = model.predict(
                image,
                conf=confidence,
                verbose=False
            )

            result_image = results[0].plot()
            helmet_count, non_helmet_count = count_image_results(results)

        show_metrics(helmet_count, non_helmet_count)

        st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('<div class="image-title">Gambar Input</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(result_image, use_container_width=True)
            st.markdown('<div class="image-title">Hasil Deteksi</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

        if non_helmet_count > 0:
            st.warning(f"Terdapat {non_helmet_count} pengendara tidak tertib yang terdeteksi.")
        else:
            st.success("Tidak terdapat pengendara tidak tertib pada gambar.")

    else:
        show_metrics(0, 0)
        st.info("Silakan upload gambar terlebih dahulu.")


# =========================================================
# 12. HALAMAN DETEKSI VIDEO
# =========================================================
elif menu == "Deteksi Video":
    st.subheader("Deteksi Menggunakan Video")

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())
        temp_file.close()

        video_path = temp_file.name
        st.video(video_path)

        if st.button("Mulai Deteksi Video"):
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Video gagal dibuka.")
                st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            frame_box = st.empty()
            metric_box = st.empty()
            progress_bar = st.progress(0)

            helmet_count = 0
            non_helmet_count = 0

            helmet_ids = set()
            non_helmet_ids = set()

            with st.spinner("Sedang memproses video..."):
                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.track(
                        frame,
                        conf=confidence,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    annotated_frame = results[0].plot()
                    boxes = results[0].boxes
                    names = results[0].names

                    helmet_count = 0
                    non_helmet_count = 0

                    if boxes is not None and boxes.cls is not None:
                        for idx, cls in enumerate(boxes.cls):
                            class_name = names[int(cls)]

                            if boxes.id is not None:
                                track_id = int(boxes.id[idx])
                            else:
                                track_id = f"{frame_count}_{idx}_{class_name}"

                            if is_helmet_class(class_name):
                                helmet_count += 1
                                helmet_ids.add(track_id)

                            elif is_non_helmet_class(class_name):
                                non_helmet_count += 1
                                non_helmet_ids.add(track_id)

                    annotated_rgb = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    frame_box.image(
                        annotated_rgb,
                        channels="RGB",
                        use_container_width=True
                    )

                    with metric_box.container():
                        show_metrics(helmet_count, non_helmet_count)

                    frame_count += 1

                    if total_frames > 0:
                        progress_bar.progress(
                            min(frame_count / total_frames, 1.0)
                        )

            cap.release()

            final_helmet = len(helmet_ids)
            final_non_helmet = len(non_helmet_ids)
            final_total = final_helmet + final_non_helmet

            helmet_percent = (final_helmet / final_total * 100) if final_total > 0 else 0
            non_helmet_percent = (final_non_helmet / final_total * 100) if final_total > 0 else 0

            st.success("Deteksi video selesai dilakukan.")

            result_col1, result_col2 = st.columns([1.25, 0.85], gap="large")

            with result_col1:
                result_html = f"""
<div class="result-card">
    <b>Hasil Akhir Deteksi Video</b><br><br>
    Total pengendara menggunakan helm: <b>{final_helmet}</b><br><br>
    Total pengendara tidak menggunakan helm: <b>{final_non_helmet}</b><br><br>
    Total seluruh pengendara terdeteksi: <b>{final_total}</b><br><br>
    Persentase pengendara tertib: <b>{helmet_percent:.2f}%</b><br>
    Persentase pengendara tidak tertib: <b>{non_helmet_percent:.2f}%</b>
</div>
"""
                st.markdown(result_html, unsafe_allow_html=True)

            with result_col2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(3.2, 3.2))

                colors = ["#2563eb", "#ef4444"]

                ax.pie(
                    [final_helmet, final_non_helmet],
                    labels=["Tertib", "Tidak Tertib"],
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=colors,
                    wedgeprops=dict(width=0.42, edgecolor="white"),
                    textprops=dict(
                        color="#334155",
                        fontsize=10,
                        fontweight="semibold"
                    )
                )

                ax.text(
                    0,
                    0,
                    f"{helmet_percent:.1f}%\nTertib",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="#0f172a"
                )

                ax.set_aspect("equal")
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")

                st.pyplot(fig, use_container_width=True)

                st.markdown("""
<div style="
    text-align:center;
    margin-top:-10px;
    color:#64748b;
    font-size:14px;
    font-weight:600;
">
    Diagram Kepatuhan Penggunaan Helm
</div>
""", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            if helmet_percent >= non_helmet_percent:
                conclusion_text = (
                    f"Berdasarkan hasil deteksi video, mayoritas pengendara di Kota Bogor "
                    f"tergolong tertib berlalu lintas karena {helmet_percent:.2f}% "
                    f"pengendara terdeteksi menggunakan helm. Namun, masih terdapat "
                    f"{non_helmet_percent:.2f}% pengendara yang tidak tertib karena tidak "
                    f"menggunakan helm, sehingga tetap diperlukan peningkatan kesadaran "
                    f"keselamatan berkendara."
                )
            else:
                conclusion_text = (
                    f"Berdasarkan hasil deteksi video, mayoritas pengendara di Kota Bogor "
                    f"tergolong tidak tertib berlalu lintas karena {non_helmet_percent:.2f}% "
                    f"pengendara terdeteksi tidak menggunakan helm. Kondisi ini menunjukkan "
                    f"perlunya peningkatan pengawasan dan edukasi keselamatan berkendara."
                )

            conclusion_html = f"""
<div class="conclusion-card">
    <b>Kesimpulan:</b><br><br>
    {conclusion_text}
</div>
"""
            st.markdown(conclusion_html, unsafe_allow_html=True)

    else:
        show_metrics(0, 0)
        st.info("Silakan upload video terlebih dahulu.")