import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from docx import Document
from io import BytesIO

st.set_page_config(page_title="ECG AI Analysis", layout="wide")

# ------------------ TITLE ------------------
st.title("🫀 ECG AI Analysis System")
st.write("Upload an ECG image to analyze heart rate and cardiac condition")

# ------------------ PATIENT DETAILS ------------------
st.header("📋 Patient Information")

name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
date = st.date_input("Date")

st.markdown("---")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

# ------------------ DOC CREATION FUNCTION ------------------
def create_doc(report_data):
    doc = Document()

    doc.add_heading("ECG AI ANALYSIS SYSTEM", 0)

    doc.add_heading("Patient Details", 1)
    doc.add_paragraph(f"Name: {name}")
    doc.add_paragraph(f"Age: {age}")
    doc.add_paragraph(f"Gender: {gender}")
    doc.add_paragraph(f"Date: {date}")

    doc.add_heading("ECG Analysis Results", 1)
    doc.add_paragraph(f"Heart Rate: {report_data['heart_rate']} BPM")
    doc.add_paragraph(f"Heart Rate Category: {report_data['hr_category']}")
    doc.add_paragraph(f"ECG Status: {report_data['condition']}")
    doc.add_paragraph(f"Average RR Interval: {report_data['mean_rr']}")
    doc.add_paragraph(f"HRV: {report_data['hrv']}")
    doc.add_paragraph(f"HRV Status: {report_data['hrv_status']}")
    doc.add_paragraph(f"Signal Quality: {report_data['signal_quality']}")
    doc.add_paragraph(f"Risk Level: {report_data['risk_level']}")

    doc.add_heading("Created By", 1)
    doc.add_paragraph("Your Name / Team Name")

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return buffer

# ------------------ MAIN PROCESS ------------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("Uploaded ECG Image")
    st.image(img, use_container_width=True)

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    st.subheader("Preprocessed ECG Image")
    fig0, ax0 = plt.subplots()
    ax0.imshow(binary, cmap="gray")
    ax0.axis("off")
    st.pyplot(fig0)

    # Signal extraction
    h, w = binary.shape
    lead = binary[int(0.45*h):int(0.55*h), :]

    signal = []
    for col in range(lead.shape[1]):
        y = np.where(lead[:, col] > 0)[0]
        signal.append(np.mean(y) if len(y) else np.nan)

    signal = np.array(signal)
    signal = signal - np.nanmean(signal)
    signal = np.nan_to_num(signal)

    st.subheader("Extracted ECG Signal")
    fig1, ax1 = plt.subplots()
    ax1.plot(signal)
    ax1.set_title("ECG Signal")
    st.pyplot(fig1)

    # Peak detection
    peaks, _ = find_peaks(signal, distance=50, prominence=1)

    st.subheader("R-Peak Detection")
    fig2, ax2 = plt.subplots()
    ax2.plot(signal)
    ax2.plot(peaks, signal[peaks], "rx")
    ax2.set_title("Detected R-Peaks")
    st.pyplot(fig2)

    # ------------------ REPORT ------------------
    if len(peaks) >= 2:
        rr = np.diff(peaks)
        heart_rate = 60 / (np.mean(rr) / 100)

        mean_rr = np.mean(rr)
        sdnn = np.std(rr)

        if heart_rate < 60:
            condition = "Bradycardia"
            hr_category = "Low Heart Rate"
        elif heart_rate > 100:
            condition = "Tachycardia"
            hr_category = "High Heart Rate"
        else:
            condition = "Normal Sinus Rhythm"
            hr_category = "Normal Heart Rate"

        hrv_status = "Low HRV (Possible Stress)" if sdnn < 20 else "Normal HRV"
        signal_quality = "Good Signal Quality" if len(peaks) > 5 else "Moderate Signal Quality"
        risk_level = "Mild Risk (Stress Related)" if hrv_status != "Normal HRV" else "Low Risk"

        st.subheader("ECG AI Analysis Report")
        st.write(f"Heart Rate: {int(heart_rate)} BPM")
        st.write(f"Heart Rate Category: {hr_category}")
        st.write(f"ECG Status: {condition}")
        st.write(f"Average RR Interval: {round(mean_rr, 2)}")
        st.write(f"HRV: {round(sdnn, 2)}")
        st.write(f"HRV Status: {hrv_status}")
        st.write(f"Signal Quality: {signal_quality}")
        st.write(f"Risk Level: {risk_level}")

        # Save data for report
        report_data = {
            "heart_rate": int(heart_rate),
            "hr_category": hr_category,
            "condition": condition,
            "mean_rr": round(mean_rr, 2),
            "hrv": round(sdnn, 2),
            "hrv_status": hrv_status,
            "signal_quality": signal_quality,
            "risk_level": risk_level
        }

        st.info("Note: AI-based supportive analysis only. Not a medical diagnosis.")

        # ------------------ DOWNLOAD BUTTON ------------------
        doc_file = create_doc(report_data)

        st.download_button(
            label="📄 Download ECG Report",
            data=doc_file,
            file_name="ECG_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    else:
        st.warning("Not enough R-peaks detected to generate ECG report.")

else:
    st.info("Please upload an ECG image to start analysis.")





