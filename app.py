import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# PDF imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
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


# ------------------ PDF FUNCTION ------------------
def create_pdf(report_data):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content = []

    # Title
    content.append(Paragraph("<b>ECG AI ANALYSIS REPORT</b>", styles['Title']))
    content.append(Spacer(1, 20))

    # Patient Details
    content.append(Paragraph("<b>Patient Details</b>", styles['Heading2']))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Name: {name}", styles['Normal']))
    content.append(Paragraph(f"Age: {age}", styles['Normal']))
    content.append(Paragraph(f"Gender: {gender}", styles['Normal']))
    content.append(Paragraph(f"Date: {date}", styles['Normal']))

    content.append(Spacer(1, 20))

    # Results
    content.append(Paragraph("<b>ECG Results</b>", styles['Heading2']))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Heart Rate: {report_data['heart_rate']} BPM", styles['Normal']))
    content.append(Paragraph(f"Heart Rate Category: {report_data['hr_category']}", styles['Normal']))
    content.append(Paragraph(f"ECG Status: {report_data['condition']}", styles['Normal']))
    content.append(Paragraph(f"Average RR Interval: {report_data['mean_rr']}", styles['Normal']))
    content.append(Paragraph(f"HRV: {report_data['hrv']}", styles['Normal']))
    content.append(Paragraph(f"HRV Status: {report_data['hrv_status']}", styles['Normal']))
    content.append(Paragraph(f"Signal Quality: {report_data['signal_quality']}", styles['Normal']))
    content.append(Paragraph(f"Risk Level: {report_data['risk_level']}", styles['Normal']))

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "<i>Note: AI-based supportive analysis only. Not a medical diagnosis.</i>",
        styles['Italic']
    ))

    # Background + Border
    def add_background(canvas, doc):
        canvas.saveState()

        # Background Image (optional)
        try:
            bg = ImageReader("heart_bg.png")  # place image in project folder
            canvas.drawImage(bg, 0, 0, width=A4[0], height=A4[1], mask='auto')
        except:
            pass

        # Red Border (hospital style)
        canvas.setStrokeColor(colors.red)
        canvas.setLineWidth(2)
        canvas.rect(20, 20, A4[0]-40, A4[1]-40)

        canvas.restoreState()

    doc.build(content, onFirstPage=add_background, onLaterPages=add_background)

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

    # REPORT
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

        # DOWNLOAD PDF
        pdf_file = create_pdf(report_data)

        st.download_button(
            label="📄 Download ECG Report (PDF)",
            data=pdf_file,
            file_name="ECG_Report.pdf",
            mime="application/pdf"
        )

    else:
        st.warning("Not enough R-peaks detected to generate ECG report.")

else:
    st.info("Please upload an ECG image to start analysis.")




