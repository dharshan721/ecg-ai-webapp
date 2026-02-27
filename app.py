import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Page settings
st.set_page_config(page_title="ECG AI Analysis", layout="wide")
st.title("ECG AI Analysis System")
st.write("Upload an ECG image to analyze heart rate and cardiac condition")

uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # ---- Read & Display Image ----
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("Uploaded ECG Image")
    st.image(img, use_container_width=True)

    # ---- Preprocessing ----
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

    # ---- ECG Signal Extraction ----
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

    # ---- R-Peak Detection ----
    peaks, _ = find_peaks(signal, distance=50, prominence=1)

    st.subheader("R-Peak Detection")
    fig2, ax2 = plt.subplots()
    ax2.plot(signal)
    ax2.plot(peaks, signal[peaks], "rx")
    ax2.set_title("Detected R-Peaks")
    st.pyplot(fig2)

    # ---- Final ECG Report ----
    if len(peaks) >= 2:

        fs = 250  # Assumed sampling frequency
        rr = np.diff(peaks)
        mean_rr = np.mean(rr)
        sdnn = np.std(rr)

        heart_rate = 60 * fs / mean_rr
        avg_rr_sec = mean_rr / fs

        # Heart Rate Category
        if heart_rate < 60:
            hr_category = "Low Heart Rate (Bradycardia)"
        elif heart_rate > 100:
            hr_category = "High Heart Rate (Tachycardia)"
        else:
            hr_category = "Normal Heart Rate"

        # HRV Interpretation
        if sdnn < 15:
            hrv_status = "Low HRV (Possible Stress)"
            rhythm_desc = "Regular rhythm (beats evenly spaced)"
        else:
            hrv_status = "Normal HRV"
            rhythm_desc = "Irregular rhythm (variable RR intervals)"

        # Signal Quality
        if len(peaks) > 8:
            signal_quality = "Good (clear waveform, minimal noise)"
        elif len(peaks) > 4:
            signal_quality = "Moderate quality"
        else:
            signal_quality = "Poor quality"

        # Risk Level
        if 60 <= heart_rate <= 100 and sdnn >= 15:
            risk_level = "Low Risk"
        elif 60 <= heart_rate <= 100 and sdnn < 15:
            risk_level = "Mild Risk (Stress Related)"
        else:
            risk_level = "Moderate Risk – Further Evaluation Recommended"

        # ---- Display Results ----
        st.subheader("ECG AI Analysis Report")

        st.write(f"*Heart Rate:* {int(heart_rate)} BPM")
        st.write(f"*Heart Rate Category:* {hr_category}")
        st.write(f"*ECG Status:* {rhythm_desc}")
        st.write(f"*Average RR Interval:* {round(avg_rr_sec, 3)} seconds")
        st.write(f"*RR Interval Variation (HRV):* {round(sdnn, 2)}")
        st.write(f"*HRV Status:* {hrv_status}")
        st.write(f"*ECG Signal Quality:* {signal_quality}")
        st.write(f"*Overall Risk Level:* {risk_level}")

        st.info("Note: AI-based supportive analysis only. Not a medical diagnosis.")

    else:
        st.warning("Not enough R-peaks detected to generate ECG report.")

else:
    st.info("Please upload an ECG image to start analysis.")





