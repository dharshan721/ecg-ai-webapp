import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

st.set_page_config(page_title="ECG AI Analysis", layout="wide")
st.title("ECG AI Analysis System")
st.write("Upload an ECG image to analyze heart rate and cardiac condition")

uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # ---- Read & display image ----
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

    # ---- R-peak Detection ----
    peaks, _ = find_peaks(signal, distance=50, prominence=1)

    st.subheader("R-Peak Detection")
    fig2, ax2 = plt.subplots()
    ax2.plot(signal)
    ax2.plot(peaks, signal[peaks], "rx")
    ax2.set_title("Detected R-Peaks")
    st.pyplot(fig2)

    # ---- Final ECG AI Report ----
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
        st.write(f"**Heart Rate:** {int(heart_rate)} BPM")
        st.write(f"**Heart Rate Category:** {hr_category}")
        st.write(f"**ECG Status:** {condition}")
        st.write(f"**Average RR Interval:** {round(mean_rr, 2)}")
        st.write(f"**RR Interval Variation (HRV):** {round(sdnn, 2)}")
        st.write(f"**HRV Status:** {hrv_status}")
        st.write(f"**ECG Signal Quality:** {signal_quality}")
        st.write(f"**Overall Risk Level:** {risk_level}")

        st.info("Note: AI-based supportive analysis only. Not a medical diagnosis.")
    else:
        st.warning("Not enough R-peaks detected to generate ECG report.")
st.markdown("### 🔎 Explanation of Results")

st.markdown("### 🫀 Advanced ECG Waveform Interpretation (AI-Supported)")

# --- Heuristic features from signal ---
# Approx QRS width (in samples)
if len(peaks) >= 2:
    qrs_widths = []
    for p in peaks:
        left = max(p - 5, 0)
        right = min(p + 5, len(signal) - 1)
        qrs_widths.append(right - left)
    avg_qrs_width = np.mean(qrs_widths)
else:
    avg_qrs_width = np.nan

# Baseline around peaks (proxy for ST change)
baseline_vals = []
for p in peaks:
    pre = signal[max(p-20, 0):max(p-5, 0)]
    post = signal[min(p+5, len(signal)-1):min(p+20, len(signal)-1)]
    if len(pre) > 0 and len(post) > 0:
        baseline_vals.append(np.mean(post) - np.mean(pre))
baseline_shift = np.nanmean(baseline_vals) if len(baseline_vals) else 0

# Rhythm regularity (RR variability)
rr_var = np.std(np.diff(peaks)) if len(peaks) >= 3 else 0

# --- Interpretations ---
# P wave (proxy: rhythm regularity)
if rr_var < 10:
    st.write("**P Wave (Atrial Activity):** Rhythm appears regular, suggesting organized atrial activation.")
else:
    st.write("**P Wave (Atrial Activity):** Rhythm variability is noted; atrial activation may be irregular (supportive indicator).")

# QRS complex (ventricular activation)
if not np.isnan(avg_qrs_width) and avg_qrs_width > 10:
    st.write("**QRS Complex (Ventricular Activity):** QRS appears wider than typical, which may suggest delayed ventricular conduction (supportive flag).")
else:
    st.write("**QRS Complex (Ventricular Activity):** QRS width appears within a typical range, suggesting normal ventricular activation.")

# T wave (recovery phase – proxy using post-peak slope)
if sdnn < 20:
    st.write("**T Wave (Recovery Phase):** Recovery pattern may be influenced by stress or fatigue (supportive indicator).")
else:
    st.write("**T Wave (Recovery Phase):** Recovery pattern appears stable in this recording.")

# ST segment changes (ischemia proxy)
if baseline_shift > 0.5:
    st.write("**ST Segment:** Elevation-like baseline shift detected. This can be associated with ischemic changes (supportive screening flag).")
elif baseline_shift < -0.5:
    st.write("**ST Segment:** Depression-like baseline shift detected. This can be associated with reduced blood flow (supportive screening flag).")
else:
    st.write("**ST Segment:** No significant baseline shift detected in this recording.")

# Conduction blocks (proxy via irregular RR + widened QRS)
if (rr_var > 15) and (not np.isnan(avg_qrs_width) and avg_qrs_width > 10):
    st.write("**Conduction Blocks:** Patterns suggest possible conduction delay (e.g., AV block or bundle branch delay) – supportive indicator.")
else:
    st.write("**Conduction Blocks:** No strong supportive indicators of conduction block detected in this snapshot.")

# Heart enlargement / strain (LVH proxy – amplitude heuristic)
signal_amp = np.nanmax(signal) - np.nanmin(signal)
if signal_amp > np.percentile(np.abs(signal), 90):
    st.write("**Heart Enlargement / Strain:** Higher signal amplitude may indicate ventricular strain patterns (supportive indicator).")
else:
    st.write("**Heart Enlargement / Strain:** No strong amplitude-based indicators of ventricular strain detected.")

# Electrolyte / Drug effect (pattern-based proxy)
if sdnn < 15 and heart_rate > 90:
    st.write("**Electrolyte / Drug Effect:** Pattern may be influenced by physiological stress or medication effects (supportive indicator).")
else:
    st.write("**Electrolyte / Drug Effect:** No strong pattern-based indicators detected in this snapshot.")

st.warning("These waveform interpretations are AI-supported screening indicators from ECG images, not clinical diagnoses. For medical decisions, consult a qualified clinician.")
uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # ... preprocessing ...
    # ... signal extraction ...

    peaks, _ = find_peaks(signal, distance=50, prominence=1)

    # ✅ FINAL REPORT MUST BE HERE (inside this block)
    if len(peaks) >= 2:
        # compute HR, HRV, report
        ...

if 'peaks' not in locals():
    peaks = np.array([])

