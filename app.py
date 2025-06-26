import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 LUNG CANCER DETECTION USING EFFICIENTNET AND RADIOMICS:A HYBRID APPROACH")
st.markdown("Upload a `.npy` CT scan file to preview a slice and simulate diagnosis.")

# Upload
uploaded_file = st.file_uploader("📤 Upload a NumPy (.npy) image file", type=["npy"])

if uploaded_file:
    try:
        image = np.load(uploaded_file)

        st.success("✅ Image loaded successfully!")
        st.write(f"Image shape: {image.shape}")

        # Handle 2D or 3D
        if image.ndim == 2:
            slice_to_show = image
        elif image.ndim == 3:
            # Choose slice
            st.subheader("🧭 Select CT Slice (depth axis):")
            slice_index = st.slider("Slice Number", 0, image.shape[0] - 1, image.shape[0] // 2)
            slice_to_show = image[slice_index]
        else:
            st.error("❌ Unsupported image shape.")
            st.stop()

        # Display the image
        st.subheader("🖼️ CT Slice Preview:")
        fig, ax = plt.subplots()
        ax.imshow(slice_to_show, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

        # Simulate prediction
        st.subheader("🔍 Diagnosis (Simulated)")
        simulated_result = np.random.choice(["Cancer Present", "Cancer Not Present"])
        color = "red" if simulated_result == "Cancer Present" else "green"
        st.markdown(f"<h2 style='color:{color}'>{simulated_result}</h2>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")