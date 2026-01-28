import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="Age Detection App", page_icon="👤", layout="centered")

# Load models (cached to avoid reloading on every interaction)
@st.cache_resource
def load_models():
    face_proto = "opencv_face_detector.pbtxt"
    face_model = "opencv_face_detector_uint8.pb"
    age_proto = "age_deploy.prototxt"
    age_model = "age_net.caffemodel"
    
    face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    
    return face_net, age_net

# Constants
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_faces(net, frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    return frame, face_boxes

def predict_age(face, net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    age_preds = net.forward()
    age = age_list[age_preds[0].argmax()]
    return age

def process_image(image, face_net, age_net):
    # Convert PIL Image to OpenCV format
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    frame, face_boxes = detect_faces(face_net, frame)
    
    ages = []
    # Process each detected face
    for (x1, y1, x2, y2) in face_boxes:
        face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1),
                     max(0, x1-20):min(x2+20, frame.shape[1]-1)]
        age = predict_age(face, age_net)
        ages.append(age)
        cv2.putText(frame, f"Age: {age}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Convert back to RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame, face_boxes, ages

# Main app
def main():
    st.title("👤 Age Detection App")
    st.write("Upload an image to detect faces and predict ages")
    
    # Load models
    try:
        face_net, age_net = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure all model files are in the same directory as this script.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Process image
        with st.spinner('Detecting faces and predicting ages...'):
            try:
                processed_image, face_boxes, ages = process_image(image, face_net, age_net)
                
                with col2:
                    st.subheader("Detected Ages")
                    st.image(processed_image, use_container_width=True)
                
                # Display results
                if len(face_boxes) == 0:
                    st.warning("No faces detected in the image. Try another image with clearer faces.")
                else:
                    st.success(f"✅ Detected {len(face_boxes)} face(s)!")
                    for i, age in enumerate(ages, 1):
                        st.info(f"**Face {i}:** Predicted age range: **{age}** years")
                        
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Click on 'Browse files' to upload an image
        2. The app will automatically detect faces in the image
        3. Age predictions will be displayed on the image
        4. Green boxes show detected faces
        5. Yellow text shows predicted age ranges
        
        **Supported formats:** JPG, JPEG, PNG
        """)

if __name__ == "__main__":
    main()
