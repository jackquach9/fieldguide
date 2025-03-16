import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO
from skimage import io
import torch
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
import plotly.graph_objects as go

def predict(model_path, img):
    img = Image.fromarray(img)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1011)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input)
        probs = F.softmax(output)[0].tolist()
        top_labels = torch.argsort(output, dim=1).tolist()[0]
        top_5 = [(top_labels[-1 * i], probs[top_labels[-1 * i]]) for i in range(1, 6)]
        print(top_5)
        return top_5
    
def word_stream(line):
    for word in line.split(" "):
        yield word + " "
        time.sleep(0.05)

def make_prediction(image):
    top_5 = predict('./birdguide_model_resnet_v2.pth', image)
    top_label, top_prob = top_5[0][0], top_5[0][1]
    label_map = pd.read_csv('./bird_mappings.csv').to_dict()["Bird Name"]
    predicted_name = label_map[top_label]
    top_labels, top_probs = [predicted_name], [top_prob]
    st.write_stream(word_stream("Your bird is a..."))
    time.sleep(1)
    st.title(predicted_name)
    st.markdown(f"with a probability of **{(100 * top_prob):.2f}%**." )
    st.empty()
    st.subheader("Our next best guesses:")
    for next_guess in top_5[1:]:
        next_label, next_prob = label_map[next_guess[0]], next_guess[1] * 100
        top_labels.append(next_label)
        top_probs.append(next_prob / 100)
        st.markdown(f"**{next_label}** with a probability of **{next_prob:.2f}%.**")
    print(top_labels.reverse(), top_probs.reverse())
    # Create a Plotly Figure
    fig = go.Figure(go.Bar(
        y=top_labels,  # Categories on y-axis (horizontal bars)
        x=top_probs,  # Scaled values for bar length
        orientation='h',  # Horizontal bars
        marker=dict(color='darkgreen')  # Custom colors
    ))

    # Remove x-axis numbers (since you don’t want exact percentages)
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        margin=dict(l=100, r=10, t=40, b=40)
    )

    # Display in Streamlit
    st.plotly_chart(fig)

    st.write("The image data for the training of this machine learning model was obtained through the open-source NABirds dataset, published by the Cornell Lab of Ornithology.")
    


def main():

    with st.container():
        st.markdown(
            "<h1 style='text-align: center; color: white; background-color: darkgreen; padding: 10px; border-radius: 5px;'>FieldGuide: AI-Powered Wildlife Identification (BirdsEye Mode)</h1>",
            unsafe_allow_html=True
        )

    st.text(" ")
    st.text("Welcome to FieldGuide (BirdsEye Mode)! Add an image of a bird (or multiple birds), and we will use a computer vision machine learning algorithm to classify the species of bird! (Works best on birds commonly found in North America).")

    # Load YOLOv8 model
    model = YOLO("yolov8n-seg.pt")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image of a bird!", type=["jpg", "jpeg"])
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)

            max_width = 500
            scale = max_width / max(image.shape[1], 1)
            new_height = int(image.shape[0] * scale)
            new_width = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            st.image(image)

            selection = st.segmented_control("Dectect using the whole image or just one bird?", options=["Whole Image", "Select Bird"])

            if selection:
                if selection == "Select Bird":
                    # Display image with clickable feature
                    coords = streamlit_image_coordinates(image, key="image_click")

                    if coords:
                        x, y = coords["x"], coords["y"]
                        # pixel_color = image[y, x]

                        # Draw a red dot at the selected pixel
                        image_with_dot = image.copy()
                        cv2.circle(image_with_dot, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
                        st.image(image_with_dot, caption="Image with Selected Pixel", use_container_width=False)

                        # st.write(f"Selected Pixel: (X: {x}, Y: {y})")
                        # st.write(f"Pixel Color (RGB): {pixel_color}")
                        # st.markdown(f"<div style='width:50px;height:50px;background-color:rgb{tuple(pixel_color)};'></div>", unsafe_allow_html=True)

                        # Perform YOLOv8 instance segmentation
                        results = model(image)

                        for result in results:
                            for box in result.boxes.xyxy:
                                x1, y1, x2, y2 = map(int, box)
                                if x1 <= x <= x2 and y1 <= y <= y2:
                                    if x2 > x1 and y2 > y1:
                                        cropped_image = image[y1:y2, x1:x2]
                                        st.image(cropped_image)
                                        make_prediction(cropped_image)
                                        return
                        st.write("No object detected at the selected pixel.")
                            
                else:
                    make_prediction(image)
                    return

        

            

if __name__ == '__main__':
    main()
        