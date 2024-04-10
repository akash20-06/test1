import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load the model
model_path = r'D:\Downloads\model.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)
    return img

def predict(image):
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        return class_names[pred]

def main():
    st.title("Dementia Classification App")
    st.sidebar.header("Settings")

    # Image selection
    uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Display images in a grid
        num_images = len(uploaded_files)
        columns = st.columns(3)  # Change the number of columns as needed

        for i in range(0, num_images, 3):
            for j in range(3):
                if i + j < num_images:
                    image = preprocess_image(uploaded_files[i + j])
                    prediction = predict(image)

                    # Display the uploaded image and prediction in a column
                    with columns[j]:
                        st.image(uploaded_files[i + j], caption=f"Prediction: {prediction}", use_column_width=True)

if _name_ == "_main_":
    main()