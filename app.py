import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt

# Load the pre-trained flower classification model
model = load_model('CNN_model.h5')

# Flower class labels
class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Function to make a prediction
def predict(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class], prediction

# Streamlit app code
st.set_page_config(
    page_title="Flower Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Anurag-ghosh-12/FlowerClassification",
        "Report a bug": "https://github.com/Anurag-ghosh-12/FlowerClassification/issues",
    },
)

# Streamlit app
st.title('Flower Classifier')

# Sidebar with description
st.sidebar.title('About the Project')
st.sidebar.subheader(":blue[[Please use a desktop for the best experience.]]")
st.sidebar.info("""
This application uses a Convolutional Neural Network (CNN) model trained on images of flowers. The model can classify an uploaded image into one of five categories: Daisy, Dandelion, Roses, Sunflowers, and Tulips.
""")

st.sidebar.image('https://images.pexels.com/photos/67857/daisy-flower-spring-marguerite-67857.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2', caption='Daisy', use_column_width=True)
st.sidebar.image('https://images.pexels.com/photos/423604/pexels-photo-423604.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2', caption='Dandelion', use_column_width=True)
st.sidebar.image('https://images.pexels.com/photos/196664/pexels-photo-196664.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2', caption='Rose', use_column_width=True)
st.sidebar.image('https://images.pexels.com/photos/1454288/pexels-photo-1454288.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2', caption='Sunflower', use_column_width=True)
st.sidebar.image('https://images.pexels.com/photos/159406/tulips-netherlands-flowers-bloom-159406.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2', caption='Tulip', use_column_width=True)
st.sidebar.markdown("[GitHub](https://github.com/Anurag-ghosh-12/)")

# Main content
st.markdown("[Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)")
st.write('Upload an image of a flower and the model will predict the class.')

# File uploader for image
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)

    # Save the uploaded image temporarily
    temp_image_path = 'temp_image.png'
    image.save(temp_image_path)

    # Make a prediction
    st.write('Classifying...')
    label, prediction = predict(temp_image_path, model)
    st.markdown(f'<h1 style="font-size:2em; font-weight:bold;">This is a {label}!</h1>', unsafe_allow_html=True)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.balloons()

    # Show the prediction results
    confidence = np.max(prediction)
    st.write(f"Prediction Result: {label}")
    st.write(f"Prediction Confidence: {confidence:.2f}")

    # Plot the prediction probabilities
    expander = st.expander("Prediction result:")
    st.markdown('<p style="background-color: #073439; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📊 Prediction Probabilities 📊</p>', unsafe_allow_html=True)
    prediction_df = pd.DataFrame({'Flower Types': class_labels, 'Probabilities': prediction[0]})
    chart = alt.Chart(prediction_df).mark_bar().encode(
        x='Flower Types',
        y='Probabilities'
    ).properties(
        width=alt.Step(60),  # controls width of bar.
        background='white'   # sets the background to white.
    ).configure_axis(
        labelColor='black',
        titleColor='black'
    ).configure_title(
        color='black'
    )
    st.altair_chart(chart, use_container_width=True)

# Example images section
expander = st.expander("Some real life images to try with...")
expander.write("Just drag-and-drop your chosen image above")
example_images = [
    "./images/dummy1.jpeg",
    "./images/dummy2.jpeg",
    "./images/dummy3.jpeg", 
    "./images/dummy4.jpeg",
    "./images/dummy5.jpeg",
    "./images/dummy6.jpeg",
]

num_columns = 3
rows = len(example_images) // num_columns + (1 if len(example_images) % num_columns else 0)
for row in range(rows):
    cols = expander.columns(num_columns)
    for col in range(num_columns):
        index = row * num_columns + col
        if index < len(example_images):
            cols[col].image(example_images[index], width=200)

# Accuracy graph
expander = st.expander("View Model Training and Validation Results")
expander.write("Confusion Matrix : CNN with Data Augmentation ")
expander.image("images/goodmodel.png", use_column_width=True)
expander.write("Accuracy with CNN + Data Augmentation = 73.57%")
expander.write("Confusion Matrix : CNN only")
expander.image("images/confusion_mat.png", use_column_width=True)
expander.write("Accuracy with CNN only = 58%")

# Footer
st.write("\n\n\n")
st.markdown(
    f"""As this is a multi-class image classification task, some errors in classification of flowers may occur. I am actively working to improve accuracy. For better results, provide clear images with a single flower.
    \nDrop in any discrepancies or give suggestions in `Report a bug` option within the `⋮` menu"""
)

st.markdown(
    f"""
    <div style="text-align: right; color:#FCFF32; padding-bottom:1px;"> Developed by Anurag Ghosh </div>""",
    unsafe_allow_html=True,
)
