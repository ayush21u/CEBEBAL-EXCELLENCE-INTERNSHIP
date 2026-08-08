
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Configuration
# -----------------------------

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# Load ResNet18
# -----------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    model.load_state_dict(
        torch.load(
            "models/resnet18_eurosat.pt",
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Page
# -----------------------------

st.set_page_config(
    page_title="Satellite Image Intelligence",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Image Intelligence System")

st.write(
    "Land-use classification and embedding-based change detection "
    "using ResNet18."
)

# -----------------------------
# Tabs
# -----------------------------

tab1, tab2 = st.tabs([
    "Land-Use Classification",
    "Change Detection"
])

# -----------------------------
# Classification
# -----------------------------

with tab1:

    st.header("Satellite Land-Use Classification")

    uploaded_image = st.file_uploader(
        "Upload a satellite image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        input_tensor = transform(
            image
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = CLASS_NAMES[
            prediction.item()
        ]

        confidence_value = confidence.item()

        st.success(
            f"Predicted Class: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence_value * 100:.2f}%"
        )

        # Probability chart

        probs = probabilities[0].cpu().numpy()

        chart_data = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(10)
        }

        st.bar_chart(chart_data)


# -----------------------------
# Change Detection
# -----------------------------

with tab2:

    st.header("Embedding-Based Change Detection")

    image_a_file = st.file_uploader(
        "Upload Image A",
        type=["jpg", "jpeg", "png"],
        key="image_a"
    )

    image_b_file = st.file_uploader(
        "Upload Image B",
        type=["jpg", "jpeg", "png"],
        key="image_b"
    )

    if image_a_file is not None and image_b_file is not None:

        image_a = Image.open(
            image_a_file
        ).convert("RGB")

        image_b = Image.open(
            image_b_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image_a,
                caption="Image A"
            )

        with col2:
            st.image(
                image_b,
                caption="Image B"
            )

        tensor_a = transform(
            image_a
        ).unsqueeze(0).to(DEVICE)

        tensor_b = transform(
            image_b
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            features_a = model.avgpool(
                model.layer4(
                    model.layer3(
                        model.layer2(
                            model.layer1(
                                model.relu(
                                    model.bn1(
                                        model.conv1(
                                            tensor_a
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )

            features_b = model.avgpool(
                model.layer4(
                    model.layer3(
                        model.layer2(
                            model.layer1(
                                model.relu(
                                    model.bn1(
                                        model.conv1(
                                            tensor_b
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )

        features_a = torch.flatten(
            features_a,
            1
        ).cpu().numpy()

        features_b = torch.flatten(
            features_b,
            1
        ).cpu().numpy()

        similarity = cosine_similarity(
            features_a,
            features_b
        )[0][0]

        change_score = 1 - similarity

        st.metric(
            "Cosine Similarity",
            f"{similarity:.4f}"
        )

        st.metric(
            "Change Score",
            f"{change_score:.4f}"
        )

        # Pixel difference heatmap

        arr_a = np.array(
            image_a.resize((256, 256))
        ).astype(float)

        arr_b = np.array(
            image_b.resize((256, 256))
        ).astype(float)

        difference = np.abs(
            arr_a - arr_b
        )

        change_map = np.mean(
            difference,
            axis=2
        ) / 255.0

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        heatmap = ax.imshow(
            change_map,
            cmap="hot"
        )

        ax.set_title(
            "Pixel Difference Heatmap"
        )

        ax.axis("off")

        fig.colorbar(
            heatmap,
            ax=ax
        )

        st.pyplot(fig)
