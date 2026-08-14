import os

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/condition_resnet50.pth"


# ============================================================
# CHANGE THIS IMAGE PATH
# ============================================================

IMAGE_PATH = r"C:\Users\singh\Downloads\magnific_macro-photography-of-a-gr_1lluGGsr4r.png"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n" + "=" * 70)
print("CAPSICUM LEAF PREDICTION")
print("=" * 70)

print(
    f"Device : {device}"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(

        f"\nModel not found:\n"
        f"{MODEL_PATH}\n\n"

        f"Run train.py first."
    )


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(

        f"\nImage not found:\n"
        f"{IMAGE_PATH}\n\n"

        f"Change IMAGE_PATH in predict.py."
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(

    MODEL_PATH,

    map_location=device
)


classes = checkpoint["classes"]

num_classes = checkpoint["num_classes"]


# ============================================================
# PRINT CLASSES
# ============================================================

print("\nClasses:")

for index, class_name in enumerate(classes):

    print(
        f"{index}: {class_name}"
    )


# ============================================================
# CREATE RESNET50
# ============================================================

model = models.resnet50(

    weights=None
)


# ============================================================
# FINAL FC LAYER
# ============================================================

num_features = model.fc.in_features


model.fc = nn.Linear(

    num_features,

    num_classes
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# MOVE MODEL
# ============================================================

model = model.to(device)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# IMAGE TRANSFORM
#
# MUST MATCH TRAINING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

print("\n")
print(
    f"Loading image:"
)

print(
    IMAGE_PATH
)


image = Image.open(

    IMAGE_PATH

).convert(
    "RGB"
)


# ============================================================
# TRANSFORM IMAGE
# ============================================================

image_tensor = transform(

    image
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image_tensor = image_tensor.unsqueeze(

    0
)


# ============================================================
# MOVE TO DEVICE
# ============================================================

image_tensor = image_tensor.to(

    device
)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(

        image_tensor
    )


    # Convert logits to probabilities
    probabilities = torch.softmax(

        outputs,

        dim=1
    )


# ============================================================
# BEST PREDICTION
# ============================================================

best_probability, best_index = torch.max(

    probabilities[0],

    0
)


predicted_class = classes[

    best_index.item()
]


confidence = (

    best_probability.item()

    * 100
)


# ============================================================
# RESULT
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    f"Image      : "
    f"{IMAGE_PATH}"
)

print(
    f"Prediction : "
    f"{predicted_class}"
)

print(
    f"Confidence : "
    f"{confidence:.2f}%"
)


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 3 PREDICTIONS")
print("=" * 70)


top_k = min(

    3,

    num_classes
)


top_probabilities, top_indices = torch.topk(

    probabilities[0],

    top_k
)


for i in range(top_k):

    index = top_indices[i].item()

    class_name = classes[index]

    probability = (

        top_probabilities[i].item()

        * 100
    )


    print(

        f"{i + 1}. "
        f"{class_name:20s}"
        f" : {probability:.2f}%"
    )


print("\n")
print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)