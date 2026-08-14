import os

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "augmented_dataset/test"

MODEL_PATH = "models/condition_resnet50.pth"

BATCH_SIZE = 32

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "=" * 70)
print("CAPSICUM MODEL TESTING")
print("=" * 70)

print(f"Device : {device}")


# ============================================================
# CHECK TEST DIRECTORY
# ============================================================

if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"\nTest dataset not found:\n"
        f"{TEST_DIR}\n\n"
        f"Run augmentation.py first."
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
# TRANSFORM
#
# NO AUGMENTATION
# ============================================================

test_transform = transforms.Compose([

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
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(

    root=TEST_DIR,

    transform=test_transform
)


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

checkpoint = torch.load(

    MODEL_PATH,

    map_location=device
)


classes = checkpoint["classes"]

num_classes = checkpoint["num_classes"]


# ============================================================
# CHECK CLASSES
# ============================================================

print("\n" + "=" * 70)
print("CLASSES")
print("=" * 70)

for index, class_name in enumerate(classes):

    print(
        f"{index}: {class_name}"
    )


print(
    f"\nNumber of classes: "
    f"{num_classes}"
)


# ============================================================
# CHECK TEST CLASSES
# ============================================================

if test_dataset.classes != classes:

    raise ValueError(

        "\nTest classes do not match "
        "model classes!\n\n"

        f"Model classes:\n"
        f"{classes}\n\n"

        f"Test classes:\n"
        f"{test_dataset.classes}"
    )


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS
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


model = model.to(device)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []

all_predictions = []


print("\n")
print("=" * 70)
print("RUNNING TEST")
print("=" * 70)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Forward pass
        outputs = model(
            images
        )


        # Prediction
        _, predictions = torch.max(

            outputs,

            1
        )


        # Save labels
        all_labels.extend(

            labels.cpu().numpy()
        )


        # Save predictions
        all_predictions.extend(

            predictions.cpu().numpy()
        )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(

    all_labels,

    all_predictions
)


correct = sum(

    true == pred

    for true, pred in zip(

        all_labels,

        all_predictions
    )
)


wrong = len(all_labels) - correct


# ============================================================
# TEST RESULT
# ============================================================

print("\n")
print("=" * 70)
print("TEST RESULT")
print("=" * 70)

print(
    f"Total Test Images : "
    f"{len(all_labels)}"
)

print(
    f"Correct           : "
    f"{correct}"
)

print(
    f"Wrong             : "
    f"{wrong}"
)

print(
    f"Test Accuracy     : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


print(

    classification_report(

        all_labels,

        all_predictions,

        labels=list(
            range(num_classes)
        ),

        target_names=classes,

        digits=4,

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=list(
        range(num_classes)
    )
)


print(cm)


# ============================================================
# PER CLASS ACCURACY
# ============================================================

print("\n")
print("=" * 70)
print("PER CLASS ACCURACY")
print("=" * 70)


for i, class_name in enumerate(classes):

    total_class = cm[i].sum()

    correct_class = cm[i][i]


    if total_class > 0:

        class_accuracy = (

            correct_class /

            total_class *

            100
        )

    else:

        class_accuracy = 0


    print(

        f"{class_name:20s} : "
        f"{class_accuracy:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TESTING COMPLETED")
print("=" * 70)

print(
    f"Final Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("=" * 70)