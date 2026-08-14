import os

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "augmented_dataset/train"
VALID_DIR = "augmented_dataset/valid"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "condition_resnet50.pth"
)

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.0001

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "=" * 70)
print("CAPSICUM RESNET50 TRAINING")
print("=" * 70)

print(
    f"Device : {device}"
)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"\nTrain dataset not found:\n"
        f"{TRAIN_DIR}\n\n"
        f"Run augmentation.py first."
    )


if not os.path.exists(VALID_DIR):

    raise FileNotFoundError(
        f"\nValidation dataset not found:\n"
        f"{VALID_DIR}\n\n"
        f"Run augmentation.py first."
    )


# ============================================================
# TRANSFORM
#
# augmentation.py already created augmented images.
#
# Therefore NO RANDOM AUGMENTATION here.
#
# Only:
# Resize
# ToTensor
# Normalize
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
# LOAD TRAIN DATASET
# ============================================================

train_dataset = datasets.ImageFolder(

    root=TRAIN_DIR,

    transform=transform
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

valid_dataset = datasets.ImageFolder(

    root=VALID_DIR,

    transform=transform
)


# ============================================================
# CLASSES
# ============================================================

classes = train_dataset.classes

num_classes = len(classes)


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
# EXPECT 10 CLASSES
# ============================================================

if num_classes != 10:

    raise ValueError(

        f"\nExpected 10 classes "
        f"but found {num_classes}.\n\n"

        f"Classes found:\n"
        f"{classes}"
    )


# ============================================================
# CHECK TRAIN / VALID CLASS ORDER
# ============================================================

if train_dataset.classes != valid_dataset.classes:

    raise ValueError(

        "\nTrain and validation "
        "classes do not match!\n\n"

        f"Train:\n"
        f"{train_dataset.classes}\n\n"

        f"Valid:\n"
        f"{valid_dataset.classes}"
    )


# ============================================================
# DATASET SIZE
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    f"Train images : "
    f"{len(train_dataset)}"
)

print(
    f"Valid images : "
    f"{len(valid_dataset)}"
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS
)


valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS
)


# ============================================================
# CREATE RESNET50
# ============================================================

print("\n" + "=" * 70)
print("CREATING RESNET50")
print("=" * 70)


model = models.resnet50(

    weights=None
)


# ============================================================
# RESNET50 INPUT / OUTPUT INFORMATION
# ============================================================

num_features = model.fc.in_features


print(
    f"Input image size : 224 x 224"
)

print(
    f"Backbone         : ResNet50"
)

print(
    f"FC input features: {num_features}"
)

print(
    f"Output classes   : {num_classes}"
)


# ============================================================
# REPLACE FINAL FC LAYER
# ============================================================

model.fc = nn.Linear(

    num_features,

    num_classes
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(

    MODEL_DIR,

    exist_ok=True
)


# ============================================================
# BEST VALIDATION ACCURACY
# ============================================================

best_val_accuracy = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:


        images = images.to(device)

        labels = labels.to(device)


        # ----------------------------------------------------
        # CLEAR GRADIENT
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(

            outputs,

            labels
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # UPDATE PARAMETERS
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        running_loss += (

            loss.item() *

            images.size(0)
        )


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        _, predicted = torch.max(

            outputs,

            1
        )


        total += labels.size(0)


        correct += (

            predicted == labels

        ).sum().item()


    # ========================================================
    # TRAIN RESULTS
    # ========================================================

    train_loss = (

        running_loss /

        total
    )


    train_accuracy = (

        100.0 *

        correct /

        total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    validation_loss = 0.0

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():


        for images, labels in valid_loader:


            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(

                outputs,

                labels
            )


            validation_loss += (

                loss.item() *

                images.size(0)
            )


            _, predicted = torch.max(

                outputs,

                1
            )


            validation_total += (

                labels.size(0)
            )


            validation_correct += (

                predicted == labels

            ).sum().item()


    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    val_loss = (

        validation_loss /

        validation_total
    )


    val_accuracy = (

        100.0 *

        validation_correct /

        validation_total
    )


    # ========================================================
    # PRINT EPOCH
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"Epoch "
        f"[{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Valid Loss     : "
        f"{val_loss:.4f}"
    )

    print(
        f"Valid Accuracy : "
        f"{val_accuracy:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:


        best_val_accuracy = val_accuracy


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "classes":
                classes,

            "class_to_idx":
                train_dataset.class_to_idx,

            "num_classes":
                num_classes,

            "input_size":
                224,

            "architecture":
                "ResNet50",

            "best_val_accuracy":
                best_val_accuracy
        }


        torch.save(

            checkpoint,

            MODEL_PATH
        )


        print(
            "✓ Best model saved!"
        )

        print(
            f"  Path: {MODEL_PATH}"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Model Path               : "
    f"{MODEL_PATH}"
)

print("=" * 70)