import os
import random
import shutil

from PIL import Image
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = "dataset"

SPLIT_DIR = "dataset_split"

OUTPUT_DIR = "augmented_dataset"

# 80% Train
# 10% Validation
# 10% Test

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# Final minimum TRAIN images per class
TRAIN_TARGET = 300

SEED = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================================
# YOUR ACTUAL 10 CLASSES
# ============================================================

CLASSES = [
    "Caterpillar",
    "diabac",
    "downleaf",
    "healthy",
    "not_grown",
    "Powdery mildew",
    "recovery",
    "snail",
    "upper leaf",
    "virus"
]


# ============================================================
# TRAIN AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.3
    ),

    transforms.RandomRotation(
        degrees=25
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.RandomResizedCrop(
        size=(224, 224),
        scale=(0.80, 1.0)
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# NO AUGMENTATION
# ============================================================

resize_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    )
])


# ============================================================
# CHECK IMAGE
# ============================================================

def is_image(filename):

    return filename.lower().endswith(
        IMAGE_EXTENSIONS
    )


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    if not os.path.exists(folder):
        return []

    images = []

    for file in os.listdir(folder):

        if is_image(file):

            images.append(
                os.path.join(
                    folder,
                    file
                )
            )

    return images


# ============================================================
# CLEAN DIRECTORY
# ============================================================

def clean_directory(path):

    if os.path.exists(path):

        print(
            f"\nRemoving old directory: {path}"
        )

        shutil.rmtree(path)

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# STEP 1
# AUTOMATIC 80/10/10 SPLIT
# ============================================================

def split_dataset():

    print("\n")
    print("=" * 70)
    print("STEP 1 - AUTOMATIC DATASET SPLIT")
    print("=" * 70)

    # Remove old split
    clean_directory(
        SPLIT_DIR
    )

    # Create train/valid/test folders
    for split in [
        "train",
        "valid",
        "test"
    ]:

        for class_name in CLASSES:

            os.makedirs(
                os.path.join(
                    SPLIT_DIR,
                    split,
                    class_name
                ),
                exist_ok=True
            )

    # --------------------------------------------------------
    # Process every class
    # --------------------------------------------------------

    for class_name in CLASSES:

        source_folder = os.path.join(
            SOURCE_DIR,
            class_name
        )

        images = get_images(
            source_folder
        )

        total = len(images)

        if total == 0:

            print(
                f"\nWARNING: {class_name} has no images"
            )

            continue

        # Shuffle images
        random.shuffle(
            images
        )

        # ----------------------------------------------------
        # Calculate split
        # ----------------------------------------------------

        train_count = int(
            total * TRAIN_RATIO
        )

        valid_count = int(
            total * VALID_RATIO
        )

        train_images = images[
            :train_count
        ]

        valid_images = images[
            train_count:
            train_count + valid_count
        ]

        test_images = images[
            train_count + valid_count:
        ]

        print("\n" + "-" * 60)

        print(
            f"Class : {class_name}"
        )

        print(
            f"Total : {total}"
        )

        print(
            f"Train : {len(train_images)}"
        )

        print(
            f"Valid : {len(valid_images)}"
        )

        print(
            f"Test  : {len(test_images)}"
        )

        # ----------------------------------------------------
        # Copy files
        # ----------------------------------------------------

        split_data = {

            "train": train_images,

            "valid": valid_images,

            "test": test_images

        }

        for split_name, split_images in split_data.items():

            destination_folder = os.path.join(
                SPLIT_DIR,
                split_name,
                class_name
            )

            for index, image_path in enumerate(
                split_images
            ):

                extension = os.path.splitext(
                    image_path
                )[1].lower()

                destination_path = os.path.join(
                    destination_folder,
                    f"{class_name}_{index + 1:05d}{extension}"
                )

                shutil.copy2(
                    image_path,
                    destination_path
                )

    print("\n")
    print("80/10/10 SPLIT COMPLETED")


# ============================================================
# STEP 2
# CREATE FINAL DATASET FOLDERS
# ============================================================

def create_output():

    print("\n")
    print("=" * 70)
    print("CREATING FINAL DATASET")
    print("=" * 70)

    clean_directory(
        OUTPUT_DIR
    )

    for split in [
        "train",
        "valid",
        "test"
    ]:

        for class_name in CLASSES:

            os.makedirs(
                os.path.join(
                    OUTPUT_DIR,
                    split,
                    class_name
                ),
                exist_ok=True
            )


# ============================================================
# STEP 3
# PREPARE TRAIN DATA
#
# Original + augmented
# Target = 300 images/class
# ============================================================

def prepare_train(class_name):

    source_folder = os.path.join(
        SPLIT_DIR,
        "train",
        class_name
    )

    output_folder = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name
    )

    images = get_images(
        source_folder
    )

    original_count = len(
        images
    )

    if original_count == 0:

        print(
            f"\nWARNING: No train images for {class_name}"
        )

        return

    random.shuffle(
        images
    )

    # --------------------------------------------------------
    # COPY ORIGINAL TRAIN IMAGES
    # --------------------------------------------------------

    for index, image_path in enumerate(
        images
    ):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = resize_transform(
                image
            )

            output_path = os.path.join(
                output_folder,
                f"original_{index + 1:05d}.jpg"
            )

            image.save(
                output_path,
                format="JPEG",
                quality=95
            )

        except Exception as e:

            print(
                f"ERROR: {image_path}"
            )

            print(e)

    # --------------------------------------------------------
    # IF ALREADY >= 300
    # DON'T AUGMENT
    # --------------------------------------------------------

    if original_count >= TRAIN_TARGET:

        print(
            f"\n{class_name}"
        )

        print(
            f"Original Train : {original_count}"
        )

        print(
            f"Final Train    : {original_count}"
        )

        print(
            "No augmentation required."
        )

        return

    # --------------------------------------------------------
    # NUMBER OF AUGMENTED IMAGES REQUIRED
    # --------------------------------------------------------

    additional = (
        TRAIN_TARGET -
        original_count
    )

    print(
        f"\n{class_name}"
    )

    print(
        f"Original Train : {original_count}"
    )

    print(
        f"Augmented      : {additional}"
    )

    # --------------------------------------------------------
    # GENERATE AUGMENTED IMAGES
    # --------------------------------------------------------

    for i in range(
        additional
    ):

        # Reuse original images as augmentation sources
        image_path = images[
            i % original_count
        ]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            augmented = train_transform(
                image
            )

            output_path = os.path.join(
                output_folder,
                f"augmented_{i + 1:05d}.jpg"
            )

            augmented.save(
                output_path,
                format="JPEG",
                quality=95
            )

        except Exception as e:

            print(
                f"ERROR augmenting:"
            )

            print(
                image_path
            )

            print(e)

    print(
        f"Final Train    : "
        f"{original_count + additional}"
    )


# ============================================================
# STEP 4
# PREPARE VALIDATION
#
# NO AUGMENTATION
# ============================================================

def prepare_valid(class_name):

    source_folder = os.path.join(
        SPLIT_DIR,
        "valid",
        class_name
    )

    output_folder = os.path.join(
        OUTPUT_DIR,
        "valid",
        class_name
    )

    images = get_images(
        source_folder
    )

    for index, image_path in enumerate(
        images
    ):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = resize_transform(
                image
            )

            output_path = os.path.join(
                output_folder,
                f"valid_{index + 1:05d}.jpg"
            )

            image.save(
                output_path,
                format="JPEG",
                quality=95
            )

        except Exception as e:

            print(
                f"ERROR: {image_path}"
            )

            print(e)

    print(
        f"VALID - {class_name}: "
        f"{len(images)} images"
    )


# ============================================================
# STEP 5
# PREPARE TEST
#
# NO AUGMENTATION
# ============================================================

def prepare_test(class_name):

    source_folder = os.path.join(
        SPLIT_DIR,
        "test",
        class_name
    )

    output_folder = os.path.join(
        OUTPUT_DIR,
        "test",
        class_name
    )

    images = get_images(
        source_folder
    )

    for index, image_path in enumerate(
        images
    ):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = resize_transform(
                image
            )

            output_path = os.path.join(
                output_folder,
                f"test_{index + 1:05d}.jpg"
            )

            image.save(
                output_path,
                format="JPEG",
                quality=95
            )

        except Exception as e:

            print(
                f"ERROR: {image_path}"
            )

            print(e)

    print(
        f"TEST - {class_name}: "
        f"{len(images)} images"
    )


# ============================================================
# STEP 6
# FINAL DATASET CHECK
# ============================================================

def final_check():

    print("\n")
    print("=" * 70)
    print("FINAL DATASET COUNT")
    print("=" * 70)

    total_train = 0
    total_valid = 0
    total_test = 0

    for class_name in CLASSES:

        train_count = len(
            get_images(
                os.path.join(
                    OUTPUT_DIR,
                    "train",
                    class_name
                )
            )
        )

        valid_count = len(
            get_images(
                os.path.join(
                    OUTPUT_DIR,
                    "valid",
                    class_name
                )
            )
        )

        test_count = len(
            get_images(
                os.path.join(
                    OUTPUT_DIR,
                    "test",
                    class_name
                )
            )
        )

        total_train += train_count
        total_valid += valid_count
        total_test += test_count

        print(
            f"\n{class_name}"
        )

        print(
            f"  Train : {train_count}"
        )

        print(
            f"  Valid : {valid_count}"
        )

        print(
            f"  Test  : {test_count}"
        )

    print("\n" + "-" * 70)

    print(
        f"TOTAL TRAIN : {total_train}"
    )

    print(
        f"TOTAL VALID : {total_valid}"
    )

    print(
        f"TOTAL TEST  : {total_test}"
    )

    print(
        f"TOTAL DATA  : "
        f"{total_train + total_valid + total_test}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("CAPSICUM DATASET PROCESSING")
    print("=" * 70)

    # Reproducible split
    random.seed(
        SEED
    )

    # --------------------------------------------------------
    # 1. Raw dataset -> train/valid/test
    # --------------------------------------------------------

    split_dataset()

    # --------------------------------------------------------
    # 2. Create final dataset
    # --------------------------------------------------------

    create_output()

    # --------------------------------------------------------
    # 3. Prepare TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREPARING TRAIN DATA")
    print("=" * 70)

    for class_name in CLASSES:

        prepare_train(
            class_name
        )

    # --------------------------------------------------------
    # 4. Prepare VALIDATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREPARING VALIDATION DATA")
    print("=" * 70)

    for class_name in CLASSES:

        prepare_valid(
            class_name
        )

    # --------------------------------------------------------
    # 5. Prepare TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREPARING TEST DATA")
    print("=" * 70)

    for class_name in CLASSES:

        prepare_test(
            class_name
        )

    # --------------------------------------------------------
    # 6. Final report
    # --------------------------------------------------------

    final_check()

    print("\n")
    print("=" * 70)
    print("DATASET PROCESSING COMPLETED")
    print("=" * 70)