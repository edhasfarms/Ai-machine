import os


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================================
# YOUR CLASSES
# ============================================================

classes = [
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
# TRAIN / VALID / TEST FOLDERS
# ============================================================

splits = [
    "train",
    "valid",
    "test"
]


# ============================================================
# COUNT FUNCTION
# ============================================================

def count_images(folder_path):

    if not os.path.exists(folder_path):
        return 0

    count = 0

    for file in os.listdir(folder_path):

        if file.lower().endswith(IMAGE_EXTENSIONS):
            count += 1

    return count


# ============================================================
# CHECK DATASET
# ============================================================

print("\n" + "=" * 80)
print("CAPSICUM DATASET - TRAIN / VALID / TEST")
print("=" * 80)


grand_total = 0


for split in splits:

    print("\n" + "-" * 80)
    print(f"{split.upper()} DATASET")
    print("-" * 80)

    split_total = 0

    split_path = os.path.join(
        DATASET_DIR,
        split
    )

    if not os.path.exists(split_path):

        print(f"{split:20s}: FOLDER NOT FOUND")
        continue


    for class_name in classes:

        class_path = os.path.join(
            split_path,
            class_name
        )

        count = count_images(class_path)

        split_total += count

        print(
            f"{class_name:20s}: "
            f"{count:5d} images"
        )


    print("-" * 80)

    print(
        f"{'TOTAL ' + split:20s}: "
        f"{split_total:5d} images"
    )

    grand_total += split_total


# ============================================================
# GRAND TOTAL
# ============================================================

print("\n" + "=" * 80)

print(
    f"{'GRAND TOTAL':20s}: "
    f"{grand_total:5d} images"
)

print("=" * 80)