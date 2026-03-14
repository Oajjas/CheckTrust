import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

fake = Faker("en_IN")

OUTPUT_DIR = "synthetic_documents"
DOC_TYPES = ["aadhaar_like", "pan_like", "certificate"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for d in DOC_TYPES:
    os.makedirs(f"{OUTPUT_DIR}/{d}", exist_ok=True)

def random_aadhaar():
    return f"{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

def random_pan():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(random.choice(letters) for _ in range(5)) + \
           str(random.randint(1000,9999)) + \
           random.choice(letters)

def create_base_image(text_lines):

    width, height = 900, 550
    img = Image.new("RGB", (width, height), "white")

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    y = 80

    for line in text_lines:
        draw.text((80, y), line, fill="black", font=font)
        y += 60

    return img

def apply_augmentation(pil_img):

    img = np.array(pil_img)

    # rotation
    angle = random.uniform(-8, 8)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

    # blur
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img, (5,5), 0)

    # brightness
    if random.random() < 0.4:
        alpha = random.uniform(0.7, 1.3)
        img = cv2.convertScaleAbs(img, alpha=alpha)

    # noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return Image.fromarray(img)

def generate_aadhaar(i):

    text = [
        "GOVERNMENT OF INDIA",
        f"Name: {fake.name()}",
        f"DOB: {fake.date_of_birth(minimum_age=18, maximum_age=80)}",
        f"Aadhaar: {random_aadhaar()}",
        f"Address: {fake.address().replace(chr(10), ' ')}"
    ]

    img = create_base_image(text)
    img = apply_augmentation(img)

    img.save(f"{OUTPUT_DIR}/aadhaar_like/aadhaar_{i}.jpg")

def generate_pan(i):

    text = [
        "INCOME TAX DEPARTMENT",
        "Permanent Account Number",
        f"Name: {fake.name()}",
        f"PAN: {random_pan()}",
        f"DOB: {fake.date_of_birth(minimum_age=18, maximum_age=80)}"
    ]

    img = create_base_image(text)
    img = apply_augmentation(img)

    img.save(f"{OUTPUT_DIR}/pan_like/pan_{i}.jpg")

def generate_certificate(i):

    text = [
        "STATE GOVERNMENT CERTIFICATE",
        f"Name: {fake.name()}",
        f"District: {fake.city()}",
        f"Issue Date: {fake.date()}",
        f"Certificate ID: {random.randint(100000,999999)}"
    ]

    img = create_base_image(text)
    img = apply_augmentation(img)

    img.save(f"{OUTPUT_DIR}/certificate/cert_{i}.jpg")

def generate_dataset():

    for i in range(40):
        generate_aadhaar(i)

    for i in range(30):
        generate_pan(i)

    for i in range(30):
        generate_certificate(i)

generate_dataset()

print("Synthetic dataset generated successfully.")