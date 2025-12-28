# -*- coding: utf-8 -*-
"""
MNIST Model Playground
----------------------
Bu program:
- Eğitilmiş MNIST CNN modelini yükler
- Kullanıcının seçtiği PNG/JPG dosyasını alır
- Görüntüyü MNIST formatına dönüştürür
- Model ile tahmin yapar
- Sonucu ekranda gösterir
"""
from PIL import Image, ImageOps

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------------------
# 1) MODELİ YÜKLE
# ----------------------------
# Eğittiğin ve kaydettiğin model dosyası
MODEL_PATH = "mnist_cnn_best.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model yüklenemedi: {e}")

# ----------------------------
# 2) PNG'Yİ OKU + ÖN İŞLEME
# ----------------------------
def preprocess_image(image_path):
    """
    Daha MNIST-benzeri preprocessing:
    - Grayscale
    - Gerekirse invert (beyaz zemin/siyah çizim -> MNIST gibi siyah zemin/beyaz çizim)
    - Rakamı bbox ile kırp
    - Kareye pad et
    - 20x20'e ölçekle, 28x28 canvas'a ortala
    - Normalize + (1, 28, 28, 1)
    """

    # 1) PIL ile oku (daha kontrol bizde)
    img = Image.open(image_path).convert("L")

    # 2) Numpy'a al
    arr = np.array(img).astype(np.uint8)

    # 3) Arka plan beyaz mı? (ortalama yüksekse) invert et
    if arr.mean() > 127:
        arr = 255 - arr

    # 4) Basit threshold ile bbox bul (rakamı yakala)
    mask = arr > 20  # 0-255 arası; 20 iyi başlangıç
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        arr = arr[y0:y1, x0:x1]
    else:
        # hiç çizim yoksa direkt boş dön
        arr = np.zeros((28, 28), dtype=np.uint8)

    # 5) Kareye pad et (aspect bozulmasın)
    h, w = arr.shape
    size = max(h, w)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2
    arr = np.pad(
        arr,
        ((pad_y, size - h - pad_y), (pad_x, size - w - pad_x)),
        mode="constant",
        constant_values=0
    )

    # 6) 20x20'e ölçekle, 28x28 içine ortala (MNIST stili)
    pil_sq = Image.fromarray(arr)
    pil_20 = pil_sq.resize((20, 20), Image.BILINEAR)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(pil_20, (4, 4))  # 28-20=8 => her yandan 4 px

    # Debug: modele giden görüntüyü kaydet
    canvas.save("input_28x28.png", "PNG")

    # 7) Normalize + shape
    x = np.array(canvas).astype(np.float32) / 255.0
    x = x.reshape(1, 28, 28, 1)
    return x


# ----------------------------
# 3) TAHMİN YAP
# ----------------------------
def predict_digit(image_path):
    processed = preprocess_image(image_path)

    # Model tahmini (olasılıklar)
    probs = model.predict(processed, verbose=0)

    # En yüksek olasılıklı sınıf
    prediction = int(np.argmax(probs, axis=1)[0])

    return prediction, probs[0]


# ----------------------------
# 4) DOSYA SEÇ + SONUCU GÖSTER
# ----------------------------
def open_and_predict():
    # Dosya seçme penceresi
    file_path = filedialog.askopenfilename(
        title="Bir rakam görseli seç",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    try:
        prediction, probs = predict_digit(file_path)

        # Sonucu popup ile göster
        messagebox.showinfo(
            "Tahmin Sonucu",
            f"Tahmin edilen rakam: {prediction}"
        )

        # Görsel + olasılıkları göster (debug / öğrenme amaçlı)
        show_debug(file_path, probs)

    except Exception as e:
        messagebox.showerror("Hata", str(e))


# ----------------------------
# 5) DEBUG GÖRÜNÜMÜ (isteğe bağlı ama çok faydalı)
# ----------------------------
def show_debug(image_path, probs):
    """
    - Modele giden 28x28 görüntüyü gösterir
    - 0–9 olasılıklarını bar chart olarak çizer
    """

    processed = preprocess_image(image_path)

    plt.figure(figsize=(10, 4))

    # Görüntü
    plt.subplot(1, 2, 1)
    plt.imshow(processed[0].squeeze(), cmap="gray")
    plt.title("Modele Giden Görüntü")
    plt.axis("off")

    # Olasılıklar
    plt.subplot(1, 2, 2)
    plt.bar(range(10), probs)
    plt.xticks(range(10))
    plt.title("Sınıf Olasılıkları")
    plt.xlabel("Rakam")
    plt.ylabel("Olasılık")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ----------------------------
    # 6) TKINTER ARAYÜZÜ
    # ----------------------------
    root = tk.Tk()
    root.title("MNIST Playground")
    root.geometry("300x200")

    label = tk.Label(
        root,
        text="El yazısı rakam görseli yükle",
        font=("Arial", 12)
    )
    label.pack(pady=20)

    button = tk.Button(
        root,
        text="PNG / JPG Seç ve Tahmin Et",
        command=open_and_predict,
        width=25,
        height=2
    )
    button.pack()

    root.mainloop()
