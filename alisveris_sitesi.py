class Urun:
    def __init__(self, ad, fiyat, stok):
        self.ad = ad
        self.fiyat = fiyat
        self.stok = stok

    def __str__(self):
        return f"{self.ad} - Fiyat: {self.fiyat} TL - Stok: {self.stok} adet"


class Sepet:
    def __init__(self):
        self.urunler = []

    def urun_ekle(self, urun, miktar):
        if miktar > urun.stok:
            print(f"Yetersiz stok! {urun.ad} ürününden en fazla {urun.stok} adet alabilirsiniz.")
            return

        for mevcut_urun in self.urunler:
            if mevcut_urun.ad == urun.ad:
                mevcut_urun.stok += miktar
                urun.stok -= miktar
                print(f"{urun.ad} miktarı güncellendi. Yeni miktar: {mevcut_urun.stok}")
                return

        yeni_urun = Urun(urun.ad, urun.fiyat, miktar)
        self.urunler.append(yeni_urun)
        urun.stok -= miktar
        print(f"{urun.ad} sepete eklendi. Alınan miktar: {miktar} adet")

    def urun_cikar(self, urun_adi):
        for urun in self.urunler:
            if urun.ad == urun_adi:
                self.urunler.remove(urun)
                print(f"{urun_adi} sepetten çıkarıldı.")
                return
        print(f"{urun_adi} sepetinizde bulunamadı.")

    def sepeti_listele(self):
        if not self.urunler:
            print("Sepetiniz boş.")
            return
        print("\nSepetinizdeki Ürünler:")
        for urun in self.urunler:
            print(f"- {urun.ad}: {urun.stok} adet, {urun.fiyat} TL/adet, Toplam: {urun.fiyat * urun.stok} TL")
        print(f"Toplam Tutar: {self.toplam_tutar()} TL\n")

    def toplam_tutar(self):
        return sum(urun.fiyat * urun.stok for urun in self.urunler)


def menu():
    print("\n----- Menü -----")
    print("1. Ürünleri Görüntüle")
    print("2. Ürün Satın Al")
    print("3. Sepeti Listele")
    print("4. Ürün Sepetten Çıkar")
    print("5. Toplam Tutarı Gör")
    print("6. Çıkış")
    print("----------------")


def kullanici_girisi():
    print("----- Kullanıcı Girişi -----")
    dogru_kullanici_adi = "admin"
    dogru_sifre = "1234"

    while True:
        kullanici_adi = input("Kullanıcı Adı: ")
        sifre = input("Şifre: ")

        if kullanici_adi == dogru_kullanici_adi and sifre == dogru_sifre:
            print("Giriş başarılı!")
            break
        else:
            print("Hatalı kullanıcı adı veya şifre. Tekrar deneyin.")


def uygulama():
    # Kullanıcı girişi
    kullanici_girisi()

    # Stokta bulunan ürünleri oluşturma
    urunler = [
        Urun("Kitap", 50, 10),
        Urun("Kalem", 5, 20),
        Urun("Defter", 20, 15),
        Urun("Çanta", 150, 5),
        Urun("Kulaklık", 200, 8)
    ]

    sepet = Sepet()

    while True:
        menu()
        try:
            secim = int(input("Bir işlem seçin (1-6): "))
        except ValueError:
            print("Geçersiz giriş. Lütfen 1 ile 6 arasında bir sayı girin.")
            continue

        if secim == 1:  # Ürünleri Görüntüle
            print("\n--- Mevcut Ürünler ---")
            for i, urun in enumerate(urunler, start=1):
                print(f"{i}. {urun}")
            print("----------------------")

        elif secim == 2:  # Ürün Satın Al
            print("\n--- Mevcut Ürünler ---")
            for i, urun in enumerate(urunler, start=1):
                print(f"{i}. {urun}")
            print("----------------------")

            try:
                urun_no = int(input("Satın almak istediğiniz ürünün numarasını girin: "))
                if urun_no < 1 or urun_no > len(urunler):
                    print("Geçersiz ürün numarası.")
                    continue

                miktar = int(input(f"Kaç adet {urunler[urun_no - 1].ad} satın almak istiyorsunuz? "))
                sepet.urun_ekle(urunler[urun_no - 1], miktar)
            except ValueError:
                print("Geçersiz giriş. Lütfen sayısal bir değer girin.")
                continue

        elif secim == 3:  # Sepeti Listele
            sepet.sepeti_listele()

        elif secim == 4:  # Ürün Sepetten Çıkar
            urun_adi = input("Çıkarmak istediğiniz ürünün adını girin: ")
            sepet.urun_cikar(urun_adi)

        elif secim == 5:  # Toplam Tutarı Gör
            print(f"Sepet toplam tutarı: {sepet.toplam_tutar()} TL")

        elif secim == 6:  # Çıkış
            print("Programdan çıkılıyor...")
            break

        else:
            print("Geçersiz seçim. Lütfen 1 ile 6 arasında bir sayı girin.")


# Ana program
uygulama() 


#verisetini çoğaltma
# import os
# import cv2
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# def rotate_image(image, angle):
#     h, w = image.shape
#     center = (w // 2, h // 2)
#     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

# def zoom_in(image, percent=0.9):
#     h, w = image.shape
#     crop_h, crop_w = int(h * percent), int(w * percent)
#     start_y = (h - crop_h) // 2
#     start_x = (w - crop_w) // 2
#     cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
#     return cv2.resize(cropped, (w, h))

# def augment_image(image):
#     aug_images = {}

#     aug_images['rot+10'] = rotate_image(image, 10)
#     aug_images['rot-10'] = rotate_image(image, -10)
#     aug_images['flip'] = cv2.flip(image, 1)
#     aug_images['blur'] = cv2.GaussianBlur(image, (5, 5), 0)
#     aug_images['bright+'] = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
#     aug_images['bright-'] = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
#     aug_images['zoom'] = zoom_in(image, percent=0.9)

#     return aug_images

# def main():
#     dataset_dir = 'dataset'
#     location_file = os.path.join(dataset_dir, 'locations.csv')
#     df = pd.read_csv(location_file)

#     augmented_rows = []

#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         filename = row['filename']
#         lat, lon = row['latitude'], row['longitude']
#         image_path = os.path.join(dataset_dir, filename)

#         if not os.path.exists(image_path):
#             print(f" Dosya bulunamadı: {filename}")
#             continue

#         gray = cv2.imread(image_path, 0)
#         aug_images = augment_image(gray)

#         for aug_name, aug_img in aug_images.items():
#             aug_filename = f"{os.path.splitext(filename)[0]}_{aug_name}.png"
#             aug_path = os.path.join(dataset_dir, aug_filename)
#             cv2.imwrite(aug_path, aug_img)
#             augmented_rows.append({'filename': aug_filename, 'latitude': lat, 'longitude': lon})

#     # Yeni CSV oluştur
#     df_aug = pd.DataFrame(augmented_rows)
#     df_full = pd.concat([df, df_aug], ignore_index=True)
#     df_full.to_csv(location_file, index=False)
#     print(f"\n✅ Veri artırımı tamamlandı. Yeni görüntü sayısı: {len(df_full)}")

# if __name__ == "__main__":
#     main()



----------------------------------------

# #kameradan anlık tek görüntü çekerek eşleştirme
# import os
# import cv2
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pickle
# from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
# from gluestick.models.two_view_pipeline import TwoViewPipeline
# from gluestick.drawing import plot_images, plot_matches, plot_color_line_matches

# # ---------- ORB ----------
# orb = cv2.ORB_create(nfeatures=500)

# def extract_orb_vector(gray):
#     kp, des = orb.detectAndCompute(gray, None)
#     if des is None:
#         return np.zeros((32,), dtype=np.float32)
#     return des.mean(axis=0)

# def cosine_similarity(v1, v2):
#     return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

# # ---------- Görselleştirme ----------
# def visualize_and_score(pipeline, img0_gray, img1_gray, img0_tensor, img1_tensor, out_path):
#     x = {'image0': img0_tensor, 'image1': img1_tensor}
#     pred = pipeline(x)
#     pred = batch_to_np(pred)

#     kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
#     m0 = pred["matches0"]
#     valid_kp = m0 != -1
#     matched_kp0 = kp0[valid_kp]
#     matched_kp1 = kp1[m0[valid_kp]]
#     keypoint_score = valid_kp.sum()

#     line_matches = pred["line_matches0"]
#     lines0, lines1 = pred["lines0"], pred["lines1"]
#     valid_lines = line_matches != -1
#     matched_lines0 = lines0[valid_lines]
#     matched_lines1 = lines1[line_matches[valid_lines]]
#     line_score = valid_lines.sum()

#     img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
#     img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

#     plot_images([img0, img1], ['Webcam', 'Dataset'], dpi=150, pad=2.0)
#     plot_matches(matched_kp0, matched_kp1, 'lime', lw=1, ps=1)
#     plt.savefig(os.path.join(out_path, "points.png"))
#     plt.close()

#     plot_images([img0, img1], ['Webcam', 'Dataset'], dpi=150, pad=2.0)
#     plot_color_line_matches([matched_lines0, matched_lines1], lw=2)
#     plt.savefig(os.path.join(out_path, "lines.png"))
#     plt.close()

#     return 1.0 * keypoint_score + 0.5 * line_score, keypoint_score, line_score

# # ---------- Kamera ----------
# def capture_frame(device=0):
#     cap = cv2.VideoCapture(device)
#     if not cap.isOpened():
#         raise RuntimeError("Kamera açılamadı")
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         raise RuntimeError("Görüntü alınamadı")
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (640, 480))
#     gray = cv2.equalizeHist(gray)
#     return gray

# # ---------- Ana Fonksiyon ----------
# def main():
#     dataset_dir = 'dataset'
#     cache_path = os.path.join(dataset_dir, 'orb_vectors.pkl')
#     output_dir = 'debug_outputs'
#     os.makedirs(output_dir, exist_ok=True)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     query_gray = capture_frame()
#     query_tensor = numpy_image_to_torch(query_gray).to(device)[None]
#     query_vector = extract_orb_vector(query_gray)

#     # GlueStick config
#     conf = {
#         'name': 'two_view_pipeline',
#         'use_lines': True,
#         'extractor': {
#             'name': 'wireframe',
#             'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
#             'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
#             'max_n_lines': 300,
#         },
#         'matcher': {
#             'name': 'gluestick',
#             'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
#             'trainable': False,
#         },
#         'ground_truth': {'from_pose_depth': False},
#     }

#     pipeline = TwoViewPipeline(conf).to(device).eval()

#     # ORB vector cache
#     if os.path.exists(cache_path):
#         with open(cache_path, 'rb') as f:
#             vec_dict = pickle.load(f)
#     else:
#         vec_dict = {}
#         for file in os.listdir(dataset_dir):
#             if file.endswith('.png'):
#                 img = cv2.imread(os.path.join(dataset_dir, file), 0)
#                 vec_dict[file] = extract_orb_vector(img)
#         with open(cache_path, 'wb') as f:
#             pickle.dump(vec_dict, f)

#     # Benzerlik listesi
#     similarities = [(fname, cosine_similarity(query_vector, vec)) for fname, vec in vec_dict.items()]
#     top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:15]

#     results = []
#     for fname, _ in top_matches:
#         target_gray = cv2.imread(os.path.join(dataset_dir, fname), 0)
#         target_tensor = numpy_image_to_torch(target_gray).to(device)[None]

#         result_dir = os.path.join(output_dir, os.path.splitext(fname)[0])
#         os.makedirs(result_dir, exist_ok=True)

#         score, kp, line = visualize_and_score(pipeline, query_gray, target_gray, query_tensor, target_tensor, result_dir)
#         print(f" {fname}: {kp} kp + {line} lines → score={score:.2f}")
#         results.append((fname, score))

#     best = max(results, key=lambda x: x[1])
#     print(f"\n Best match: {best[0]} | Score: {best[1]:.2f}")

#     # Konum
#     loc_path = os.path.join(dataset_dir, 'locations.csv')
#     if os.path.exists(loc_path):
#         df = pd.read_csv(loc_path)
#         row = df[df['filename'] == best[0]]
#         if not row.empty:
#             lat, lon = row.iloc[0]['latitude'], row.iloc[0]['longitude']
#             print(f" Location: Latitude = {lat}, Longitude = {lon}")
#         else:
#             print(" Konum bulunamadı.")
#     else:
#         print(" locations.csv dosyası eksik.")

# if __name__ == "__main__":
#     main()




--------------------------------------------------
import os
import cv2
import torch
import time
import numpy as np
import pandas as pd
import pickle

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.models.two_view_pipeline import TwoViewPipeline


# ORB vektör çıkarma
orb = cv2.ORB_create(nfeatures=500)

def extract_orb_vector(gray):
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        return np.zeros((32,), dtype=np.float32)
    return des.mean(axis=0)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

def load_dataset_vectors(dataset_dir, cache_file='orb_vectors.pkl'):
    cache_path = os.path.join(dataset_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    vecs = {}
    for file in os.listdir(dataset_dir):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(dataset_dir, file), 0)
            vecs[file] = extract_orb_vector(img)
    with open(cache_path, 'wb') as f:
        pickle.dump(vecs, f)
    return vecs

def best_gluestick_match(pipeline, query_gray, query_tensor, dataset_dir, top_filenames, device):
    best_score = -1
    best_filename = None
    best_img = None

    for fname in top_filenames:
        img_path = os.path.join(dataset_dir, fname)
        img = cv2.imread(img_path, 0)
        if img is None:
            continue
        img_tensor = numpy_image_to_torch(img).to(device)[None]

        x = {'image0': query_tensor, 'image1': img_tensor}
        pred = pipeline(x)
        pred = batch_to_np(pred)

        kp_score = (pred['matches0'] != -1).sum()
        line_score = (pred['line_matches0'] != -1).sum()
        total_score = kp_score * 1.0 + line_score * 0.5

        if total_score > best_score:
            best_score = total_score
            best_filename = fname
            best_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return best_filename, best_score, best_img

def get_location_from_csv(csv_path, filename):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    row = df[df['filename'] == filename]
    if not row.empty:
        return row.iloc[0]['latitude'], row.iloc[0]['longitude']
    return None

def main():
    dataset_dir = 'dataset'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vector_dict = load_dataset_vectors(dataset_dir)

    # GlueStick pipeline
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
            'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
            'max_n_lines': 300,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {'from_pose_depth': False},
    }
    pipeline = TwoViewPipeline(conf).to(device).eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("🚫 Kamera açılamadı.")
        return

    last_time = 0
    delay = 2.0  # saniye

    print("🎥 Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (640, 480))
        query_gray = cv2.equalizeHist(resized)
        query_tensor = numpy_image_to_torch(query_gray).to(device)[None]
        query_vec = extract_orb_vector(query_gray)

        current_time = time.time()
        if current_time - last_time > delay:
            # ORB ön eleme
            similarities = [(f, cosine_similarity(query_vec, vec)) for f, vec in vector_dict.items()]
            top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
            top_filenames = [f for f, _ in top_matches]

            best_fname, score, best_img = best_gluestick_match(pipeline, query_gray, query_tensor, dataset_dir, top_filenames, device)
            location = get_location_from_csv(os.path.join(dataset_dir, 'locations.csv'), best_fname)

            # Görselleştirme
            display = frame.copy()
            text = f"Best Match: {best_fname or 'None'} | Score: {score:.2f}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            if location:
                loc_text = f"Lat: {location[0]:.5f}, Lon: {location[1]:.5f}"
                cv2.putText(display, loc_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if best_img is not None:
                top = cv2.resize(display, (640, 360))
                bottom = cv2.resize(best_img, (640, 360))
                combined = np.vstack((top, bottom))
                cv2.imshow("📍 Live Matching", combined)
            else:
                cv2.imshow("📍 Live Matching", display)

            last_time = current_time
        else:
            cv2.imshow("📍 Live Matching", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# import os
# import cv2
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt

# from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
# from gluestick.models.two_view_pipeline import TwoViewPipeline
# from gluestick.drawing import plot_images, plot_matches, plot_color_line_matches


# def capture_webcam_frame(device=0):
#     cap = cv2.VideoCapture(device)
#     if not cap.isOpened():
#         raise RuntimeError("Kamera açılamadı")
    
#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         raise RuntimeError("Kare alınamadı")

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return gray


# def load_image_as_tensor_from_array(gray_img, device):
#     torch_img = numpy_image_to_torch(gray_img).to(device)[None]
#     return gray_img, torch_img


# def load_image_as_tensor_from_file(path, device):
#     gray = cv2.imread(path, 0)
#     torch_img = numpy_image_to_torch(gray).to(device)[None]
#     return gray, torch_img


# def compute_score_and_visualize(pipeline, img0_gray, img1_gray, img0_tensor, img1_tensor, out_path):
#     x = {'image0': img0_tensor, 'image1': img1_tensor}
#     pred = pipeline(x)
#     pred = batch_to_np(pred)

#     # Nokta eşleşmeleri
#     kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
#     m0 = pred["matches0"]
#     valid_kp = m0 != -1
#     matched_kp0 = kp0[valid_kp]
#     matched_kp1 = kp1[m0[valid_kp]]
#     keypoint_score = valid_kp.sum()

#     # Çizgi eşleşmeleri
#     line_matches = pred["line_matches0"]
#     lines0, lines1 = pred["lines0"], pred["lines1"]
#     valid_lines = line_matches != -1
#     matched_lines0 = lines0[valid_lines]
#     matched_lines1 = lines1[line_matches[valid_lines]]
#     line_score = valid_lines.sum()

#     # Görselleri BGR yap
#     img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
#     img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

#     # Nokta eşleşmeleri
#     plot_images([img0, img1], ['Query (Webcam)', 'Dataset'], dpi=150, pad=2.0)
#     plot_matches(matched_kp0, matched_kp1, 'lime', lw=1, ps=1)
#     plt.savefig(os.path.join(out_path, "point_matches.png"))
#     plt.close()

#     # Çizgi eşleşmeleri
#     plot_images([img0, img1], ['Query (Webcam)', 'Dataset'], dpi=150, pad=2.0)
#     plot_color_line_matches([matched_lines0, matched_lines1], lw=2)
#     plt.savefig(os.path.join(out_path, "line_matches.png"))
#     plt.close()

#     total_score = 1.0 * keypoint_score + 0.5 * line_score
#     return total_score, keypoint_score, line_score


# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     dataset_dir = 'dataset'
#     output_dir = 'live_outputs'
#     os.makedirs(output_dir, exist_ok=True)

#     print("📷 Kameradan görüntü alınıyor...")
#     query_gray = capture_webcam_frame()
#     _, query_tensor = load_image_as_tensor_from_array(query_gray, device)

#     # GlueStick pipeline konfigürasyonu
#     conf = {
#         'name': 'two_view_pipeline',
#         'use_lines': True,
#         'extractor': {
#             'name': 'wireframe',
#             'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
#             'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
#             'max_n_lines': 300,
#         },
#         'matcher': {
#             'name': 'gluestick',
#             'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
#             'trainable': False,
#         },
#         'ground_truth': {'from_pose_depth': False},
#     }

#     pipeline = TwoViewPipeline(conf).to(device).eval()

#     scores = {}

#     for filename in os.listdir(dataset_dir):
#         if filename.endswith('.png'):
#             print(f" Karşılaştırılıyor: {filename}")
#             img_path = os.path.join(dataset_dir, filename)
#             target_gray, target_tensor = load_image_as_tensor_from_file(img_path, device)

#             result_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
#             os.makedirs(result_dir, exist_ok=True)

#             total_score, kp_score, line_score = compute_score_and_visualize(
#                 pipeline, query_gray, target_gray, query_tensor, target_tensor, result_dir
#             )

#             scores[filename] = {
#                 'total': total_score,
#                 'keypoints': kp_score,
#                 'lines': line_score,
#                 'output_dir': result_dir
#             }

#             print(f"{filename}: {kp_score} keypoints + {line_score} lines → score = {total_score:.2f}")

#     best_match = max(scores, key=lambda k: scores[k]['total'])
#     best = scores[best_match]
#     print(f"\n En iyi eşleşme: {best_match} | Skor: {best['total']:.2f}")

#     loc_csv = os.path.join(dataset_dir, 'locations.csv')
#     if os.path.exists(loc_csv):
#         df = pd.read_csv(loc_csv)
#         loc_row = df[df['filename'] == best_match]
#         if not loc_row.empty:
#             lat, lon = loc_row.iloc[0]['latitude'], loc_row.iloc[0]['longitude']
#             print(f' Konum: Enlem = {lat}, Boylam = {lon}')
#         else:
#             print(' CSV dosyasında konum bulunamadı.')
#     else:
#         print(' locations.csv bulunamadı.')


# if __name__ == "__main__":
#     main()







