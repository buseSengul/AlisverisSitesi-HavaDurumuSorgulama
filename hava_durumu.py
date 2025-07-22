

# import os
# import cv2
# import torch
# import pandas as pd
# from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
# from gluestick.models.two_view_pipeline import TwoViewPipeline


# def load_image_as_tensor(path, device):
#     gray = cv2.imread(path, 0)
#     torch_img = numpy_image_to_torch(gray).to(device)[None]
#     return gray, torch_img


# def compute_combined_score(pipeline_model, img1_tensor, img2_tensor, point_weight=1.0, line_weight=0.5):
#     x = {'image0': img1_tensor, 'image1': img2_tensor}
#     pred = pipeline_model(x)
#     pred = batch_to_np(pred)

#     # Nokta eşleşmeleri
#     kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
#     m0 = pred["matches0"]
#     valid_kp = m0 != -1
#     keypoint_score = valid_kp.sum()

#     # Çizgi eşleşmeleri
#     line_matches = pred["line_matches0"]
#     valid_line = line_matches != -1
#     line_score = valid_line.sum()

#     # Kombine skor
#     total_score = point_weight * keypoint_score + line_weight * line_score
#     return total_score, keypoint_score, line_score


# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     query_path = 'query.png'  # referans görüntü yolu
#     dataset_dir = 'dataset'

#     # GlueStick konfigürasyonu
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

#     # Referans görüntüyü yükle
#     ref_gray, ref_tensor = load_image_as_tensor(query_path, device)

#     scores = {}

#     for filename in os.listdir(dataset_dir):
#         if filename.endswith('.png'):
#             path = os.path.join(dataset_dir, filename)
#             target_gray, target_tensor = load_image_as_tensor(path, device)
#             total_score, kp_score, line_score = compute_combined_score(pipeline, ref_tensor, target_tensor)

#             scores[filename] = {
#                 'total': total_score,
#                 'keypoints': kp_score,
#                 'lines': line_score
#             }

#             print(f'{filename}: {kp_score} keypoints + {line_score} lines → score = {total_score:.2f}')

#     # En iyi eşleşmeyi bul
#     best_match = max(scores, key=lambda k: scores[k]['total'])
#     best = scores[best_match]

#     print(f'\n Best Match: {best_match}')
#     print(f'Score: {best["total"]:.2f} ({best["keypoints"]} keypoints + {best["lines"]} lines)')

#     # Konum bilgisini CSV'den al
#     df = pd.read_csv(os.path.join(dataset_dir, 'locations.csv'))
#     loc_row = df[df['filename'] == best_match]
#     if not loc_row.empty:
#         lat, lon = loc_row.iloc[0]['latitude'], loc_row.iloc[0]['longitude']
#         print(f' Location: Latitude = {lat}, Longitude = {lon}')
#     else:
#         print(' Location not found in CSV.')


# if __name__ == "__main__":
#     main()




#bir referans göreselini tek tek karşılaştırarak eşliyor
import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.models.two_view_pipeline import TwoViewPipeline
from gluestick.drawing import plot_images, plot_matches, plot_color_line_matches


def load_image_as_tensor(path, device):
    gray = cv2.imread(path, 0)
    torch_img = numpy_image_to_torch(gray).to(device)[None]
    return gray, torch_img


def visualize_and_score(pipeline_model, img0_gray, img1_gray, img0_tensor, img1_tensor, out_path):
    x = {'image0': img0_tensor, 'image1': img1_tensor}
    pred = pipeline_model(x)
    pred = batch_to_np(pred)

    # Keypoint eşleşmeleri
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    valid_kp = m0 != -1
    match_indices = m0[valid_kp]
    matched_kp0 = kp0[valid_kp]
    matched_kp1 = kp1[match_indices]
    keypoint_score = valid_kp.sum()

    # Line eşleşmeleri
    line_matches = pred["line_matches0"]
    lines0, lines1 = pred["lines0"], pred["lines1"]
    valid_lines = line_matches != -1
    matched_lines0 = lines0[valid_lines]
    matched_lines1 = lines1[line_matches[valid_lines]]
    line_score = valid_lines.sum()

    # Görselleri renkli hale getir
    img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

    # Nokta eşleşmesi görseli
    plot_images([img0, img1], ['Query', 'Dataset'], dpi=150, pad=2.0)
    plot_matches(matched_kp0, matched_kp1, 'lime', lw=1, ps=1)
    plt.savefig(os.path.join(out_path, "point_matches.png"))
    plt.close()

    # Çizgi eşleşmesi görseli
    plot_images([img0, img1], ['Query', 'Dataset'], dpi=150, pad=2.0)
    plot_color_line_matches([matched_lines0, matched_lines1], lw=2)
    plt.savefig(os.path.join(out_path, "line_matches.png"))
    plt.close()

    total_score = 1.0 * keypoint_score + 0.5 * line_score
    return total_score, keypoint_score, line_score


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_path = 'query.png'
    dataset_dir = 'dataset'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # GlueStick config
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

    ref_gray, ref_tensor = load_image_as_tensor(query_path, device)
    scores = {}

    for filename in os.listdir(dataset_dir):
        if filename.endswith('.png'):
            print(f"\n🔄 Processing: {filename}")
            dataset_path = os.path.join(dataset_dir, filename)
            target_gray, target_tensor = load_image_as_tensor(dataset_path, device)

            result_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(result_dir, exist_ok=True)

            total_score, kp_score, line_score = visualize_and_score(
                pipeline, ref_gray, target_gray, ref_tensor, target_tensor, result_dir
            )

            scores[filename] = {
                'total': total_score,
                'keypoints': kp_score,
                'lines': line_score,
                'output_dir': result_dir
            }

            print(f"📌 {filename}: {kp_score} keypoints + {line_score} lines → score = {total_score:.2f}")

    # En iyi eşleşen görseli bul
    best_match = max(scores, key=lambda k: scores[k]['total'])
    best = scores[best_match]
    print(f"\n✅ Best match: {best_match} with score {best['total']:.2f}")

    # Konumu yazdır
    df = pd.read_csv(os.path.join(dataset_dir, 'locations.csv'))
    loc_row = df[df['filename'] == best_match]
    if not loc_row.empty:
        lat, lon = loc_row.iloc[0]['latitude'], loc_row.iloc[0]['longitude']
        print(f' Location: Latitude = {lat}, Longitude = {lon}')
    else:
        print(' Location not found in CSV.')


if __name__ == "__main__":
    main()

------------------------------------
class Sehir:
    def __init__(self, ad, sicaklik):
        self.ad = ad
        self.sicaklik = sicaklik

class HavaDurumu:
    def __init__(self):
        self.sehirler = {
            "İstanbul": Sehir("İstanbul", 12),
            "Ankara": Sehir("Ankara", -2),
            "İzmir": Sehir("İzmir", 20),
            "Antalya": Sehir("Antalya", 25),
            "Trabzon": Sehir("Trabzon", 16)
        }

    def sehir_ekle(self, sehir_adi, sicaklik):
        if sehir_adi in self.sehirler:
            print(f"{sehir_adi} zaten mevcut, sıcaklık güncelleniyor.")
        self.sehirler[sehir_adi] = Sehir(sehir_adi, sicaklik)
        print(f"{sehir_adi} eklendi ve sıcaklık {sicaklik}°C olarak ayarlandı.")

    def sicaklik_guncelle(self, sehir_adi, yeni_sicaklik):
        if sehir_adi in self.sehirler:
            self.sehirler[sehir_adi].sicaklik = yeni_sicaklik
            print(f"{sehir_adi} için sıcaklık {yeni_sicaklik}°C olarak güncellendi.")
        else:
            print(f"{sehir_adi} bulunamadı. Lütfen önce şehri ekleyin.")

    def hava_durumu_sorgula(self, sehir_adi):
        if sehir_adi in self.sehirler:
            sehir = self.sehirler[sehir_adi]
            tavsiye = self.tavsiye_ver(sehir.sicaklik)
            print(f"{sehir.ad} için hava durumu: {sehir.sicaklik}°C. {tavsiye}")
        else:
            print(f"{sehir_adi} bulunamadı.")

    def tavsiye_ver(self, sicaklik):
        if sicaklik < 0:
            return "Soğuk, sıkı giyinin."
        elif 0 <= sicaklik <= 15:
            return "Serin, mont almayı unutmayın."
        else:
            return "Hava güzel, rahat giyin."

def menu():
    print("\n--- Hava Durumu Uygulaması ---")
    print("1. Şehir ekle")
    print("2. Şehir sıcaklığını güncelle")
    print("3. Şehir hava durumu sorgula")
    print("4. Çıkış")
    
    secim = input("Yapmak istediğiniz işlemi seçin (1-4): ")
    return secim

def main():
    hava_durumu = HavaDurumu()
    
    while True:
        secim = menu()
        
        if secim == '1':  # Şehir ekleme
            sehir_adi = input("Eklemek istediğiniz şehri girin: ")
            try:
                sicaklik = float(input(f"{sehir_adi} için sıcaklık değerini girin: "))
                hava_durumu.sehir_ekle(sehir_adi, sicaklik)
            except ValueError:
                print("Geçerli bir sıcaklık değeri girin!")
        
        elif secim == '2':  # Sıcaklık güncelleme
            sehir_adi = input("Sıcaklığını güncellemek istediğiniz şehri girin: ")
            try:
                yeni_sicaklik = float(input(f"{sehir_adi} için yeni sıcaklık değerini girin: "))
                hava_durumu.sicaklik_guncelle(sehir_adi, yeni_sicaklik)
            except ValueError:
                print("Geçerli bir sıcaklık değeri girin!")
        
        elif secim == '3':  # Hava durumu sorgulama
            sehir_adi = input("Hava durumu sorgulamak istediğiniz şehri girin: ")
            hava_durumu.hava_durumu_sorgula(sehir_adi)
        
        elif secim == '4':  # Çıkış
            print("Uygulamadan çıkılıyor...")
            break
        
        else:
            print("Geçersiz seçim! Lütfen 1-4 arasında bir seçenek girin.")

if __name__ == "__main__":
    main() 
