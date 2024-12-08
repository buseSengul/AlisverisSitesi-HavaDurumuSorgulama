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