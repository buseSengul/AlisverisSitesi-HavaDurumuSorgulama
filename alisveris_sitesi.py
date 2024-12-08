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