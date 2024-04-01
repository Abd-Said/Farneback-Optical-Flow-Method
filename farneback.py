import numpy as np
import cv2

# Video dosyasını aç
cap = cv2.VideoCapture(0)

# Optical flow parametreleri
farneback_params = dict(pyr_scale=0.5,  # Görüntü piramidinin ölçek faktörü
                        levels=3,       # Piramid seviye sayısı
                        winsize=15,     # Her bir iterasyonda kullanılan pencere boyutu
                        iterations=3,   # Her seviyedeki iterasyon sayısı
                        poly_n=5,       # Polinom genişliği
                        poly_sigma=1.2, # Polinom standart sapması
                        flags=0)        # Ek kontrol flagleri (cv2.OPTFLOW_USE_INITIAL_FLOW, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

# İlk kareyi oku ve griye dönüştür
ret, frame1 = cap.read()
prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Video boyunca döngü
while True:
    
    # Bir sonraki kareyi oku ve griye dönüştür
    ret, frame2 = cap.read()
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optik akışı hesapla
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, **farneback_params)

    # Optik akış vektörlerini görselleştir
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Akış vektörlerini görselleştirilmiş kareyi göster
    cv2.imshow('Optical Flow', rgb)

    # Bir sonraki kareye geç
    prev_frame = next_frame.copy()

    # Çıkış için 'q' tuşuna basılıp basılmadığını kontrol et
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
