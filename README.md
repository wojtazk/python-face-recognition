# Python Face Recognition
Analiza, rozpoznawanie oraz wyszukiwanie twarzy przy użyciu [deepface](https://pypi.org/project/deepface/) oraz [opencv](https://pypi.org/project/opencv-python/)

<br>

## Instalowanie niezbędnych pakietów
```shell
pip install -r requirements.txt
```

## Analiza twarzy
Należy zmodyfikować ścieżkę do folderu ze zdjęciami do analizy, w pliku `face_analyze_img.py`:
```python
IMG_DIRECTORY = '/home/wojtazk/Desktop/biometria_zdjecia'
```

<br>

Uruchamianie analizy twarzy:
```shell
python face_analyze_img.py
```
![face_analysis](https://github.com/user-attachments/assets/d1432eba-a7c9-4152-8d28-64669f4b8ccf)


## Znajdowanie podobnych twarzy
Należy zmodyfikować ścieżki do zdjęcia szukanej osoby i folderu ze zdjęciami, w pliku `face_find.py`:
```python
IMG_PATH = '/home/wojtazk/Desktop/Elon_Musk_Royal_Society_crop.jpg'
DB_PATH = '/home/wojtazk/Desktop/elon_musk'
```

<br>

Uruchamianie szukania twarzy:
```shell
python face_find.py
```
![face_finding](https://github.com/user-attachments/assets/3c1902c3-2227-4d55-b54c-a73ff58936ba)



## Weryfikacja twarzy - czy na zdjecie1 i zdjecie2 jest ta sama osoba
Należy zmodyfikować ścieżki do zdjęć w pliku `face_verify.py`
```python
IMG1_PATH = '/home/wojtazk/Desktop/Pope_John_Paul_II_smile.jpg'
IMG2_PATH = '/home/wojtazk/Desktop/Nancy_Reagan_and_Pope_John_Paul_II_(cropped).jpg'
```

<br>

Uruchamianie weryfikacji twarzy:
```shell
python face_verify.py
```
![face_verification](https://github.com/user-attachments/assets/6af596fc-169c-4ec1-8329-78cf57b3376d)



## Zdjęcia użyte w przykładach
- https://unsplash.com/photos/photography-of-five-people-near-outdoor-during-daytime-hOF1bWoet_Q
- https://commons.wikimedia.org/wiki/File:Elon_Musk_Royal_Society_crop.jpg
- https://commons.wikimedia.org/wiki/File:Elon_Musk_(12271223586).jpg
- https://commons.wikimedia.org/wiki/File:Elon_Musk_2015.jpg
- https://commons.wikimedia.org/wiki/File:Elon_Musk_(12271217906).jpg
- https://commons.wikimedia.org/wiki/File:Elon_Musk_2021.jpg
- https://commons.wikimedia.org/wiki/File:Elon_Musk_Brazil_2022.png
- https://commons.wikimedia.org/wiki/File:Pope_John_Paul_II_smile.jpg
- https://commons.wikimedia.org/wiki/File:Nancy_Reagan_and_Pope_John_Paul_II_(cropped).jpg


## Przydatne linki:
- https://thedatafrog.com/en/articles/human-detection-video/
<!-- - https://pypi.org/project/face-recognition/ -->
- https://pypi.org/project/deepface/
<!-- - https://www.kaggle.com/datasets/adg1822/7-celebrity-images -->
