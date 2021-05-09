# "Reconnaissance d’émotions sur un visage"

Lien du dataset disponible ici : https://www.kaggle.com/msambare/fer2013

Lien des CSV déjà prêt ici : https://drive.google.com/drive/folders/1audrS6_TG7IjBRYRswDWtzrRjHRUVo-c?usp=sharing

Le dataset est à placer dans le répertoire "Dossier" présent à la racine de ce projet.
Les CSV sont à placer à la racine de ce projet. 

Projet réalisé avec NumPy, OpenCV, et Keras + Tensorflow.
Utilisation de Google Collab pour entraîner le model, et utilisation de Jupyter Notebook en local. 

## Notebook 

L'archive et les CSV sont indiqués juste au-dessus, le model est quant à lui directement inclus à la racine de ce répertoire.

## Comment lancer les deux scripts

### Script Image 

python scriptImage.py 

Une fois la caméra lancée, appuyer sur espace pour prendre une photo. 
Se rendre sur ./StockImagesOpenCV/normal pour obtenir le résultat de sa photo. 
Pour se faire, des images sont stockées en continues sur ./StockImagesOpenCV/48x48 puis supprimés lorsqu'on quitte le script. 
Pour quitter, appuyer sur ECHAP. 

### Script vidéo 

python scriptVideo.py

Une fois la caméra lancée, affichage en temps réel sur le visage les émotions de l'utilisateur.
Pour se faire, des images sont stockées en continues sur ./StockImagesOpenCV/48x48 puis supprimés lorsqu'on quitte le script. 
Pour quitter, appuyer sur ECHAP. 