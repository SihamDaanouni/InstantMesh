# InstantMesh : Déploiement, Évaluation et Optimisation

## Présentation du projet

Ce projet d'ingénierie, réalisé dans le cadre du cursus à **CY Tech**, porte sur l'étude et l'amélioration d'**InstantMesh**, un framework permettant la génération de modèles 3D à partir d'une seule image 2D.

L'objectif a été de stabiliser un environnement d'exécution complexe, de comparer l'efficacité de différents modèles de diffusion multi-vues (Zero123++ et SyncDreamer) et d'optimiser la qualité géométrique des maillages via l'intégration de techniques de super-résolution.

## Équipe du projet

* **Amine AIT MOUSSA**
* **Victor ANDRE**
* **Youenn BOGAER**
* **Siham DAANOUNI**
* **Ewen DANO**
* **Clément DELAMOTTE**

**Encadrant :** PhD Arthur SATOUF

**Date :** 9 mai 2026

## Contenu du dépôt

Le dépôt est organisé autour de trois composants principaux pour l'exécution et l'expérimentation :

* **run.py** : Script principal permettant de lancer l'inférence des modèles de génération 3D. Tout en optimisant les ressources de l'ordinateur
* **pipeline-genai.ipynb** : Notebook complet regroupant l'intégralité de la pipeline. Il permet d'inférer sur les différents modèles de diffusion et de réaliser les comparaisons qualitatives et quantitatives. Ce notebook est optimisé pour être exécuté dans un environnement **Kaggle** (nécessite un GPU).
* **test_lissage_tiles.ipynb** : Notebook dédié à l'expérimentation sur l'optimisation de la reconstruction. Il implémente le lissage "intelligent" des vues intermédiaires via **Real-ESRGAN** afin d'améliorer la topologie finale du maillage. Ce notebook peut être facilement lancé sur google colab.

## Architecture technique

Le framework InstantMesh repose sur une architecture en deux étapes :

1. **Diffusion Multi-vues** : Génération de 6 vues cohérentes à partir de l'image source.
2. **Large Reconstruction Model (LRM)** : Transformation de ces vues en une représentation 3D triplane, puis extraction du maillage via FlexiCubes.

## Optimisations et Résultats

* **Gestion des dépendances** : Résolution des conflits liés à la librairie `taming-transformers` pour permettre un déploiement stable en environnement Cloud.
* **Comparaison de modèles** : Évaluation de Zero123++ (fine-tuné et base) et SyncDreamer sur les datasets Objaverse et Google Scanned Objects.
* **Amélioration Real-ESRGAN** : L'intégration d'un GAN pour traiter les images intermédiaires à une résolution de 512px a permis d'atteindre une fidélité structurelle supérieure à 98% (Score de Chamfer moyen de 0.019).

## Références

* Xu et al. (2024) - InstantMesh
* Shi et al. (2023) - Zero123++
* Wang et al. (2021) - Real-ESRGAN
