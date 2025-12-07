# 🚀 Projet Numba – Calcul Parallèle & Optimisation Automatique

Ce projet présente une pipeline **complète, automatisée et reproductible** permettant :

- d’identifier automatiquement les *hotspots* dans un fichier Python,
- d’analyser le code (AST + complexité Radon),
- de profiler les performances en Python pur,
- de générer automatiquement une version optimisée avec **Numba**,
- de produire automatiquement les tests de parité,
- de générer les benchmarks reproductibles,
- d’exécuter ces tests et benchmarks directement dans une **application Streamlit**.

---

## 🎯 1. Objectifs du projet

L’objectif principal est de proposer un système capable d’optimiser automatiquement des fonctions Python coûteuses, en appliquant une démarche rigoureuse :

1. **Analyse statique** : AST, boucles, structure interne, complexité.  
2. **Profiling dynamique** : temps d'exécution réel avec `cProfile`.  
3. **Optimisation automatique** : génération d’une nouvelle version avec `@njit`.  
4. **Vérification automatique** : tests Python vs Numba.  
5. **Benchmarks reproductibles** : calcul du speedup.  
6. **Interface Streamlit** : visualisation + exécution directe.

---

---

## 🔍 3. Analyse statique du code (AST + Radon)

Le système :

- parcourt le code avec `ast`,
- détecte les fonctions contenant des boucles `for`,
- identifie automatiquement les hotspots,
- mesure la complexité cyclomatique avec **Radon**,
- affiche les résultats directement dans Streamlit.

Cette étape permet de cibler **automatiquement** les fonctions optimisables.

---

## ⚡ 4. Optimisation automatique avec Numba

Le cœur du projet repose sur la génération automatique de fichiers :

monfichier_numba_auto.py

Chaque version optimisée contient :

- un import `from numba import njit`,
- des décorateurs `@njit` ajoutés automatiquement,
- le reste du code original parfaitement conservé.

Résultats observés : **accélération ×50 à ×350** selon les fonctions.

---

## 🧪 5. Tests automatiques

Un fichier `*_auto_test.py` est généré automatiquement.  
Il :

- charge dynamiquement les modules Python et Numba,
- génère automatiquement des entrées adaptées (`generate_inputs_for`),
- compare Python vs Numba via `np.allclose`.

🎯 Objectif : garantir la **correction fonctionnelle** de chaque optimisation.

---

## ⏱️ 6. Benchmarks automatiques

Un fichier `*_auto_bench.py` est également créé :

- warm-up Numba,
- timing Python pur,
- timing Numba compilé,
- affichage du speedup directement en console ou dans Streamlit.

Les tests ET benchmarks peuvent être lancés **depuis Streamlit**, sans ouvrir un terminal.

---

## 🖥️ 7. Application Streamlit

### L’application contient 5 onglets :

#### **1) Sum of squares**
Exemple simple pour illustrer la démarche.

#### **2) Search Sequence**
Comparaison Python / NumPy / Numba / Numba parallèle.

#### **3) Truss 10 barres**
Cas scientifique réel, avec boucles complexes.

#### **4) Auto-optimisation (Cœur du projet)**
Permet :
- de choisir un fichier,
- d’analyser ses hotspots,
- de générer une version optimisée,
- de produire les fichiers tests + benchmarks,
- d’exécuter ces fichiers en un clic.

#### **5) Synthèse globale**
Démarche scientifique + résultats + analyse.

---

## 🤖 8. Rôle des modèles de langage (LLMs)

Les LLMs ont été utilisés pour :

- organiser la structure du projet,
- documenter les différentes étapes,
- structurer les fichiers auto-générés,
- améliorer la clarté du rapport,
- expliquer la logique d’optimisation.

Cette utilisation est totalement transparente et encouragée par l’enseignant.

---

## 🚀 9. Installation & Lancement

### 🔧 Installer les dépendances

```bash
pip install -r requirements.txt
```
### Lancer l'application Streamlit
```bash
streamlit run app.py
```
