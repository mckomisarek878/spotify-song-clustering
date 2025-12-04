# Spotify Song Clustering with KMeans

This project applies **unsupervised machine learning** to cluster over
2,000 Spotify songs using **KMeans** and their associated audio
features.\
The goal is to discover *feature‑based musical groupings* that go beyond
traditional genre labels.

------------------------------------------------------------------------

##  Project Overview

Music does not fall into discrete categories, and genre labels are often
vague or inconsistent.\
Using clustering, we attempt to learn **data‑driven song groups** based
solely on numerical audio features such as:

-   Danceability\
-   Energy\
-   Acousticness\
-   Valence\
-   Tempo\
-   Instrumentalness\
-   Speechiness\
-   Key & Mode

These clusters represent "hidden genres" or song families based on how
tracks *sound*, not how they are labeled.

------------------------------------------------------------------------

##  Methods

### **Data Preprocessing**

-   Loaded 2,017 songs from a Kaggle dataset\
-   Removed non‑learnable or irrelevant columns\
-   Standardized all numerical features using `StandardScaler`

### **PCA Exploration**

-   Tested 1--16 principal components\
-   Found that variance was evenly distributed\
-   Determined PCA would not meaningfully improve clustering

### **Clustering with KMeans**

-   Evaluated `k = 2` to `k = 15`\
-   Used **SSE (Elbow Method)** and **Silhouette Scores**\
-   Selected **k = 6** as the optimal number of clusters\
-   Trained final KMeans model on standardized data

------------------------------------------------------------------------

##  Cluster Interpretations

The centroids revealed six musically meaningful groups:

1.  **Acoustic / Chill Ballads**\
2.  **Upbeat Dance Pop (Minor Key)**\
3.  **Ambient / Instrumental**\
4.  **Bright Major‑Key Pop**\
5.  **Rap / Spoken‑Word Energy**\
6.  **Live / Raw Performances (Low Energy)**

These names are based on feature patterns such as acousticness,
instrumentalness, speechiness, energy, and mode.

------------------------------------------------------------------------

##  Key Learnings

-   Clusters overlapped significantly, which is *expected* for
    continuous musical features\
-   PCA did **not** improve results\
-   Standardization was essential\
-   Silhouette scoring helped guide appropriate *k*\
-   Even with weak separation, clusters were **interpretable and
    musically coherent**

------------------------------------------------------------------------

##  Results & Insights

-   Model successfully grouped similar artists (e.g., Future, Travis
    Scott)\
-   Also revealed **cross‑genre similarities** humans might miss\
-   Demonstrated potential for playlist generation & content‑based
    recommendation

------------------------------------------------------------------------

##  Future Improvements

-   Incorporate **audio embeddings** (OpenL3, CLMR, YAMNet)\
-   Add **lyrics embeddings** for semantic similarity\
-   Apply **Gaussian Mixture Models** or **HDBSCAN** for softer cluster
    boundaries\
-   Explore **genre prediction** using these learned clusters

------------------------------------------------------------------------

##  Files Include

-   Notebook with full analysis\
-   Clustering results\
-   Labeled dataset with assigned cluster names\
-   Visualizations of centroid features

------------------------------------------------------------------------

##  Acknowledgements

Dataset from Kaggle:\
https://www.kaggle.com/datasets/geomack/spotifyclassification
