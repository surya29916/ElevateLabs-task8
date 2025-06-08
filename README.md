# ElevateLabs_Task8
# 🛍️ Mall Customer Segmentation using K-Means Clustering

## ✅ Steps Performed

### 1. Load and Visualize Dataset
- Loaded the dataset using `pandas`.
- Selected relevant features: `Annual Income` and `Spending Score`.

### 2. Preprocessing
- Scaled the numerical features using `StandardScaler` for better clustering performance.

### 3. Elbow Method for Optimal Clusters
- Applied K-Means clustering for `K = 1 to 10`.
- Plotted the **Elbow Curve** to find the optimal number of clusters.
- The "elbow point" suggests the optimal `K`.

### 4. Apply K-Means and Assign Clusters
- Applied `KMeans` with the chosen `K` (e.g., 5).
- Assigned each customer to a cluster.
- Added the cluster label as a new column.

### 5. Visualize the Clusters
- Used a scatter plot to visualize customer groups in 2D space.
- Highlighted centroids with yellow markers.

### 6. Evaluate Using Silhouette Score
- Calculated the **Silhouette Score** to measure the cohesion and separation of clusters.

---

## 🧠 Interpretation

- **Elbow Method** helps decide how many clusters (K) to use.
- **Silhouette Score** tells how well-separated and cohesive the clusters are.
- By clustering based on income and spending score, we can:
  - Identify high-income low-spending vs low-income high-spending customers.
  - Tailor marketing strategies for each group.

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---
