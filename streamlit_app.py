import streamlit as st
import numpy as np

st.set_page_config(page_title="ML Project Report", layout="wide")

# ======================
# TITLE
# ======================
st.title("📘 ML Project Report: LiDAR Point Cloud Classification")
st.markdown("""
Capturing information about the world is the ultimate goal of any data prediction model.
Data collection comes in all forms, from text input to video recordings.  
LiDAR sensors are widely used to capture the world in 3D accurately.
""")

# ======================
# INTRODUCTION / BACKGROUND
# ======================
st.header("Introduction / Background")
st.markdown("""
Data collection comes in all forms, from text input to video recordings. To capture the world in 3D as it actually exists is a challenge on its own, with 2D media not being sufficient to actually model real environments. LiDAR sensors capture and encode distance from source to surrounding objects, outperforming compressed 2D video data.
""")

# ======================
# PROBLEM DEFINITION
# ======================
st.header("Problem Definition")
st.markdown("""
LiDAR is at the forefront of technology for digital navigation. Applications include AR on mobile devices and autonomous vehicle navigation [1].  

Segmentation techniques optimize classification and identify local point cloud structures [4]. Our goal is to normalize point cloud data and equalize sensors’ perception to build accurate computational models. We also aim to classify objects quickly using 3D data.  
Quantitative metrics such as **accuracy**, **F1 score**, and **precision** will verify model performance.
""")

# ======================
# METHODS
# ======================
st.header("Methods Overview")

st.subheader("Data Preprocessing")
st.markdown("""
- **Point Cloud Normalization**
  - `pc_normalize()` (from PointNet++ repo)
  - Ensures consistent scale and position across samples, improving model generalization
""")

st.subheader("Machine Learning Models / Algorithms")
st.markdown("""
- **PointNet++** (`PointNet2ClassificationSSG`)  
- **PointMLP**  
- **DGCNN**
""")

# ======================
# PC_NORMALIZE FUNCTION
# ======================
st.subheader("Data Preprocessing Method: pc_normalize")
st.code("""
def pc_normalize(pc): 
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
""", language="python")

st.markdown("""
`pc_normalize(pc)` centers the point cloud at the origin and scales points to fit inside a unit sphere. This improves generalization and ensures uniform scaling.
""")

# ======================
# CHOSEN MODELS
# ======================
st.header("Chosen Models")

st.subheader("PointNet++")
st.markdown("""
- Hierarchical feature learning from point clouds  
- Works best with large, diverse datasets  
- Focused on capturing local features and semantic segmentation
""")

st.subheader("DGCNN")
st.markdown("""
- Builds dynamic k-nearest-neighbor graph at each layer  
- Captures fine-grained geometric relationships  
- Effective for segmentation tasks with complex local structures
""")

st.subheader("PointMLP")
st.markdown("""
- MLP-based per-point feature extraction  
- Uses residual MLP blocks and geometric affine transformations  
- Fast inference with strong performance on large point cloud datasets
""")

# ======================
# RESULTS AND DISCUSSION
# ======================
st.header("Results and Discussion")
st.markdown("""
**Quantitative Metrics:**  
- **F1 Score**: classification performance  
- **Accuracy**: global and local structure learning  
- **Precision**: quality of local geometry capture

**Performance:**  
- PointNet++ consistently outperforms PointNet for classification, part segmentation, and semantic segmentation.
- Example: ModelNet dataset → PointNet++ MSG accuracy: 92.8%
- Overfitting observed in lighter/sparser point clouds (train ~0.99 vs test ~0.92)
""")

st.image("images/PointNet_Accuracy_Over_Training_Epochs.png", caption="Accuracy over Epochs")

st.markdown("""
**Next Steps:**  
- Strengthen augmentation (rotation, scaling, jitter, dropout)  
- Loss rebalancing with class weighting / focal loss  
- Regularization and optimized training schedule  
- Diagnostics with per-class confusion matrix  
- Rebalance dataset for rare classes
""")

# ======================
# GANTT CHART
# ======================
st.header("Gantt Chart")
st.image("images/gantt.jpeg", caption="Project Gantt Chart")

# ======================
# CONTRIBUTION TABLE
# ======================
st.header("Contribution Table")
st.image("images/contribution.jpg", caption="Team Contributions")

# ======================
# PROJECT GOALS
# ======================
st.header("Project Goals")
st.markdown("""
- **Latency**: minimize time to classify objects  
- **Accuracy**: achieve high classification performance
""")

# ======================
# EXPECTED RESULTS
# ======================
st.header("Expected Results")
st.markdown("""
- Working object classification system with high accuracy  
- Measure improvements from preprocessing techniques
""")

# ======================
# REFERENCES
# ======================
st.header("References")
st.markdown("""
[1] C. Vishnu et al., EVAA–Exchange Vanishing Adversarial Attack on LiDAR, IEEE Trans. Geosci. Remote Sens., 2023  
[2] J. Park & Y. K. Cho, Point Cloud Information Modeling, J. Construct. Eng. Manage., 2022  
[3] M. Hao et al., Coarse to fine-based image–point cloud fusion network, Information Fusion, 2024  
[4] H. Ni et al., Classification of ALS Point Cloud, Remote Sens., 2017
""")

# ======================
# END
# ======================
st.markdown("---")
st.markdown("📌 **Repo Structure:** See original README for detailed file breakdown")
