import streamlit as st
import numpy as np

st.set_page_config(page_title="ML Project Report", layout="wide")

# ======================
# TITLE
# ======================
st.title("ML-Project")

# ======================
# INTRODUCTION / BACKGROUND
# ======================
st.header("Introduction/Background")
st.markdown("""
Capturing information about the world is the ultimate goal of any data prediction model- first observe current conditions, then develop reasoning about the world. Data collection comes in all forms, from text input to video recordings. To capture the world in 3D as it actually exists is a challenge on its own, with 2D media not being sufficient to actually model real environments. Hence, LiDAR sensors have been the primary technology used to collect dimensional data in order to simulate the real world environment because they capture and encode distance from source to surroundings objects, beating out the compressed data that traditional 2D videos have traditionally captured.
""")

# ======================
# PROBLEM DEFINITION
# ======================
st.header("Problem Definition")
st.markdown("""
LiDAR is at the forefront of technology used to digitally navigate the real world. Applications range from simple AR on mobile devices to physically navigating autonomous vehicles quickly and safely [1]. Further applications of point clouds can assist retaining physical information that could be lost in the scanning process [2].  However, the data amassed by these sensors necessitate additional methods to actually make use of it [3]. Techniques utilizing segmentation can help with optimizing classification and identifying local point cloud structures [4]. We seek to utilize point clouds to normalize data and equalize sensors’ perceptions of the world to build an accurate mental model in the computational system. We also seek to quickly identify and classify objects using data about the 3D world. To verify our solutions to these problems we will track quantitative metrics that give us insight into model accuracy and speed.
""")

# ======================
# METHODS OVERVIEW
# ======================
st.header("Methods Overview")
st.markdown("""
Data Preprocessing:
- Point Cloud Normalization 
  - pc_normalize()
  - It is from the PointNet++ repo
  - Ensures consistent scale and position across samples, improving model generalization
     
Machine Learning Models/Algorithms:
- PointNet++
  - Class: PointNet2ClassificationSSG
  - PointNetPlusClassifier() in PyTorch/ Matlab
- PointMLP
- DGCNN
""")

# ======================
# DATA PREPROCESSING METHOD
# ======================
st.header("1 Data Preprocessing Method")
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
pc_normalize(pc) normalizes the point clouds by centering it at the origin and scaling it so that all points fit inside a unit sphere. It does this by subtracting the centroid and dividing by the maximum distance from the origin. This makes the point clouds smaller and uniformly scaled around the center. We chose this because it was recommended by the authors who created the PointNet++ model.
""")

# ======================
# CHOSEN MODELS
# ======================
st.header("3 Chosen Models")

st.markdown("""
For our first model, we chose the open source PointNet++ model (repo attached within our directories) on our LiDar point cloud data. It is a supervised model that is widely popular for training on LiDar data for classification purposes. It also works best with large, diverse data sets, which is the case with the data of our choice. Our goal for this project was to capture local features and relationships in LiDAR point clouds (semantic segmentation), a key feature of PointNet++ among it's other strengths.
""")

st.markdown("""
For our second model, we implemented DGCNN (Dynamic Graph CNN), another supervised model commonly used in point cloud understanding. DGCNN builds a dynamic k-nearest-neighbor graph at each layer, allowing it to capture fine-grained geometric relationships between points. This makes it particularly effective for segmentation tasks where local neighborhood structure is important. By continuously updating the graph as features evolve, DGCNN adapts to complex shapes in LiDAR data and can learn more expressive local descriptors than static graph approaches.
""")

st.markdown("""
For our third model, we selected PointMLP, a supervised MLP-based model designed to avoid heavy reliance on convolutions or graph operations. PointMLP focuses on efficient per-point feature extraction and uses residual MLP blocks with geometric affine transformations to improve robustness to variations in point distributions. It is known for its strong performance on large-scale point cloud datasets and its relatively simple, fast inference. For our project, PointMLP serves as a comparison to graph-based and hierarchical models, letting us evaluate how an MLP-driven method handles semantic segmentation of LiDAR data.
""")

# ======================
# RESULTS AND DISCUSSION
# ======================
st.header("Results and Discussion")
st.markdown("""
Quantitative Metrics: 
- F1 Score
  - Indicates how well points are classified and sorted into the correct classes.
- Accuracy
  - Tells us how well the models has learned local structures in relation to global structure.
- Precision
  - How fine the local geometry is captures leading to correct class predictions.
""")

st.markdown("""
Why our models performed well and what are the next steps we plan to take?
""")

st.markdown("""
The results across classification, part segmentation, and semantic segmentation show that PointNet++ consistently outperforms PointNet. When classifying on the ModelNet dataset, PointNet++, especially the MSG model, achieves the highest accuracy (92.8%), which shows that learning local structure leads to better overall shape recognition. For part segmentation, PointNet++ has higher instance and class IoU, demonstrating higher ability to capture small details that separate object parts. On S3DIS semantic segmentation, PointNet++ raises class-average IoU from 43.7% to 53.5%, showing more consistent performance across different categories. Overall, the results show that learning local features at multiple scales leads to more accurate and reliable 3D understanding than the original PointNet.

""")

st.image("PointNet_Accuracy_Over_Training_Epochs.png", caption="Accuracy over Epochs")

st.markdown("""
All in all, the PointNet++ model is **solid but not stellar**. We are currently seeing ~0.99 train instance accuracy against the ~0.92 test instance accuracy (best ~0.929) with class accuracy ~0.90, which points to decent learning plus a modest generalization gap and likely class imbalance (where: instance > class). It is perfecetly usable; however, some classes are clearly underperfoming and later epochs do not improve on the earlier best, suggesting we are overfitting or under-augmenting for our lighter point clouds. (i.e., sparser samples with fewer points per object and more local noise, so the train distribution is denser than what we evaluate on).
""")

st.markdown("""
The PointNet++ model shows solid performance across 155 training epochs, with the macro F1 score improving from 0.8335 initially to 0.8910 at the end, peaking at 0.8940 around epoch 129. The weighted F1 score follows a similar pattern, starting at 0.8750 and reaching 0.9212 by the final epoch, with a best result of 0.9259 at epoch 126. The gap between macro and weighted scores (roughly 3-4 percentage points) suggests the model performs better on more common classes in the dataset. Precision and recall remain well-balanced throughout training, with both metrics hovering around 0.88-0.89 for macro averages and 0.92-0.93 for weighted averages in later epochs, indicating the model doesn't heavily favor one over the other. Test accuracy stabilizes around 0.92, aligning well with the F1 scores. Performance plateaus around epoch 125 with minimal gains afterward, and occasional dips like those seen around epochs 78-80 are quickly recovered, suggesting healthy training without significant overfitting. Overall, the convergence pattern and balanced metrics indicate the model has likely reached its performance limit for this architecture and dataset.
""")

st.markdown("""
PointMLP
The PointMLP model maintains consistently strong performance over its training period, with macro F1 scores starting around 0.860 and improving steadily to roughly 0.900 by the final epochs. Weighted F1 mirrors this pattern, increasing from the upper 0.88 range to about 0.93, indicating that the model performs well not only on common classes but also handles minority categories more effectively than PointNet++. Precision and recall remain balanced, both stabilizing in the 0.89–0.91 macro range and 0.93–0.94 weighted range by the end of training. Test accuracy hovers around 0.93, showing minimal divergence from the weighted F1, which suggests strong consistency between training and evaluation. Unlike PointNet++, the performance plateau arrives slightly earlier, around epoch 120, with less variance between checkpoints. The smoother convergence curve implies that PointMLP is better regularized and benefits from its MLP-based feature extraction, which likely captures more stable global representations of the point clouds.

DGCNN
The DGCNN model delivers competitive results overall, with macro F1 gradually improving from around 0.84 to the high 0.88s and weighted F1 reaching just above 0.91 by the final epochs. The smaller gap between macro and weighted scores (around 2–3 percentage points) indicates the model generalizes more uniformly across classes compared to PointNet++. Precision and recall remain well-aligned throughout training, both increasing in parallel and stabilizing around 0.88–0.89 macro and 0.91–0.92 weighted. Test accuracy settles close to 0.92, consistent with the F1 trends. Training curves show that DGCNN converges somewhat slower than PointMLP, likely due to its reliance on dynamic graph updates, but it compensates with better robustness to local geometric noise. The results suggest that DGCNN’s edge-based feature learning helps preserve fine-grained structural relationships between neighboring points, giving it a small advantage on classes that depend on local spatial patterns rather than global shape cues.


""")

st.markdown("""
We compared three point cloud classifiers: PointNet++, PointMLP, and DGCNN. Our main metrics are overall accuracy, macro precision, and macro F1. Because the dataset is class imbalanced, macro precision and macro F1 matter a lot, since they weight each class equally and show whether a model is doing well on rare classes and not just the common ones.

PointNet++
For our run of PointNet++ with semi-supervised fine tuning, test accuracy quickly gets into the low 90% range. Train accuracy is a bit higher, which gives us a modest generalization gap, but nothing extreme. Macro precision and macro F1 are a few points below the overall accuracy. This tells us that PointNet++ is handling the frequent classes fairly well, but performance on rarer or harder classes still lags behind. In other words, the model is solid, but there is room to improve how evenly it behaves across all categories.

PointMLP.
PointMLP ends up with the best overall numbers among the three models. Its test accuracy is slightly higher than PointNet++ and its macro precision and macro F1 are also higher. The gap between train and test accuracy is smaller, which suggests that our regularization and augmentation are working reasonably well. The smaller gap between accuracy and macro F1 indicates that PointMLP spreads its performance more fairly across classes and is less biased toward the majority ones.

DGCNN.
DGCNN is architecturally different from the other two. It builds a dynamic k-NN graph and uses edge convolutions on local neighborhoods, so it focuses more on local geometric relationships between points. In practice, DGCNN tends to land in a similar accuracy range as the other models, but it can help on classes where fine-grained part relationships are important. Even if its raw accuracy is similar, it gives us a different inductive bias compared to PointNet++ and PointMLP. That makes it useful as a complementary model rather than a strict replacement.

Cross model comparison.
Putting everything together, all three models reach high overall accuracy, but they differ in how they treat minority classes. PointNet++ with SSL is a strong baseline and benefits from pretraining, but its macro precision and macro F1 lag its accuracy, which means some classes are still underperforming. PointMLP gives us the best balance of the three: highest accuracy and noticeably stronger macro precision and macro F1, so it is the best single model if we care about both overall performance and fairness across classes. DGCNN adds diversity by focusing on local edges and is especially helpful for shapes where relative part positions matter more than global shape alone.

Next steps:

Across all models, macro F1 is still several points below overall accuracy, so the main next step is to improve performance on minority and “difficult” classes rather than just chasing a small bump in accuracy. Concretely, we would like to

use class-weighted cross entropy or focal loss, and possibly oversample rare classes, to directly target macro precision and macro F1,

strengthen geometric data augmentation (rotations, scaling, jitter, point dropout) so train and test distributions match better,

explore simple ensembles or knowledge distillation that combine PointMLP with PointNet++ and DGCNN to leverage their different strengths, and

run a more detailed per class confusion matrix analysis to see exactly which categories are driving the gaps in precision and F1.

Overall, our results show that all three models are viable for 3D shape classification on this dataset, with PointMLP as the strongest single choice and PointNet++ and DGCNN providing useful complementary behavior for future improvements.


""")

# ======================
# GANTT CHART
# ======================
st.header("Gantt Chart")
st.image("gantt.jpeg", caption="Gantt Chart")

# ======================
# CONTRIBUTION TABLE
# ======================
st.header("Contribution Table")
st.image("contribution.jpg", caption="Contribution Table")

# ======================
# PROJECT GOALS
# ======================
st.header("Project Goals")
st.markdown("""
- Latency: Minimize time taken to classify an object
  - Preprocessing techniques can help minimize latency ensuring objects get recognized within a few moments of being seen
- Accuracy: Successfully achieve high classification
  - The objects should be classified as the object that it is, with low rates of error
""")

# ======================
# EXPECTED RESULTS
# ======================
st.header("Expected Results")
st.markdown("""
- A working object classification system that can classify objects with high accuracy
- Measure improvements in performance from data preprocessing
""")

# ======================
# REFERENCES
# ======================
st.header("References")
st.markdown("""
[1] C. Vishnu, J. Khandelwal, C. K. Mohan and C. L. Reddy, “EVAA–Exchange Vanishing Adversarial Attack on LiDAR Point Clouds in Autonomous Vehicles,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023, Art. no. 5703410, doi: 10.1109/TGRS.2023.3292372.

[2] J. Park and Y. K. Cho, “Point Cloud Information Modeling: Deep Learning–Based Automated Information Modeling Framework for Point Cloud Data,” J. Construct. Eng. Manage., vol. 148, no. 2, Art. no. 04021191, 2022, doi: 10.1061/(ASCE)CO.1943-7862.0002227.

[3] M. Hao et al., “Coarse to fine-based image–point cloud fusion network for 3D object detection,” Information Fusion, vol. 112, p. 102551, Dec. 2024. doi:10.1016/j.inffus.2024.102551 

[4] H. Ni, X. Lin and J. Zhang, “Classification of ALS Point Cloud with Improved Point Cloud Segmentation and Random Forests,” Remote Sens. (Basel, Switzerland), vol. 9, no. 3, Art. no. 288, 2017, doi: 10.3390/rs9030288.
""")

# ======================
# REPO STRUCTURE
# ======================
st.header("Repo Structure")
st.markdown("""
`/classification_ModelNet40/`: Training and evaluation pipeline for ModelNet40 point-cloud classification.

`/classification_ModelNet40/models/`: Contains model definitions.

`/classification_ModelNet40/models/__init__.py`: Initializes model imports.

`/classification_ModelNet40/models/pointmlp.py`: Full PointMLP architecture (sampling, KNN grouping, residual MLP blocks, classifier).

`/classification_ModelNet40/utils/`: Utility scripts for training/testing.

`/classification_ModelNet40/utils/progress.py`: Progress bar utilities.

`/classification_ModelNet40/utils/logger.py`: Logs accuracy, loss, and LR.

`/classification_ModelNet40/utils/misc.py`: RNG seeding, accuracy functions, timers.

`/classification_ModelNet40/utils/__init__.py`: Utility initialization.

`/classification_ModelNet40/data.py`: Loads and processes ModelNet40.

`/classification_ModelNet40/helper.py`: Loss functions & evaluation utilities.

`/classification_ModelNet40/main.py`: Main training script.

`/classification_ModelNet40/test.py`: Evaluation script for trained models.

`/classification_ModelNet40/voting.py`: Multi-view voting (random rotations/scaling).

`/classification_ModelNet40/Log.txt`: Summarized training log.

`/classification_ModelNet40/out.txt`: Full terminal output during training.

`/classification_ScanObjectNN/`: Classification pipeline for the ScanObjectNN dataset.

`/`: Images used for documentation, model visualization, and README assets.

`/part_segmentation/`: Point-cloud part segmentation using ShapeNetPart.

`/pointnet2_ops_lib/`: CUDA extension library for KNN, FPS, and grouping operations.

`/pointnet2_ops_lib/pointnet2_ops/`: CUDA kernels.

`/pointnet2_ops_lib/setup.py`: Build script for CUDA ops.

`/analysis.py`: Script for visualization, debugging, and experimental analysis.

`/environment.yml`: Default environment file provided by authors.

`/environment_Dennis.yml`: Custom PACE ICE environment (stabler dependencies).

`/Overview.pdf`: PDF explaining the PointMLP architecture.

`/Overview.png`: Image version of the model overview.

`/requirements.txt`: Python dependencies needed to run the project.

`/README.me`: Original documentation for running models.

`/LICENSE`: Open-source license for this repository.

`/.gitignore`: Prevents large files (logs, checkpoints) from being committed.
""")
