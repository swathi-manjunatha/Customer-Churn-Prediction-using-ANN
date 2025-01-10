# **Customer Churn Prediction Using Deep Learning**

## **Overview**
This project involves building a deep learning model to predict **customer churn** using historical data. The model is designed to help businesses identify at-risk customers and implement strategies to retain them. The project demonstrates data preprocessing, artificial neural network (ANN) modeling, and performance evaluation using industry-standard metrics.

---

## **Technologies and Tools**
- **Programming Language:** Python  
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Model Architecture:** Artificial Neural Network (ANN)

---

## **Dataset**
The dataset contains customer data, including demographic information, account tenure, monthly charges, total charges, and churn status. The data is structured as follows:
- **Total Records:** 5,625  
- **Features:** 26 (e.g., tenure, MonthlyCharges, TotalCharges)  
- **Target Variable:** Churn (0 = No Churn, 1 = Churn)

### **Class Distribution**
- **No Churn (0):** 4,164 (74%)  
- **Churn (1):** 1,461 (26%)  

---

## **Project Workflow**

### **1. Data Preprocessing**
- Encoded categorical variables using **one-hot encoding** to convert them into numerical representations.  
- Scaled numerical features (**tenure, MonthlyCharges, TotalCharges**) using **MinMaxScaler** for consistent data ranges.  
- Split the dataset into **training (80%)** and **testing (20%)** sets.  

### **2. Model Development**
- **Architecture:**
  - Input Layer: 26 neurons (matching the number of input features).  
  - Hidden Layers: Two layers with 26 and 15 neurons, each using **ReLU activation**.  
  - Output Layer: 1 neuron with **sigmoid activation** for binary classification.  

- **Compilation and Optimization:**
  - **Loss Function:** Binary Cross-Entropy for binary classification tasks.  
  - **Optimizer:** Adam optimizer for adaptive learning rates.  

- **Training:**
  - Trained for 100 epochs with a batch size of 32.  

### **3. Evaluation**
- Evaluated the model using **accuracy**, **precision**, **recall**, and **F1-score**.  
- Visualized the results using a **confusion matrix** and **Seaborn heatmaps**.

---

## **Results**
- **Accuracy:** 83.5%  
- **F1-Score:** 0.72  
- **Precision:** 0.70  
- **Recall:** 0.75  

### **Insights:**
1. **Churn Trends:**
   - Customers with **higher monthly charges** are more likely to churn.  
   - Customers with **longer tenure** are less likely to churn.

2. **Confusion Matrix:**
   - Model effectively predicts the majority class (No Churn) but has room for improvement in identifying churned customers.

---

## **Limitations**
1. **Class Imbalance:** The dataset is moderately imbalanced, which might affect the model's ability to predict churned customers.  
2. **Generalization:** Regularization techniques like dropout layers were not applied, which could reduce overfitting.  
3. **Additional Evaluation Metrics:** ROC-AUC curves could provide a more comprehensive evaluation of the model.

---

## **Future Improvements**
1. **Address Class Imbalance:**
   - Implement **SMOTE (Synthetic Minority Over-sampling Technique)** or class weighting to balance the dataset.  

2. **Enhance Model Generalization:**
   - Add **dropout layers** and perform hyperparameter tuning.  

3. **Model Comparisons:**
   - Benchmark results against simpler models like **Logistic Regression** and ensemble models like **Random Forest** or **Gradient Boosting**.

4. **Deploy the Model:**
   - Use **Flask** or **FastAPI** to deploy the model as a web application for real-time predictions.

---

## **How to Run**
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script:  
   ```bash
   jupyter notebook
   # or
   python churn_prediction.py
   ```

---

## **Files in the Repository**
1. **churn_prediction.py:** Python script for data preprocessing, model building, and evaluation.  
2. **churn_data.csv:** Dataset used for training and testing the model.  
3. **README.md:** Detailed documentation of the project.

---
