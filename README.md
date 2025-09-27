# Machine-Learning--Health-Analysis

Health Analysis: Prostate Cancer Detection Using Machine Learning


# Overview
This project implements a machine learning system to aid in the early prediction and diagnostic assistance of prostate cancer. Developed as a solution to address the high incidence rates, particularly in regions like the Caribbean and Jamaica, the system uses clinical and physical features to assess a patient's susceptibility.

The primary goal is to provide a standardized, high-quality diagnostic aid by leveraging predictive analytics to minimize false positives and negatives in clinical decision-making.

# 🚀 Key Features and Functionality
- Dual Classification Models: Utilizes two machine learning algorithms for robust prediction:

    - Logistic Regression: Trained for general classification and evaluated for performance using an R2 score and Confusion Matrix.

     - Decision Tree Classifier: Used for the final user prediction due to its interpretability. The model is also visualized in a .dot file.

- User Interaction: The main script prompts the user with a series of questions (e.g., age, PSA levels, family history, and physical symptoms) to gather necessary input.

- Risk Assessment: Provides an immediate prediction and an estimated probability of prostate cancer based on user responses, along with recommended next steps.

- Data Preprocessing: Handles data cleaning, filtering of NaN values, and mapping categorical diagnosis labels ('M'/'B') to binary numeric values.

💻 Installation and Setup
Prerequisites
Python: Version 3.8 or newer

Pip: Python package installer

Step 1: Clone the Repository (or download files)
Ensure the following files are in the same directory:

prostate_detection.py (or the name of your main Python script, e.g., Prostate_Dectection_Original.py)

Prostate_Cancer.csv (Your dataset)

Step 2: Install Dependencies
Open your terminal or command prompt and run the following command to install all necessary libraries:

pip install pandas scikit-learn numpy matplotlib seaborn

⚙️ How to Run the Program
1. Navigate to the Directory:
    cd /path/to/your/project

2. Execute the Script:
   Run the main Python script from your terminal:
    python prostate_detection.py
(Replace prostate_detection.py with the actual name of your script).

3. Answer Prompts: The program will guide you through a series of questions. Provide responses as prompted (e.g., enter numerical values for age/PSA and yes or no for binary questions).

📄 Model Documentation and Output

The script generates the following:
- Model Performance: Outputs the Confusion Matrix and R2 score for the Logistic Regression model to the console.
- Decision Tree Visualization: Exports the trained Decision Tree structure to a file named prostate.dot, which can be visualized using tools like Graphviz.
- Final Prediction: Prints the predicted risk level and probability based on the user's input.

🧑‍💻 Team Members
This project was developed by the following students for the Machine Learning [CIT4038] module at the University of Technology (UTECH), Jamaica:

1. Jeamelee Smith
2. Stephan McLean
3. Kayanna Bonner
4. Quinsbert Tyndall
