 A machine learning project that uses K-Means clustering to group customers based on their income and spending behavior.

# Customer Segmentation

This project uses machine learning to group customers into different segments based on their annual income and spending behavior.

The project uses the **K-Means Clustering** algorithm to identify groups of customers with similar characteristics.

## Project Objectives

* Load and explore the customer dataset
* Prepare the data for clustering
* Select relevant customer features
* Scale the data
* Determine a suitable number of clusters
* Apply K-Means clustering
* Visualize the customer segments
* Understand the different customer groups

## Dataset

The project uses the `Mall_Customers.csv` dataset.

The main features used are:

* **Annual Income**
* **Spending Score**

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm that groups similar data points into clusters.

In this project, customers with similar income and spending behavior are placed into the same cluster.

```text
Customer Data
     ↓
Data Preparation
     ↓
Feature Scaling
     ↓
Find Optimal Number of Clusters
     ↓
K-Means Clustering
     ↓
Customer Segments
     ↓
Visualization
```

## Project Structure

```text
Customer-Segmentation/
│
├── Mall_Customers.csv
├── main.py
└── README.md
```

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the project:

```bash
python main.py
```

## Results

The K-Means model divides the customers into different groups based on their income and spending score.

The results are visualized using a 2D scatter plot, making it easier to understand the different customer segments.

## Learning Outcomes

Through this project, I practiced:

* Working with real-world datasets
* Data preprocessing
* Feature scaling
* Unsupervised machine learning
* K-Means clustering
* Finding the optimal number of clusters
* Data visualization using Matplotlib

## Internship Task

**Level 1 — Task 2: Customer Segmentation**

This project was completed as part of a Machine Learning internship.
