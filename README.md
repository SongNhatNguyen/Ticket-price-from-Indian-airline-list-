# Flight Price Analysis and Prediction in India

This repository contains a technical project focused on analyzing and predicting airfares for major airlines in India. The study utilizes data from the Ease My Trip website to identify pricing patterns and build predictive models using machine learning.

## Project Overview
The project is structured into four main phases:
1. Data Understanding: Initial exploration of the flight dataset.
2. Data Processing: Cleaning and preparing data for analysis.
3. Data Analysis (EDA): Visualizing how factors like booking time, class, and flight duration affect prices.
4. Machine Learning: Developing and evaluating regression models to predict ticket costs.

## Key Insights from Analysis

### Booking and Timing
- Prices increase significantly when booked within 14 days of departure. For the best rates, Economy tickets should be purchased at least 20 days in advance.
- Late-night flights (both departure and arrival) are consistently the cheapest options.
- Evening departures and afternoon arrivals tend to be the most expensive time slots.

### Airlines and Service Class
- Vistara is the most prominent airline in the dataset, leading in both Economy and Business class flight counts.
- Only Vistara and Air India offer Business class services.
- While Business class is generally much more expensive, there is occasional price overlap between premium Economy and Business fares.

### Flight Characteristics
- Non-stop flights are the most affordable.
- Flights with one stop during the journey usually have the highest prices.
- There is a clear positive correlation between flight duration and ticket price.

## Technical Implementation
- Language: Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- Pre-processing: Categorical features are handled via Label Encoding, and numerical data is normalized using MinMaxScaler.

## Machine Learning Model Comparison
Three regression models were evaluated based on their R2 score and error rates:

1. Random Forest Regressor: The best performing model with an R2 score of approximately 0.98. It shows the strongest linear relationship between actual and predicted prices with minimal outliers.
2. Decision Tree Regressor: Achieved high accuracy (~0.97) but displayed more outliers in prediction plots compared to Random Forest.
3. Linear Regression: Found to be unsuitable for this dataset due to the high volume of outliers and the non-linear complexity of flight pricing.

## Repository Structure
- archive/: Contains the Clean_Dataset.csv file.
- flight_analysis.py: The complete Python source code for analysis and modeling.
- Technical_Report.pdf: Full documentation of the study.
- README.md: Project documentation.

## Installation and Usage
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed:
   pip install pandas numpy matplotlib seaborn scikit-learn
3. Run the analysis script:
   python flight_analysis.py

## Contributors
- Nguyen Le Cong Duy (20110161)
- Nguyen Thanh Truc (20110342)
- Nguyen Song Nhat (19110139)

Instructor: Nguyen Tien Dat
Class: 20TTH - Ho Chi Minh City University of Science (HCMUS)
