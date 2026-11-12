# Overview
Create a model to predict 12-Month Default for AIFUL loan customers using their application data---Model performance evaluation by AUC---
*12-Month Default: An account is in default status as of month 12 after loan origination. At AIFUL, an account is classified as being in default when no payment has been received for three consecutive months and the contract is terminated.

Learning Data and Test Data
From the start of Selection round Day 1, each team will get the following 4 files from this competition page.

Learning Data Set (train_dataset.csv)
Test Data (test_without_gt.csv)
Submission template (submission_template_for_test.csv)
Data Dictionary (data_dictionary_customers.xlsx)
The teams will create the model by using the learning data set. Then, the team will use the model to predict the Test Data set.

The target variable is 12-Month Default Flag (1 = Default, 0 = Not in Default). Submit the predicted probability of default for each test record using the submission template.

By the final submission deadline, students must select two files for Private score evaluation.
On the Submissions page, choose your final files by checking “Final submission” for each of the two selected entries.
Note: If no files are selected, the file with the highest Public score will be automatically used as the final submission.

Output
The team will be 100% evaluated by private data AUC ranking.
The top 30 teams on the leaderboard will qualify for the Final Round.

Programming Language and Tool
Python

Development Flow
Data preprocessing
Data analysis
Feature engineering
Model Selection
Examples: Logistic Regression / Decision Tree / Random Forest / XGBoost / LightGBM / Neural Networks
Model Evaluation
Hold Out Method
Cross Validation
Model Tuning
Request
After the competition, you'll be required to delete all competition data, including any copies.
Be sure to delete the data and empty the recycle bin.
If a data leak is found, the team responsible will be banned from the competition, and legal action may be taken if necessary.


Data Overview
The data used in this competition consists of the following four files:

Learning Data Set (train_dataset.csv)
Test Data (test_without_gt.csv)
Submission Template (submission_template_for_test.csv)
Data Dictionary (data_dictionary_customers.xlsx)
The dataset contains application information for accounts that applied for and received loans from AIFUL in 2018. The features describe information available at the time of application.

The learning data set contains 68,546 records, including both default and non-default accounts. Each record contains an ID, the target variable, and 25 applicant features.

The test data set contains 137,091 records with the same ID and feature columns. The target variable is withheld.

# Data Columns
The table below summarizes the columns in the learning data set. For detailed definitions and category codes, please refer to the attached Data Dictionary (data_dictionary_customers.xlsx).

No.	Column Name	Description	Notes
1	ID	Unique identifier for each account	
2	12-Month Default Flag	Default status as of month 12 after loan origination	Target variable: 1 = Default, 0 = Not in Default. Not included in the test data.
3	Application Channel	Channel through which the application was submitted	1 = PC, 17 = iPhone, 18 = Android, 20 = Contract Office
4	Gender	Gender of the applicant	1 = Male, 2 = Female
5	Marital Status	Marital status of the applicant	1 = Single, 2 = Married
6	Age (in months)	Age at the time of application	In months
7	Residence Period (in months)	Length of residence at the current address at the time of application	In months
8	Occupation Category	Occupation of the applicant	Refer to the Data Dictionary for category codes
9	Industry Category	Industry of the applicant's employer	Refer to the Data Dictionary for category codes
10	Employment Type Category	Employment type of the applicant	Refer to the Data Dictionary for category codes
11	Insurance Type	Type of health insurance held by the applicant	Refer to the Data Dictionary for category codes
12	Company Size Category	Size category of the applicant's employer	Refer to the Data Dictionary for category codes
13	Business Entity Type	Legal form of the applicant's employer	Refer to the Data Dictionary for category codes
14	Declared Annual Income	Annual income declared by the applicant	
15	Payday	Day of the month on which the applicant receives salary	1–31
16	Declared Number of Borrowings	Number of loans from other lenders declared by the applicant	
17	Declared Borrowing Amount	Total amount borrowed from other lenders declared by the applicant	
18	Desired Credit Limit	Credit limit requested by the applicant	
19	Residence Type	Type of current residence	Refer to the Data Dictionary for category codes
20	Rent Burden	Monthly rent of the current residence	
21	Residence Ownership Status	Ownership status of the current residence	0 = Unknown, 1 = Sole Ownership, 2 = Joint Ownership, 3 = Not Owner
22	Number of Dependents	Number of family members financially supported by the applicant	
23	Number of Dependent Children	Number of children financially supported by the applicant	
24	Home Phone Number Available	Whether a home phone number was provided	0 = No, 1 = Yes
25	Mobile Phone Number Available	Whether a mobile phone number was provided	0 = No, 1 = Yes
26	Employer Phone Number Available	Whether an employer phone number was provided	0 = No, 1 = Yes
27	Driver Licence Number Available	Whether a driver's licence number was provided	0 = No, 1 = Yes
Submission Format
Use submission_template_for_test.csv to prepare your submission.

Keep the ID column and row order unchanged.
Replace the values in the 12-Month Default Flag column with your predicted probabilities of default, ranging from 0 to 1.
Submit predictions for all 137,091 records in the test data set.
Keep the column names unchanged and save the file in CSV format.
Model performance will be evaluated using AUC (Area Under the ROC Curve).

# Objective
The objective is to predict whether an AIFUL loan account is in default status as of month 12 after loan origination, using customer information available at the time of application.

For each record in the test data, predict the probability that 12-Month Default Flag is 1 (Default).

Evaluation Metric
Model performance will be evaluated using the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
ROC-AUC measures how well a model distinguishes between default and non-default accounts.
The ROC (Receiver Operating Characteristic) curve plots the true positive rate (Sensitivity) against the false positive rate (1 − Specificity) at different thresholds.
AUC (Area Under the Curve) is the area under the ROC curve. Values range from 0 to 1, with higher values indicating better performance. A value of 0.5 corresponds to random ranking.
ROC-AUC evaluates predictions across thresholds, without requiring a single classification threshold.
Reference (scikit-learn)

Submission File Format
Prepare a CSV file using submission_template_for_test.csv and save it with UTF-8 encoding.

The file must include a header row and predictions for all 137,091 records in the test data. Extra rows or columns will result in an error.

The submission file must contain only the following columns, in this order:

ID
12-Month Default Flag
Keep the ID values and row order exactly as provided in the submission template. Replace the values in 12-Month Default Flag with your predicted probabilities of default, ranging from 0 to 1.

The following example illustrates the submission format. The IDs and probabilities are examples only; use the actual IDs from the submission template.

ID,12-Month Default Flag
10001,0.12
10002,0.35

# Nameing Rules
During the competition, provisional rankings will be displayed on the Public leaderboard (LB). Final rankings will be determined entirely by AUC on the Private test data. The top 30 teams will qualify for the Final Round.
Each team may select up to two submissions for Private score evaluation. On the Submissions page, check “Final submission” for each selected entry.
If no submissions are selected, the submission with the highest Public score will automatically be used as the final submission.
If teams have the same score, the team with the earlier submission time will rank higher.
If misconduct is discovered, even after the competition has ended, the team responsible will be disqualified and the rankings will be adjusted accordingly.
Timeline
Submission Deadline: November 14, 2026, 10:00 Thailand Time (12:00 Japan Time)
Final Round Qualifier Announcement: November 14, 2026, 13:00 Thailand Time (15:00 Japan Time)
The 30 teams advancing to the Final Round will be announced, followed by guidance on the next steps.
Final Leaderboard Publication: November 14, 2026, 14:00 Thailand Time (16:00 Japan Time)
The final leaderboard will be published after the announcement of the qualifying teams.
Bag Usage
Each participant must use only one account.
Each team may submit up to 10 times per day.