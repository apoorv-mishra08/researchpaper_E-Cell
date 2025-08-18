import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Structured_Survey_Dataset.csv')

print("="*80)
print("E-CELL RESEARCH PAPER: first STATISTICAL ANALYSIS")
print("="*80)

print("3A. DESCRIPTIVE STATISTICS")



print("\n1. SAMPLE CHARACTERISTICS")
print("-" * 40)
n_total = len(df)
print(f"Total Sample Size: {n_total}")

gender_dist = df['Gender'].value_counts()
print(f"\nGender Distribution:")
for gender, count in gender_dist.items():
    percentage = (count/n_total)*100
    print(f"  {gender}: {count} ({percentage:.1f}%)")


eng_degree = df['Engineering_Degree'].value_counts()
print(f"\nEngineering Degree Status:")
for status, count in eng_degree.items():
    percentage = (count/n_total)*100
    status_label = "Yes" if status == 1 else "No"
    print(f"  {status_label}: {count} ({percentage:.1f}%)")


year_dist = df['Year_of_Study'].value_counts().sort_index()
print(f"\nYear of Study Distribution:")
year_labels = {1: "First Year", 2: "Second Year", 3: "Third Year", 4: "Fourth Year", 5: "Graduated"}
for year, count in year_dist.items():
    percentage = (count/n_total)*100
    print(f"  {year_labels.get(year, f'Year {year}')}: {count} ({percentage:.1f}%)")

print(f"\n2. E-CELL EXPOSURE AND PARTICIPATION")
print("-" * 40)

ecell_exists = df['ECell_Exists'].value_counts()
print(f"E-Cell exists in college:")
for status, count in ecell_exists.items():
    percentage = (count/n_total)*100
    status_label = "Yes" if status == 1 else "No"
    print(f"  {status_label}: {count} ({percentage:.1f}%)")

case_study_part = df['CaseStudy_Participation'].value_counts()
print(f"\nCase Study Participation:")
for status, count in case_study_part.items():
    percentage = (count/n_total)*100
    status_label = "Yes" if status == 1 else "No"
    print(f"  {status_label}: {count} ({percentage:.1f}%)")

print(f"\n3. CAREER GOALS ANALYSIS")
print("-" * 40)
career_cols = [col for col in df.columns if 'CareerGoal_' in col and not col.endswith('.1')]
career_goals = {}
for col in career_cols:
    goal_name = col.replace('CareerGoal_', '').replace('_', ' ').title()
    count = df[col].sum()
    percentage = (count/n_total)*100
    career_goals[goal_name] = {'count': count, 'percentage': percentage}
    print(f"  {goal_name}: {count} ({percentage:.1f}%)")


print(f"\n4. FUTURE PLANS")
print("-" * 40)

mba_plans = df['MBA_Plan'].value_counts()
print(f"MBA Plans:")
for plan, count in mba_plans.items():
    percentage = (count/n_total)*100
    if pd.isna(plan):
        print(f"  No Response: {count} ({percentage:.1f}%)")
    else:
        plan_label = "Yes" if plan == 1 else "No"
        print(f"  {plan_label}: {count} ({percentage:.1f}%)")

entr_plans = df['Entrepreneurship_Plan'].value_counts()
print(f"\nEntrepreneurship Plans:")
for plan, count in entr_plans.items():
    percentage = (count/n_total)*100
    if pd.isna(plan):
        print(f"  No Response: {count} ({percentage:.1f}%)")
    else:
        plan_label = "Yes" if plan == 1 else "No"
        print(f"  {plan_label}: {count} ({percentage:.1f}%)")


print(f"\n5. EFFECTIVENESS RATINGS")
print("-" * 40)
effectiveness_cols = ['SoftSkills_Importance', 'CaseStudy_Effectiveness', 'Events_Effectiveness']
for col in effectiveness_cols:
    col_name = col.replace('_', ' ').title()
    mean_val = df[col].mean()
    std_val = df[col].std()
    median_val = df[col].median()
    print(f"{col_name}:")
    print(f"  Mean: {mean_val:.2f} ± {std_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Range: {df[col].min():.0f} - {df[col].max():.0f}")


print(f"\n6. BARRIERS TO PARTICIPATION")
print("-" * 40)
barrier_cols = [col for col in df.columns if 'Barrier_' in col and not col.endswith('.1')]
barrier_analysis = {}
for col in barrier_cols:
    barrier_name = col.replace('Barrier_', '').replace('_', ' ').title()
    count = df[col].sum()
    percentage = (count/n_total)*100
    barrier_analysis[barrier_name] = percentage
    print(f"  {barrier_name}: {count} ({percentage:.1f}%)")


print("3B. CORRELATION/ASSOCIATION TESTS")
print("-"*60)

# Test H1: E-Cell participation vs Entrepreneurship interest
print("\n1. HYPOTHESIS TESTING")
print("-" * 40)

# H1: Students who participate in E-Cell activities report higher interest in entrepreneurship
print("H1: E-Cell participation vs Entrepreneurship interest")
#  participation variable (if case study participation = 1, then participated)
df['Participated'] = df['CaseStudy_Participation']
#  entrepreneurship plan as interest measure
df_clean_h1 = df.dropna(subset=['Participated', 'Entrepreneurship_Plan'])

if len(df_clean_h1) > 0:
    contingency_h1 = pd.crosstab(df_clean_h1['Participated'], df_clean_h1['Entrepreneurship_Plan'])
    print("Contingency Table:")
    print(contingency_h1)
    
    chi2_h1, p_h1, dof_h1, expected_h1 = chi2_contingency(contingency_h1)
    print(f"Chi-square statistic: {chi2_h1:.4f}")
    print(f"p-value: {p_h1:.4f}")
    print(f"Degrees of freedom: {dof_h1}")
    
    #  Cramer's V for effect size
    n = contingency_h1.sum().sum()
    cramers_v_h1 = np.sqrt(chi2_h1 / (n * (min(contingency_h1.shape) - 1)))
    print(f"Cramer's V (effect size): {cramers_v_h1:.4f}")
    
    if p_h1 < 0.05:
        print("Result: REJECT NULL HYPOTHESIS - Significant association exists")
    else:
        print("Result: FAIL TO REJECT NULL HYPOTHESIS - No significant association")

# H2: Case study preference vs MBA plans
print(f"\nH2: Case study participation vs MBA plans")
df_clean_h2 = df.dropna(subset=['CaseStudy_Participation', 'MBA_Plan'])

if len(df_clean_h2) > 0:
    contingency_h2 = pd.crosstab(df_clean_h2['CaseStudy_Participation'], df_clean_h2['MBA_Plan'])
    print("Contingency Table:")
    print(contingency_h2)
    
    chi2_h2, p_h2, dof_h2, expected_h2 = chi2_contingency(contingency_h2)
    print(f"Chi-square statistic: {chi2_h2:.4f}")
    print(f"p-value: {p_h2:.4f}")
    
    cramers_v_h2 = np.sqrt(chi2_h2 / (contingency_h2.sum().sum() * (min(contingency_h2.shape) - 1)))
    print(f"Cramer's V (effect size): {cramers_v_h2:.4f}")

# H3: Barriers analysis - which is most significant
print(f"\nH3: Barriers analysis - statistical significance test")
barrier_cols_clean = [col for col in df.columns if 'Barrier_' in col and not col.endswith('.1')]
barrier_data = df[barrier_cols_clean].sum().values

# Chi-square goodness of fit test for equal distribution of barriers
expected_equal = [len(df)] * len(barrier_data)
chi2_barriers, p_barriers = stats.chisquare(barrier_data)
print(f"Chi-square goodness of fit test for barriers:")
print(f"Chi-square statistic: {chi2_barriers:.4f}")
print(f"p-value: {p_barriers:.4f}")

if p_barriers < 0.05:
    print("Result: Barriers are NOT equally distributed - some are more significant")
else:
    print("Result: Barriers are equally distributed")

# H4: E-Cell awareness vs participation
print(f"\nH4: E-Cell existence vs participation rates")
contingency_h4 = pd.crosstab(df['ECell_Exists'], df['CaseStudy_Participation'])
print("Contingency Table:")
print(contingency_h4)

chi2_h4, p_h4, dof_h4, expected_h4 = chi2_contingency(contingency_h4)
print(f"Chi-square statistic: {chi2_h4:.4f}")
print(f"p-value: {p_h4:.4f}")

cramers_v_h4 = np.sqrt(chi2_h4 / (contingency_h4.sum().sum() * (min(contingency_h4.shape) - 1)))
print(f"Cramer's V (effect size): {cramers_v_h4:.4f}")

#
print(f"\n2. CORRELATION ANALYSIS")
print("-" * 40)

# Spearman correlations  ordinal variables
corr_vars = ['SoftSkills_Importance', 'CaseStudy_Effectiveness', 'Events_Effectiveness', 'Year_of_Study']
corr_data = df[corr_vars].dropna()

print("Spearman Correlation Matrix:")
spearman_corr = corr_data.corr(method='spearman')
print(spearman_corr.round(3))

# Specific correlations of interest
if len(corr_data) > 1:
    # Soft skills imp vs Case study effectiveness
    rho1, p1 = spearmanr(corr_data['SoftSkills_Importance'], corr_data['CaseStudy_Effectiveness'])
    print(f"\nSoft Skills Importance vs Case Study Effectiveness:")
    print(f"Spearman's rho: {rho1:.4f}, p-value: {p1:.4f}")
    
    # Year of study vs effectiveness ratings
    rho2, p2 = spearmanr(corr_data['Year_of_Study'], corr_data['Events_Effectiveness'])
    print(f"Year of Study vs Events Effectiveness:")
    print(f"Spearman's rho: {rho2:.4f}, p-value: {p2:.4f}")



print("3C. REGRESSION MODELS")


# Logistic Regression: Predicting Case Study Participation
print("\n1. LOGISTIC REGRESSION: Predicting Case Study Participation")
print("-" * 50)

# Prepare features for logistic regression
features = ['Engineering_Degree', 'ECell_Exists', 'SoftSkills_Importance', 'Year_of_Study']
target = 'CaseStudy_Participation'

# Clean data for regression
reg_data = df[features + [target]].dropna()
print(f"Sample size for regression: {len(reg_data)}")

if len(reg_data) > 10:  # Minimum sample size 
    X = reg_data[features]
    y = reg_data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    
    # Model performance
    train_score = log_reg.score(X_train, y_train)
    test_score = log_reg.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")
    
    # Coefficients and odds ratios
    print(f"\nLogistic Regression Coefficients:")
    print(f"Intercept: {log_reg.intercept_[0]:.4f}")
    
    for i, feature in enumerate(features):
        coef = log_reg.coef_[0][i]
        odds_ratio = np.exp(coef)
        print(f"{feature}: β = {coef:.4f}, OR = {odds_ratio:.4f}")
    
    # Predictions and classification report
    y_pred = log_reg.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Mathematical model equation
    print(f"\nMATHEMATICAL MODEL:")
    print(f"logit(P) = {log_reg.intercept_[0]:.4f}", end="")
    for i, feature in enumerate(features):
        coef = log_reg.coef_[0][i]
        sign = "+" if coef >= 0 else ""
        print(f" {sign}{coef:.4f}*{feature}", end="")
    print()
    print(f"Where P = Probability of Case Study Participation")

# Linear Regression: Predicting Effectiveness Ratings
print(f"\n2. LINEAR REGRESSION: Predicting Event Effectiveness")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Prepare features for linear regression
lin_features = ['Engineering_Degree', 'ECell_Exists', 'CaseStudy_Participation', 'SoftSkills_Importance']
lin_target = 'Events_Effectiveness'

lin_data = df[lin_features + [lin_target]].dropna()
print(f"Sample size for linear regression: {len(lin_data)}")

if len(lin_data) > 10:
    X_lin = lin_data[lin_features]
    y_lin = lin_data[lin_target]
    
    # Split data
    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin, test_size=0.3, random_state=42)
    
    # Fit linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_lin, y_train_lin)
    
    # Model performance
    y_pred_lin = lin_reg.predict(X_test_lin)
    r2_train = lin_reg.score(X_train_lin, y_train_lin)
    r2_test = r2_score(y_test_lin, y_pred_lin)
    rmse = np.sqrt(mean_squared_error(y_test_lin, y_pred_lin))
    
    print(f"Training R²: {r2_train:.4f}")
    print(f"Testing R²: {r2_test:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Coefficients
    print(f"\nLinear Regression Coefficients:")
    print(f"Intercept: {lin_reg.intercept_:.4f}")
    
    for i, feature in enumerate(lin_features):
        coef = lin_reg.coef_[i]
        print(f"{feature}: β = {coef:.4f}")
    
    # Mathematical model equation
    print(f"\nMATHEMATICAL MODEL:")
    print(f"Events_Effectiveness = {lin_reg.intercept_:.4f}", end="")
    for i, feature in enumerate(lin_features):
        coef = lin_reg.coef_[i]
        sign = "+" if coef >= 0 else ""
        print(f" {sign}{coef:.4f}*{feature}", end="")
    print()

# Factor Analysis (Optional)
print(f"\n3. FACTOR ANALYSIS: Identifying Underlying Dimensions")
print("-" * 50)

# Prepare data for factor analysis
factor_cols = ['SoftSkills_Importance', 'CaseStudy_Effectiveness', 'Events_Effectiveness']
factor_data = df[factor_cols].dropna()

if len(factor_data) > 10:
    # Standardize data
    scaler = StandardScaler()
    factor_data_scaled = scaler.fit_transform(factor_data)
    
    # Perform factor analysis
    fa = FactorAnalysis(n_components=2, random_state=42)
    fa.fit(factor_data_scaled)
    
    print(f"Factor Analysis Results:")
    print(f"Number of factors: 2")
    print(f"Factor loadings:")
    
    loadings = fa.components_.T
    for i, col in enumerate(factor_cols):
        print(f"{col}:")
        print(f"  Factor 1: {loadings[i][0]:.4f}")
        print(f"  Factor 2: {loadings[i][1]:.4f}")


print("SUMMARY STATISTICS FOR RESEARCH PAPER")


print(f"\nKEY FINDINGS:")
print(f"1. Sample Characteristics: n={n_total}, {(df['Gender']=='Male').sum()}/{(df['Gender']=='Female').sum()} M/F ratio")
print(f"2. E-Cell Presence: {(df['ECell_Exists']==1).sum()}/{n_total} ({(df['ECell_Exists']==1).sum()/n_total*100:.1f}%) have E-Cells")
print(f"3. Case Study Participation: {(df['CaseStudy_Participation']==1).sum()}/{n_total} ({(df['CaseStudy_Participation']==1).sum()/n_total*100:.1f}%)")
print(f"4. Mean Soft Skills Importance: {df['SoftSkills_Importance'].mean():.2f}/5.0")
print(f"5. Mean Case Study Effectiveness: {df['CaseStudy_Effectiveness'].mean():.2f}/5.0")
print(f"6. Mean Events Effectiveness: {df['Events_Effectiveness'].mean():.2f}/5.0")

print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
if 'p_h1' in locals(): print(f"H1 (Participation vs Entrepreneurship): p={p_h1:.4f}")
if 'p_h2' in locals(): print(f"H2 (Case Study vs MBA): p={p_h2:.4f}")
if 'p_barriers' in locals(): print(f"H3 (Barriers distribution): p={p_barriers:.4f}")
if 'p_h4' in locals(): print(f"H4 (E-Cell vs Participation): p={p_h4:.4f}")

print(f"\nMODEL PERFORMANCE:")
if 'test_score' in locals(): print(f"Logistic Regression Accuracy: {test_score:.4f}")
if 'r2_test' in locals(): print(f"Linear Regression R²: {r2_test:.4f}")


print("="*60)