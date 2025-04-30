import random
import pandas as pd
import os
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
from scipy.stats import mannwhitneyu, shapiro
import statsmodels.api as sm

# Initialize variables
treatment_win_count = 0
treatment_game_count = 0

q2_control_count = 0
q3_control_count = 0

control_win_count = 0
control_game_count = 0

q2_runs = []
q3_runs = []

treatment_favorite_odds = []
control_favorite_odds = []

treatment_run_magnitudes = []
control_run_magnitudes = []

TREATMENT_MIN = 150    
TREATMENT_MAX = 210
CONTROL_Q2_START_MIN = 360
CONTROL_Q2_START_MAX= 420
CONTROL_Q2_END_MIN = 180
CONTROL_Q2_END_MAX = 240
CONTROL_Q3_START_MIN = 660
CONTROL_Q3_START_MAX= 720
CONTROL_Q3_END_MIN = 480
CONTROL_Q3_END_MAX = 540
SCORE_MARGIN_MAX = 3
RUN_THRESHOLD = 7

# Loop through play-by-play and moneyline .csv files
data_dir = "data"
for year in range(2007, 2019):  
    pbp_file_path = os.path.join(data_dir, f"pbp/pbp{year}.csv")
    moneyline_file_path = os.path.join(data_dir, f"nba_betting_money_line.csv")
    
    if not os.path.exists(pbp_file_path):
        print(f"❌ Missing: {pbp_file_path}")
        continue

    if not os.path.exists(moneyline_file_path):
        print(f"❌ Missing: {moneyline_file_path}")
        continue

    # Read play-by-play and moneyline .csv's and merge them together
    pbp_df = pd.read_csv(pbp_file_path)
    print(f"📂 Processing {pbp_file_path}")
    moneyline_df = pd.read_csv(moneyline_file_path)
    filtered_df = moneyline_df[moneyline_df['book_name'] == 'Pinnacle Sports']
    df = pbp_df.merge(filtered_df, left_on='gameid', right_on='game_id', how='inner')

    # Convert time object data to seconds remaining in quarter and add it to dataframe
    def convert_clock_to_seconds(clock_str):
        try:
            minutes = int(clock_str[2:4])
            seconds = float(clock_str[5:-1])
            return minutes * 60 + seconds
        except: 
            return None
    df['seconds_remaining'] = df['clock'].apply(convert_clock_to_seconds)
        
    # Forward-fill missing values in score columns so that all events have valid numerical scores
    df['h_pts'] = df['h_pts'].ffill()
    df['a_pts'] = df['a_pts'].ffill()

    # Helper function to check if a run is valid and then log if the game was won or not
    def check_run_and_win(start_row, end_row, game_df):
        valid_run = False
        team_won = False
        h_delta = end_row['h_pts'] - start_row['h_pts']
        a_delta = end_row['a_pts'] - start_row['a_pts']
        run_margin = h_delta - a_delta
        if abs(run_margin) >= 7:
            valid_run = True
            game_sorted = game_df.sort_values(by="seconds_remaining")
            game_last_row = game_sorted.iloc[0]
            h_final = game_last_row['h_pts']
            a_final = game_last_row['a_pts']

            if((h_delta - a_delta >= RUN_THRESHOLD and h_final > a_final)
                or (a_delta - h_delta >= RUN_THRESHOLD and a_final > h_final)):
                team_won = True

        return valid_run, team_won

    # Loop through each game to detect treatment group runs (i.e. runs where teams close the second quarter on a +7 run)
    for game_id in df['gameid'].unique():
        game_df = df[df['gameid'] == game_id]

        # Set timing window, with start point between 3.5-2.5 minutes left in the 2nd quarter. And end point
        # being the end of the second quarter.
        q2_window = game_df[
        (game_df['period'] == 2) &
        (game_df['seconds_remaining'] > TREATMENT_MIN) &
        (game_df['seconds_remaining'] < TREATMENT_MAX)]
        if q2_window.empty:
            continue

        start_row = q2_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
        start_margin = abs(start_row['h_pts'] - start_row['a_pts'])

        if (start_margin > SCORE_MARGIN_MAX):
            continue
            
        q2 = game_df[(game_df['period'] == 2)]
        end_row = q2.sort_values(by="seconds_remaining", ascending=False).iloc[-1]
        
        # Check if treatment group runs are valid and log whether game was won or not
        valid_run, team_won = check_run_and_win(start_row, end_row, game_df)
        if valid_run:
            h_delta = end_row['h_pts'] - start_row['h_pts']
            a_delta = end_row['a_pts'] - start_row['a_pts']
            run_margin = abs(h_delta - a_delta)
            treatment_run_magnitudes.append(run_margin)

            treatment_game_count += 1
            price1 = game_df['price1'].iloc[0]
            price2 = game_df['price2'].iloc[0]
            if price1 < price2:
                treatment_favorite_odds.append(game_df['price1'].iloc[0])
            elif price2 < price1:
                treatment_favorite_odds.append(game_df['price2'].iloc[0])
            if team_won:
                treatment_win_count += 1    

    # Loop through each game to detect control group runs (i.e. runs where teams have a +7 run at the middle
    # of 2nd quarter or start of 3rd quarter)          
    for game_id in df['gameid'].unique():
        game_df = df[df['gameid'] == game_id]

        # Set timing window, with start point between 7-6 minutes left in the 2nd quarter. And end point
        # between 3.5–2.5 minutes left in the 2nd quarter.
        q2_window = game_df[
        (game_df['period'] == 2) &
        (game_df['seconds_remaining'] > CONTROL_Q2_START_MIN) &
        (game_df['seconds_remaining'] < CONTROL_Q2_START_MAX)]
        if q2_window.empty:
            continue

        start_row = q2_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
        start_margin = abs(start_row['h_pts'] - start_row['a_pts'])

        if (start_margin > SCORE_MARGIN_MAX):
            continue

        q2_end_window = game_df[(game_df['period'] == 2)
        & (game_df['seconds_remaining'] > CONTROL_Q2_END_MIN)
        & (game_df['seconds_remaining'] < CONTROL_Q2_END_MAX)]
        if q2_end_window.empty:
            continue
        q2_control_count += 1
        
        end_row =  q2_end_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
        q2_runs.append((start_row, end_row, game_df))

    for game_id in df['gameid'].unique():
        game_df = df[df['gameid'] == game_id]

        # Set timing window, with start point between 12-11 minutes left in the 3rd quarter. And end point
        # between 9.5–8.5 minutes left in the 3rd quarter.    
        q3_window = game_df[
        (game_df['period'] == 3) &
        (game_df['seconds_remaining'] > CONTROL_Q3_START_MIN) &
        (game_df['seconds_remaining'] < CONTROL_Q3_START_MAX)]
        if q3_window.empty:
            continue

        start_row = q3_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
        start_margin = abs(start_row['h_pts'] - start_row['a_pts'])

        if (start_margin > SCORE_MARGIN_MAX):
            continue

        q3_end_window = game_df[(game_df['period'] == 3)
        & (game_df['seconds_remaining'] > CONTROL_Q3_END_MIN)
        & (game_df['seconds_remaining'] < CONTROL_Q3_END_MAX)]
        if q3_end_window.empty:
            continue
        q3_control_count += 1

        end_row =  q3_end_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
        q3_runs.append((start_row, end_row, game_df))

# Ensure equal representation from q2 and q3 by randomly sampling an equal number of runs from each group
min_len = min(len(q2_runs), len(q3_runs))
q2_runs = random.sample(q2_runs, min_len)
q3_runs = random.sample(q3_runs, min_len)
final_control_runs = q2_runs + q3_runs

# Check if control group runs are valid and log whether game was won or not
for start_row, end_row, game_df in final_control_runs:
    valid_run, team_won = check_run_and_win(start_row, end_row, game_df)
    if valid_run:
        h_delta = end_row['h_pts'] - start_row['h_pts']
        a_delta = end_row['a_pts'] - start_row['a_pts']
        run_margin = abs(h_delta - a_delta)
        control_run_magnitudes.append(run_margin)

        control_game_count += 1
        price1 = game_df['price1'].iloc[0]
        price2 = game_df['price2'].iloc[0]
        if price1 < price2:
            control_favorite_odds.append(game_df['price1'].iloc[0])
        elif price2 < price1:
            control_favorite_odds.append(game_df['price2'].iloc[0])
        if team_won:
            control_win_count += 1

print("\n--- Results ---")
# Treatment versus control group win rate
print(f"Treatment: {treatment_win_count}/{treatment_game_count} = {treatment_win_count / treatment_game_count:.2%}")
print(f"Control:   {control_win_count}/{control_game_count} = {control_win_count / control_game_count:.2%}")

# Perform two-sided z-test to detect if there is statistical significance between the two groups
successes = [treatment_win_count, control_win_count]
nobs = [treatment_game_count, control_game_count]
z_stat, p_value = proportions_ztest(count=successes, nobs=nobs, alternative='two-sided')
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value:     {p_value:.4f}")

# Calculate observed difference
p1 = treatment_win_count / treatment_game_count
p2 = control_win_count / control_game_count
diff = p1 - p2
print(f"Observed Difference: {diff*100:.2f}%")

# Calculate 95% confidence interval of observed difference
se = np.sqrt((p1 * (1 - p1)) / treatment_game_count + (p2 * (1 - p2)) / control_game_count)
z = 1.96
lower = diff - z * se
upper = diff + z * se
print(f"95% Confidence Interval: [{lower*100:.2f}%, {upper*100:.2f}%]")

# Compare favorite odds between treatment and control group
print("Treatment Favorite Odds", np.mean(treatment_favorite_odds))
print("Control Favorite Odds", np.mean(control_favorite_odds))

# Perform Shapiro-Wilk tests to check if treatment and control group odds are normally distributed.
# This determines whether to use a parametric (t-test) or non-parametric (Mann-Whitney U) comparison.
print("\nShapiro-Wilk Test for Normality:")
shapiro_stat_treat, p_treat = shapiro(treatment_favorite_odds)
print(f"Treatment: p = {p_treat:.4f} → {'Not normal' if p_treat < 0.05 else 'Normal'}")
shapiro_stat_ctrl, p_ctrl = shapiro(control_favorite_odds)
print(f"Control:   p = {p_ctrl:.4f} → {'Not normal' if p_ctrl < 0.05 else 'Normal'}")

# Since odds are not normally distributed, use the Mann-Whitney U test to compare medians
# between treatment and control groups
u_stat, p_val = mannwhitneyu(treatment_favorite_odds, control_favorite_odds, alternative='two-sided')
print(f"Mann-Whitney U (odds): U = {u_stat:.4f}, p = {p_val:.4f}")
# Since there is no statistically significant difference in favorite odds, no further adjustment to the results is needed.

# Calculate average run magnitude for treatment and control groups
print(f"Avg Treatment Run Magnitude: {np.mean(treatment_run_magnitudes):.2f}")
print(f"Avg Control Run Magnitude:   {np.mean(control_run_magnitudes):.2f}")

# Perform Shapiro-Wilk tests to check if treatment and control group run magnitudes are normally distributed.
# This determines whether to use a parametric (t-test) or non-parametric (Mann-Whitney U) comparison.
print("\nShapiro-Wilk Test for Normality:")
shapiro_stat_treat, p_treat = shapiro(treatment_run_magnitudes)
print(f"Treatment: p = {p_treat:.4f} → {'Not normal' if p_treat < 0.05 else 'Normal'}")
shapiro_stat_ctrl, p_ctrl = shapiro(control_run_magnitudes)
print(f"Control:   p = {p_ctrl:.4f} → {'Not normal' if p_ctrl < 0.05 else 'Normal'}")

# Since run magnitudes are not normally distributed, use the Mann-Whitney U test to compare medians
# between treatment and control groups
u_stat, p_val = mannwhitneyu(treatment_run_magnitudes, control_run_magnitudes, alternative='two-sided')
print(f"Mann-Whitney U: {u_stat:.4f}, P-value: {p_val:.4f}")

# Since the difference in run magnitudes between treatment and control groups is statistically significant,
# perform logistic regression to statistically adjust win rates based on run magnitude.

# Build dataset: one row per game, with outcome and run magnitude
wins = [1] * treatment_win_count + [0] * (treatment_game_count - treatment_win_count) + \
       [1] * control_win_count + [0] * (control_game_count - control_win_count)
run_magnitudes = treatment_run_magnitudes + control_run_magnitudes
# Create DataFrame
df = pd.DataFrame({
    "win": wins,
    "run_magnitude": run_magnitudes
})
# Add constant term for intercept
X = sm.add_constant(df["run_magnitude"])
y = df["win"]
# Run logistic regression
model = sm.Logit(y, X).fit()
# Show results
print(model.summary())
# Interpret as win rate increase per run point
odds_ratio = np.exp(model.params["run_magnitude"])
print(f"\nOdds ratio per run point: {odds_ratio:.4f}")
approx_win_pct_gain = (odds_ratio - 1) / (odds_ratio + 1)  # ≈ win probability boost
print(f"Approx. win rate increase per run point: {approx_win_pct_gain*100:.2f}%")
# After accounting for run magnitude, there is still no meaningful difference in win rate and therefore
# no further statistical adjustment is necessary. The final results remain as follows:

# Treatment Win Rate: 3387/4113 = 82.35%
# Control Win Rate: 2127/2696 = 78.89%
# Observed Difference: 3.46%
# 95% Confidence Interval: [1.52%, 5.39%]
# P-value: 0.0004