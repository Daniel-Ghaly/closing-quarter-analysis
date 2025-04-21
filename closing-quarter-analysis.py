import random
import pandas as pd
import os
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

treatment_win_count = 0
treatment_game_count = 0

q2_control_count = 0
q3_control_count = 0

control_win_count = 0
control_game_count = 0

q2_runs = []
q3_runs = []

data_dir = "data"
for year in range(1997, 2024):  
    file_path = os.path.join(data_dir, f"pbp{year}.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ Missing: {file_path}")
        continue

    print(f"📂 Processing {file_path}")
    df = pd.read_csv(file_path)

    def convert_clock_to_seconds(clock_str):
        try:
            minutes = int(clock_str[2:4])
            seconds = float(clock_str[5:-1])
            return minutes * 60 + seconds
        except: 
            return None
        
    df['seconds_remaining'] = df['clock'].apply(convert_clock_to_seconds)
        
    df['h_pts'] = df['h_pts'].ffill()
    df['a_pts'] = df['a_pts'].ffill()

    print(df.head(20))

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
    CLOSE_SCORE_MAX = 3
    RUN_THRESHOLD = 7

    for game_id in df['gameid'].unique():
        game_df = df[df['gameid'] == game_id]

        q2_window = game_df[
        (game_df['period'] == 2) &
        (game_df['seconds_remaining'] > TREATMENT_MIN) &
        (game_df['seconds_remaining'] < TREATMENT_MAX)]
        if q2_window.empty:
            continue

        
        start_row = q2_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
        start_margin = abs(start_row['h_pts'] - start_row['a_pts'])

        if (start_margin > CLOSE_SCORE_MAX):
            continue
            
        q2 = game_df[(game_df['period'] == 2)]
        end_row = q2.sort_values(by="seconds_remaining", ascending=False).iloc[-1]
        
        valid_run, team_won = check_run_and_win(start_row, end_row, game_df)
        if valid_run:
            treatment_game_count += 1
            if team_won:
                treatment_win_count += 1    
                
    for game_id in df['gameid'].unique():
        game_df = df[df['gameid'] == game_id]

        if q2_control_count == q3_control_count:   

            q2_window = game_df[
            (game_df['period'] == 2) &
            (game_df['seconds_remaining'] > CONTROL_Q2_START_MIN) &
            (game_df['seconds_remaining'] < CONTROL_Q2_START_MAX)]
            if q2_window.empty:
                continue

            start_row = q2_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
            start_margin = abs(start_row['h_pts'] - start_row['a_pts'])

            if (start_margin > CLOSE_SCORE_MAX):
                continue

            q2_end_window = game_df[(game_df['period'] == 2)
            & (game_df['seconds_remaining'] > CONTROL_Q2_END_MIN)
            & (game_df['seconds_remaining'] < CONTROL_Q2_END_MAX)]
            if q2_end_window.empty:
                continue
            q2_control_count += 1

            end_row =  q2_end_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
            q2_runs.append((start_row, end_row, game_df))
        
        else:
            
            q3_window = game_df[
            (game_df['period'] == 3) &
            (game_df['seconds_remaining'] > CONTROL_Q3_START_MIN) &
            (game_df['seconds_remaining'] < CONTROL_Q3_START_MAX)]
            if q3_window.empty:
                continue

            start_row = q3_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
            start_margin = abs(start_row['h_pts'] - start_row['a_pts'])

            if (start_margin > CLOSE_SCORE_MAX):
                continue

            q3_end_window = game_df[(game_df['period'] == 3)
            & (game_df['seconds_remaining'] > CONTROL_Q3_END_MIN)
            & (game_df['seconds_remaining'] < CONTROL_Q3_END_MAX)]
            if q3_end_window.empty:
                continue
            q3_control_count += 1

            end_row =  q3_end_window.sort_values(by="seconds_remaining", ascending=False).iloc[0]
            q3_runs.append((start_row, end_row, game_df))

            min_len = min(len(q2_runs), len(q3_runs))
            q2_runs = random.sample(q2_runs, min_len)
            q3_runs = random.sample(q3_runs, min_len)
            final_control_runs = q2_runs + q3_runs

            for start_row, end_row, game_df in final_control_runs:
                valid_run, team_won = check_run_and_win(start_row, end_row, game_df)
                if valid_run:
                    control_game_count += 1
                    if team_won:
                        control_win_count += 1

print("\n--- Results ---")
print(f"Treatment: {treatment_win_count}/{treatment_game_count} = {treatment_win_count / treatment_game_count:.2%}")
print(f"Control:   {control_win_count}/{control_game_count} = {control_win_count / control_game_count:.2%}")


# Counts of successes (wins)
successes = [6842, 10874781]

# Total observations
nobs = [8315, 13330520]

# Perform two-sided z-test
z_stat, p_value = proportions_ztest(count=successes, nobs=nobs, alternative='larger')

print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value:     {p_value:.4f}")

treatment_wins = 6842
treatment_total = 8315
control_wins = 10874781
control_total = 13330520

# Calculate proportions
p1 = treatment_wins / treatment_total
p2 = control_wins / control_total
diff = p1 - p2

# Standard error for difference of proportions
se = np.sqrt((p1 * (1 - p1)) / treatment_total + (p2 * (1 - p2)) / control_total)

# Z value for 95% confidence
z = 1.96
lower = diff - z * se
upper = diff + z * se

# Print results as percentages
print(f"Observed Difference: {diff*100:.2f}%")
print(f"95% Confidence Interval: [{lower*100:.2f}%, {upper*100:.2f}%]")