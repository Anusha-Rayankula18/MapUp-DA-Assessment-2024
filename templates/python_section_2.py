import pandas as pd
import numpy as np
import datetime as dt

Question 9: Distance Matrix Calculation
def calculate_distance_matrix(df) -> pd.DataFrame:
 
    ids = sorted(set(df['id_start']).union(set(df['id_end'])))
    
    distance_matrix = pd.DataFrame(np.zeros((len(ids), len(ids))), index=ids, columns=ids)
    
    for _, row in df.iterrows():
        start, end, dist = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = dist
        distance_matrix.at[end, start] = dist  # Symmetry
    
    return distance_matrix


Question 10: Unroll Distance Matrix
def unroll_distance_matrix(df) -> pd.DataFrame:
    
    unrolled_data = []
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': df.at[id_start, id_end]})

    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df

Question 11: Finding IDs within Percentage Threshold
def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
   
    reference_avg = df[df['id_start'] == reference_id]['distance'].mean()

    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1

    avg_distances = df.groupby('id_start')['distance'].mean()

    matching_ids = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index

    result_df = avg_distances.loc[matching_ids].reset_index()
    result_df.columns = ['id_start', 'average_distance']

    return result_df

Question 12: Calculate Toll Rate
def calculate_toll_rate(df) -> pd.DataFrame:
    coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    df['moto'] = df['distance'] * coefficients['moto']
    df['car'] = df['distance'] * coefficients['car']
    df['rv'] = df['distance'] * coefficients['rv']
    df['bus'] = df['distance'] * coefficients['bus']
    df['truck'] = df['distance'] * coefficients['truck']

    return df


Question 13: Calculate Time-Based Toll Rates
def calculate_time_based_toll_rates(df) -> pd.DataFrame():
    def get_discount_factor(day, start_time):
        weekday_discounts = {
            'morning': 0.8,  # 00:00:00 to 10:00:00
            'day': 1.2,      # 10:00:00 to 18:00:00
            'night': 0.8     # 18:00:00 to 23:59:59
        }
        weekend_discount = 0.7

        if day in ['Saturday', 'Sunday']:
            return weekend_discount
        else:
            if dt.time(0, 0, 0) <= start_time < dt.time(10, 0, 0):
                return weekday_discounts['morning']
            elif dt.time(10, 0, 0) <= start_time < dt.time(18, 0, 0):
                return weekday_discounts['day']
            else:
                return weekday_discounts['night']
                
    for index, row in df.iterrows():
        discount_factor = get_discount_factor(row['start_day'], row['start_time'])

        df.at[index, 'moto'] = row['moto'] * discount_factor
        df.at[index, 'car'] = row['car'] * discount_factor
        df.at[index, 'rv'] = row['rv'] * discount_factor
        df.at[index, 'bus'] = row['bus'] * discount_factor
        df.at[index, 'truck'] = row['truck'] * discount_factor

    return df
