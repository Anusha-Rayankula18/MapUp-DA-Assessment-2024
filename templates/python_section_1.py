from typing import Dict, List

import pandas as pd

# Question 1: Reverse List by N Elements
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(i, min(i + n, len(lst))):
            group.insert(0, lst[j])
        result.extend(group)
    return result


Question 2: Lists & Dictionaries
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    
    for word in lst:
        length = len(word)
        
        if length not in result:
            result[length] = []
        
        result[length].append(word)
    
    return dict(sorted(result.items()))


Question 3: Flatten a Nested Dictionary
def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(current_dict: Any, parent_key: str = '') -> Dict[str, Any]:
        items = {}
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.update(_flatten(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        items.update(_flatten(item, list_key))
                    else:
                        items[list_key] = item
            else:
                items[new_key] = value
            return items
    
    return _flatten(nested_dict)


Question 4: Generate Unique Permutations
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    all_perms = permutations(nums)
    unique_perms = set(all_perms)
    
    return [list(perm) for perm in unique_perms]


Question 5: Find All Dates in a Text
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    ""
    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    dates = re.findall(date_pattern, text)
 return dates


Question 6: Decode Polyline, Convert to DataFrame with Distances
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.
        
    Returns:
        Distance in meters between the two points.
    """
    R = 6371000  # Radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    distances = [0]  # First point has a distance of 0
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i-1]
        lat2, lon2 = coordinates[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    df['distance'] = distances
    
    return df



Question 7: Matrix Rotation and Transformation
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    with the sum of its row and column excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - i - 1] = matrix[i][j]

    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]  # Sum of row excluding current element
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]  # Sum of column excluding current element
            final_matrix[i][j] = row_sum + col_sum  # Add row and column sums
    
    return final_matrix

Question 8: Time Check
def time_check(df) -> pd.Series:
    """
    Verify the completeness of the data by checking whether the timestamps for each unique (id, id_2) pair cover
    a full 24-hour and 7-day period.
    
    Args:
        df (pandas.DataFrame): A DataFrame with columns 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'
    
    Returns:
        pd.Series: A multi-index Boolean Series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    days_of_week = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    full_day_hours = set(range(0, 24))  # Hours from 0 to 23
    
    def check_full_coverage(group):
        days_covered = set(group['start_timestamp'].dt.day_name())
        start_hours_covered = set(group['start_timestamp'].dt.hour)
        end_hours_covered = set(group['end_timestamp'].dt.hour)
        
        all_days_covered = days_of_week.issubset(days_covered)
        full_24_hours_covered = full_day_hours.issubset(start_hours_covered.union(end_hours_covered))
        
        return not (all_days_covered and full_24_hours_covered)
    
    result = df.groupby(['id', 'id_2']).apply(check_full_coverage)
    
    return result
