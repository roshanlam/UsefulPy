from collections import deque
import math
import csv
import json
import os


def chunk_list(lst, size):
    """
    Split a list into smaller lists of a specified size.
    
    Args:
        lst (List): The list to be chunked.
        size (int): The size of each chunk.
        
    Returns:
        List[List]: A list of lists where each inner list is of the specified size.
    """
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def flatten_list(nested_lst):
    """
    Convert a nested list into a single list.
    
    Args:
        nested_lst (List[List]): The nested list to be flattened.
        
    Returns:
        List: A flattened list.
    """
    return [item for sublist in nested_lst for item in sublist]


def frequency_counter(lst):
    """
    Count the occurrence of each element in a list.
    
    Args:
        lst (List): The list for which frequencies are to be counted.
        
    Returns:
        Dict: A dictionary where keys are unique items from the list and values are their counts.
    """
    freq = {}
    for item in lst:
        freq[item] = freq.get(item, 0) + 1
    return freq


def find_duplicates(lst):
    """
    Return a list of duplicate items in the given list.
    
    Args:
        lst (List): The list to check for duplicates.
        
    Returns:
        List: A list of duplicate items.
    """
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def capitalize_words(s):
    """
    Capitalize the first letter of each word in a string.
    
    Args:
        s (str): The string to be capitalized.
        
    Returns:
        str: The capitalized string.
    """
    return ' '.join(word.capitalize() for word in s.split())


def safe_divide(a, b, default=0.0):
    """
    Perform division and return a default value when dividing by zero.
    
    Args:
        a (float): The numerator.
        b (float): The denominator.
        default (float, optional): The default value to return if dividing by zero. Defaults to 0.0.
        
    Returns:
        float: The result of the division or the default value if dividing by zero.
    """
    return a / b if b != 0 else default


def filter_none(lst):
    """
    Remove None values from a list.
    
    Args:
        lst (List): The list from which None values should be removed.
        
    Returns:
        List: A list without any None values.
    """
    return [item for item in lst if item is not None]


def deep_merge(dict1, dict2):
    """
    Recursively merge two dictionaries.
    
    Args:
        dict1 (Dict): The base dictionary.
        dict2 (Dict): The dictionary to merge into the base dictionary.
        
    Returns:
        Dict: The merged dictionary.
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def is_palindrome(s):
    """
    Check if a string is a palindrome.
    
    Args:
        s (str): Input string.
        
    Returns:
        bool: True if the string is a palindrome, otherwise False.
    """
    cleaned_str = ''.join(char for char in s if char.isalnum()).lower()
    return cleaned_str == cleaned_str[::-1]

def gcd(a, b):
    """
    Compute the greatest common divisor of two numbers.
    
    Args:
        a (int): First number.
        b (int): Second number.
        
    Returns:
        int: Greatest common divisor of a and b.
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """
    Compute the least common multiple of two numbers.
    
    Args:
        a (int): First number.
        b (int): Second number.
        
    Returns:
        int: Least common multiple of a and b.
    """
    return abs(a * b) // gcd(a, b)


def group_by(lst, key):
    """
    Group a list of dictionaries by a given key.
    
    Args:
        lst (List[Dict]): List of dictionaries.
        key (str): Dictionary key to group by.
        
    Returns:
        Dict[List]: Dictionary with items grouped by the key.
    """
    result = {}
    for item in lst:
        result.setdefault(item[key], []).append(item)
    return result

def compact(lst):
    """
    Remove all falsy values from a list.
    
    Args:
        lst (List): List with possible falsy values (like None, 0, empty string).
        
    Returns:
        List: A list with falsy values removed.
    """
    return [item for item in lst if item]


def rotate(lst, positions=1):
    """
    Rotate a list by a given number of positions.
    
    Args:
        lst (List): The list to be rotated.
        positions (int): Number of positions to rotate by (can be negative for left rotation).
        
    Returns:
        List: The rotated list.
    """
    if not lst:
        return []
    positions %= len(lst)
    return lst[-positions:] + lst[:-positions]


def binary_search(lst, target):
    """
    Perform binary search on a sorted list to find the target.

    Args:
        lst (List): Sorted list.
        target: The element to search for.

    Returns:
        int: Index of the target if found, otherwise -1.
    """
    left, right = 0, len(lst) - 1
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def memoize(func):
    """
    Memoize a function. Stores results of expensive function calls and returns cached result.

    Args:
        func (callable): Function to be memoized.

    Returns:
        callable: Memoized function.
    """
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

import time

def debounce(seconds):
    """
    Decorator to debounce a function's calls.

    Args:
        seconds (float): Minimum interval between two successive calls.

    Returns:
        callable: Debounced function.
    """
    def decorator(fn):
        last_called = [0]

        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed >= seconds:
                last_called[0] = time.time()
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def bfs(graph, start):
    """
    Breadth-first search on a graph.

    Args:
        graph (Dict[List]): Graph represented as an adjacency list.
        start: Starting node.

    Returns:
        List: Nodes visited in BFS order.
    """
    visited = set()
    queue = deque([start])
    output = []

    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            output.append(vertex)
            queue.extend(node for node in graph[vertex] if node not in visited)
    return output

def paginate(lst, page=1, per_page=10):
    """
    Paginate a list.

    Args:
        lst (List): List to be paginated.
        page (int): Current page number.
        per_page (int): Number of items per page.

    Returns:
        List: Paginated list for the current page.
    """
    start = (page - 1) * per_page
    end = start + per_page
    return lst[start:end]

def nested_get(dictionary, keys, default=None):
    """
    Get a nested key from a dictionary.

    Args:
        dictionary (Dict): Dictionary to be searched.
        keys (List[str]): List of nested keys.
        default: Default value to return if key is not found.

    Returns:
        Value at the nested key or default.
    """
    for key in keys:
        if dictionary is None or key not in dictionary:
            return default
        dictionary = dictionary[key]
    return dictionary

def compose(*functions):
    """
    Compose multiple functions.

    Args:
        *functions: Functions to be composed.

    Returns:
        callable: Single function composed of the input functions.
    """
    def composed_function(x):
        for func in reversed(functions):
            x = func(x)
        return x
    return composed_function

def unique(lst):
    """
    Get unique elements from a list while maintaining order.

    Args:
        lst (List): Input list.

    Returns:
        List: List with unique elements.
    """
    seen = set()
    return [item for item in lst if item not in seen and not seen.add(item)]


def transpose(matrix):
    """
    Transpose a matrix.

    Args:
        matrix (List[List]): Matrix to be transposed.

    Returns:
        List[List]: Transposed matrix.
    """
    return [list(row) for row in zip(*matrix)]


def ngrams(lst, n=2):
    """
    Generate n-grams from a list.

    Args:
        lst (List): List to generate n-grams from.
        n (int, optional): Size of each n-gram. Default is 2.

    Returns:
        List[Tuple]: List of n-gram tuples.
    """
    return [tuple(lst[i:i+n]) for i in range(len(lst) - n + 1)]


def partition(lst, predicate):
    """
    Partition a list into two lists based on a predicate.

    Args:
        lst (List): List to be partitioned.
        predicate (Callable): A function to determine partitioning.

    Returns:
        Tuple[List, List]: Two lists - first one with elements where predicate is True, second one otherwise.
    """
    trues, falses = [], []
    for item in lst:
        (trues if predicate(item) else falses).append(item)
    return trues, falses

def mean(lst):
    """
    Calculate the mean of a list of numbers.

    Args:
        lst (List[float]): List of numbers.

    Returns:
        float: The mean.
    """
    return sum(lst) / len(lst) if lst else 0.0


def std_dev(lst):
    """
    Calculate the standard deviation of a list of numbers.

    Args:
        lst (List[float]): List of numbers.

    Returns:
        float: Standard deviation.
    """
    if len(lst) < 2:
        return 0.0
    avg = mean(lst)
    var = sum((x - avg) ** 2 for x in lst) / len(lst)
    return math.sqrt(var)

def read_file(filename):
    """
    Read a file into a string.

    Args:
        filename (str): Name of the file to read.

    Returns:
        str: Contents of the file.
    """
    with open(filename, 'r') as f:
        return f.read()


def write_file(filename, content):
    """
    Write a string to a file.

    Args:
        filename (str): Name of the file to write to.
        content (str): Content to write.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        f.write(content)

def append_to_file(filename, content):
    """
    Append a string to a file.

    Args:
        filename (str): Name of the file to append to.
        content (str): Content to append.

    Returns:
        None
    """
    with open(filename, 'a') as f:
        f.write(content)

def read_lines(filename):
    """
    Read a file into a list of lines.

    Args:
        filename (str): Name of the file to read.

    Returns:
        List[str]: List of lines in the file.
    """
    with open(filename, 'r') as f:
        return f.readlines()

def write_lines(filename, lines):
    """
    Write a list of lines to a file.

    Args:
        filename (str): Name of the file to write to.
        lines (List[str]): List of lines to write.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")


def write_csv(filename, data, fieldnames):
    """
    Write a list of dictionaries to a CSV file.

    Args:
        filename (str): Name of the file to write to.
        data (List[Dict[str, Any]]): Data to write.
        fieldnames (List[str]): Headers for the CSV columns.

    Returns:
        None
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def read_csv(filename):
    """
    Read a CSV file into a list of dictionaries.

    Args:
        filename (str): Name of the file to read.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the CSV data.
    """
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_json(filename, data):
    """
    Write data to a JSON file.

    Args:
        filename (str): Name of the file to write to.
        data (Any): Data to write.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_json(filename):
    """
    Read data from a JSON file.

    Args:
        filename (str): Name of the file to read.

    Returns:
        Any: Data loaded from the JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)

def append_to_json(filename, data):
    """
    Append data to a JSON file. Assumes the JSON file contains a list.

    Args:
        filename (str): Name of the file to append to.
        data (Any): Data to append.

    Returns:
        None
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = json.load(f)
            if isinstance(content, list):
                content.append(data)
            else:
                raise ValueError("JSON content is not a list")
    else:
        content = [data]

    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)
        

def delete_from_json(filename, condition):
    """
    Delete data from a JSON file based on a condition. Assumes the JSON file contains a list.

    Args:
        filename (str): Name of the file to delete from.
        condition (Callable[[Any], bool]): A function that returns True for data you wish to delete.

    Returns:
        None
    """
    with open(filename, 'r') as f:
        content = json.load(f)

    if not isinstance(content, list):
        raise ValueError("JSON content is not a list")

    # Filter out data entries that meet the condition
    content = [item for item in content if not condition(item)]

    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)