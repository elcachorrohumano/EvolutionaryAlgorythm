#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:11:37 2023

"""
import geopy.distance
from itertools import combinations
from itertools import permutations
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Team:
    """
    Represents an NBA team.

    Attributes:
    - name (str): The name of the team.
    - city (City): The City object representing the team's location.
    """

    def __init__(self, name, city):
        """Initialize a Team object with a name and city."""
        self.name = name
        self.city = city

    def __repr__(self):
        """Representation of the Team object."""
        return self.name


class City:
    """
    Represents a city with latitude and longitude coordinates.

    Attributes:
    - name (str): The name of the city.
    - latitude (float): The latitude coordinate of the city.
    - longitude (float): The longitude coordinate of the city.
    """

    def __init__(self, name, latitude, longitude):
        """Initialize a City object with name, latitude, and longitude."""
        self.name = name
        self.coords = (latitude, longitude)
    
    def distance(self, other):
        """
        Calculates the distance between two cities.

        Args:
        - other (City): Another City object.

        Returns:
        - float: The distance between the current city and the other city in kilometers.
        """
        return geopy.distance.geodesic(self.coords, other.coords).km


class Match:
    """
    Represents a match between two teams.

    Attributes:
    - home (Team): The home team.
    - visitor (Team): The visiting team.
    - city (City): The City object where the match takes place.
    """

    def __init__(self, home, visitor):
        """Initialize a Match object with home and visitor teams."""
        self.home = home
        self.visitor = visitor
        self.city = home.city
    
    def location(self):
        """Get the coordinates of the match location."""
        return self.city.coords
    
    def __repr__(self):
        """Representation of the Match object."""
        return self.home.name + " at " + self.visitor.name


class Calendar:
    """
    Represents a schedule calendar.

    Attributes:
    - schedule (list): A schedule containing Match objects.
    - grade (float): The grade assigned to the schedule.
    """

    def __init__(self, schedule, grade):
        """Initialize a Calendar object with a schedule and grade."""
        self.schedule = schedule
        self.grade = grade



# Define city coordinates for NBA cities
cities_info = {
    "Atlanta": (33.7488, -84.3877),
    "Boston": (42.3601, -71.0589),
    "Brooklyn": (40.6501, -73.94958),
    "Charlotte": (35.22709, -80.84313),
    "Chicago": (41.85003, -87.65005),
    "Cleveland": (41.4995, -81.69541),
    "Dallas": (32.78306, -96.80667),
    "Denver": (39.73915, -104.9847),
    "Detroit": (42.33143, -83.04575),
    "Houston": (29.76328, -95.36327),
    "Indianapolis": (39.76838, -86.15804),
    "Los Angeles": (34.05223, -118.24368),
    "Memphis": (35.14953, -90.04898),
    "Miami": (25.77427, -80.19366),
    "Milwaukee": (43.0389, -87.90647),
    "Minneapolis": (44.97997, -93.26384),
    "New Orleans": (29.95465, -90.07507),
    "New York": (40.78343, -73.96625),
    "Oklahoma City": (35.46756, -97.51643),
    "Orlando": (28.53834, -81.37924),
    "Philadelphia": (39.95233, -75.16379),
    "Phoenix": (33.44838, -112.07404),
    "Portland": (45.52345, -122.67621),
    "Sacramento": (38.58157, -121.4944),
    "San Antonio": (29.42412, -98.49363),
    "San Francisco": (37.77493, -122.41942),
    "Toronto": (43.70011, -79.4163),
    "Salt Lake City": (40.76078, -111.89105),
    "Washington": (38.89511, -77.03637)   
}

# Create City objects
cities = {city_name: City(city_name, lat, long) for city_name, (lat, long) in cities_info.items()}

# Define NBA teams with their information and respective cities
nba_teams_info = [
    {"name": "Hawks", "city": cities["Atlanta"]},
    {"name": "Celtics", "city": cities["Boston"]},
    {"name": "Nets", "city": cities["Brooklyn"]},
    {"name": "Hornets", "city": cities["Charlotte"]},
    {"name": "Bulls", "city": cities["Chicago"]},
    {"name": "Cavaliers", "city": cities["Cleveland"]},
    {"name": "Pistons", "city": cities["Detroit"]},
    {"name": "Pacers", "city": cities["Indianapolis"]},
    {"name": "76ers", "city": cities["Philadelphia"]},
    {"name": "Raptors", "city": cities["Toronto"]},
    {"name": "Wizards", "city": cities["Washington"]},
    {"name": "Mavericks", "city": cities["Dallas"]},
    {"name": "Nuggets", "city": cities["Denver"]},
    {"name": "Warriors", "city": cities["San Francisco"]},
    {"name": "Rockets", "city": cities["Houston"]},
    {"name": "Clippers", "city": cities["Los Angeles"]},
    {"name": "Lakers", "city": cities["Los Angeles"]},
    {"name": "Grizzlies", "city": cities["Memphis"]},
    {"name": "Pelicans", "city": cities["New Orleans"]},
    {"name": "Thunder", "city": cities["Oklahoma City"]},
    {"name": "Suns", "city": cities["Phoenix"]},
    {"name": "Trail Blazers", "city": cities["Portland"]},
    {"name": "Kings", "city": cities["Sacramento"]},
    {"name": "Spurs", "city": cities["San Antonio"]},
    {"name": "Jazz", "city": cities["Salt Lake City"]},
    {"name": "Bucks", "city": cities["Milwaukee"]},
    {"name": "Heat", "city": cities["Miami"]},
    {"name": "Knicks", "city": cities["New York"]},
    {"name": "Magic", "city": cities["Orlando"]},
    {"name": "Timberwolves", "city": cities["Minneapolis"]}
      
]

# Create Team objects
nba_teams = [Team(info["name"], info["city"]) for info in nba_teams_info]

# Function that returns a specific column of a matrix
def get_column(matrix, column_number):
    return [row[column_number] for row in matrix]

def transpose_matrix(matrix):
    """
    Transposes the given matrix.

    Args:
    - matrix (list of lists): The input matrix to be transposed.

    Returns:
    - list of lists: The transposed matrix.
    """

    # Calculate dimensions of the original matrix
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0  # Check if the matrix is empty

    # Transpose the matrix using list comprehension
    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    return transposed


def generate_schedule(teams):
    """
    Generates a schedule for given NBA teams.

    Args:
    - teams (list): List of Team objects representing NBA teams.

    Returns:
    - list of lists: Transposed schedule matrix of Match objects.
    """

    # In case of an odd number of teams, add a fictional 'BYE' team
    if len(teams) % 2 == 1:
        teams.append(Team("BYE", None))

    random.shuffle(teams)  # Shuffle the teams randomly

    # Generate combinations of teams for the first and second halves of the schedule
    first_half = [list(comb) for comb in combinations(teams, 2)]
    second_half = [[match[1], match[0]] for match in first_half]
    matches = first_half + second_half  # Combine the first and second half matches

    num_weeks = int(len(teams)-1) * 2  # Calculate the number of weeks in the schedule
    num_teams = len(teams)  # Total number of teams

    schedule = []  # Initialize the schedule list to hold matches for each week

    for j in range(num_weeks):
        week = []
        for i in range(num_teams // 2):
            # Calculate home and visitor teams for each week
            home = (j + i) % (num_teams - 1)
            visitor = (j + num_teams - i - 1) % (num_teams - 1)
            if i == 0:
                visitor = num_teams - 1

            # Assign teams to matches based on the week
            if j < num_weeks // 2:
                week.append((teams[home], teams[visitor]))
            else:
                week.append((teams[visitor], teams[home]))
        
        schedule.append(week)  # Append the week's matches to the schedule

    # Create Match objects for each scheduled match and shuffle them
    for r in range(len(schedule)):
        for c in range(len(schedule[0])):
            schedule[r][c] = list(schedule[r][c])  # Convert tuples to lists
            random.shuffle(schedule[r][c])  # Shuffle the teams in each match
            schedule[r][c] = Match(schedule[r][c][0], schedule[r][c][1])  # Create Match objects

    return transpose_matrix(schedule)  # Transpose and return the schedule matrix

def schedule_to_dataframe(matrix):
    """
    Converts a schedule matrix into a DataFrame.

    Args:
    - matrix (list of lists): The schedule matrix to convert.

    Returns:
    - pandas DataFrame: DataFrame representing the schedule matrix.
    """

    num_columns = len(matrix[0])  # Determine the number of columns in the matrix
    column_names = [f"Week {i + 1}" for i in range(num_columns)]  # Generate column names

    # Create a DataFrame with the given matrix and column names
    df = pd.DataFrame(matrix, columns=column_names)
    return df


def write_schedule_to_file(matrix, filename="schedule.txt"):
    """
    Writes the schedule matrix to a text file.

    Args:
    - matrix (list of lists): The schedule matrix to write.
    - filename (str): The name of the file to write to (default: 'schedule.txt').
    """

    with open(filename, 'w') as file:
    # Generate column titles and write them to the file
        column_titles = "\t".join([f"Week {i + 1}" for i in range(len(matrix[0]))])
        file.write(column_titles + '\n')

    # Write matrix rows to the file
    for row in matrix:
        # Convert each cell value in the row to a string and join them with tabs
        row_data = "\t".join(str(cell) for cell in row)
        file.write(row_data + '\n')

def total_distance(schedule, teams=nba_teams):
    """
    Calculates the total distance traveled by teams in the schedule.

    Args:
    - schedule (list of lists): The schedule matrix of Match objects.
    - teams (list): List of Team objects representing NBA teams (default: nba_teams).

    Returns:
    - float: The total distance traveled by teams in the schedule.
    """

    total = 0  # Initialize total distance traveled

    for team in teams:
        route = []
        route.append(team.city)  # Start from the team's city

        # Traverse through each column (week) in the schedule
        for c in range(len(schedule[0])):
            match = None

            # Find the match associated with the team for the current week
            for r in range(len(schedule)):
                if team == schedule[r][c].home or schedule[r][c].visitor:  # Check if the team is playing
                    match = schedule[r][c]
                    break
            
            route.append(match.city)  # Add the city of the match to the route

        # Calculate distances between consecutive cities in the route
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]

            # If start or end city is None (due to BYE team), use the team's city
            if start is None:
                start = team.city
            if end is None:
                end = team.city

            # Calculate and accumulate distance between cities
            total += start.distance(end)

    return total  # Return the total distance traveled by teams in the schedule

def home_visitor_differential(schedule, teams=nba_teams):
    """
    Calculates the home and visitor game differential for teams in the schedule.

    Args:
    - schedule (list of lists): The schedule matrix of Match objects.
    - teams (list): List of Team objects representing NBA teams (default: nba_teams).

    Returns:
    - int: The differential between home and visitor games for all teams.
    """

    differential = 0  # Initialize differential count

    for team in teams:
        num_local = 0  # Counter for the number of home games
        num_visitor = 0  # Counter for the number of visitor games
        
        # Traverse through each week (column) in the schedule
        for c in range(len(schedule[0])):
            match = None

            # Find the match associated with the team for the current week
            for r in range(len(schedule)):
                if team == schedule[r][c].home or schedule[r][c].visitor:  # Check if the team is playing
                    match = schedule[r][c]
                    break
            
            # Check if the match cities are valid and count home and visitor games
            if match.home.city is not None or match.visitor.city is not None:
                if match.city == team.city:
                    num_local += 1  # Increment if it's a home game
                else:
                    num_visitor += 1  # Increment if it's a visitor game

        differential += abs(num_local - num_visitor)  # Calculate the differential for the team

    return differential  # Return the differential between home and visitor games for all teams

    
        
def heuristic(schedule, teams=nba_teams):
    """
    Computes a heuristic value based on the schedule's attributes.

    Args:
    - schedule (list of lists): The schedule matrix of Match objects.
    - teams (list): List of Team objects representing NBA teams (default: nba_teams).

    Returns:
    - float: Heuristic value calculated based on distance and home-visitor differential.
    """

    value = 0  # Initialize heuristic value

    # Calculate distance and home-visitor differential and normalize them
    distance = (1 - ((total_distance(schedule) / 2500000))) * 0.5
    dif = (1 - ((home_visitor_differential(schedule) / 25000))) * 0.5

    value += distance + dif  # Add both values to get the final heuristic value

    return value  # Return the heuristic value based on schedule attributes



def mutation1(matrix, max_columns_to_swap=None):
    """
    Performs a mutation operation on the matrix by shuffling a subset of columns.

    Args:
    - matrix (numpy.ndarray): The input matrix.
    - max_columns_to_swap (int): The maximum number of columns to shuffle (default: None).

    Returns:
    - numpy.ndarray: The matrix with shuffled columns.
    """

    # Get the number of columns in the matrix
    num_columns = len(matrix[0])

    # If max_columns_to_swap is not specified, swap all columns
    if max_columns_to_swap is None:
        max_columns_to_swap = num_columns

    # Ensure that the specified max_columns_to_swap is not greater than the total number of columns
    max_columns_to_swap = min(max_columns_to_swap, num_columns)

    # Copy the original matrix to avoid modifying it in-place
    new_matrix = np.copy(matrix)

    # Generate a list of column indices to swap randomly
    columns_to_swap = np.random.choice(num_columns, size=max_columns_to_swap, replace=False)

    # Randomly shuffle the selected columns
    np.random.shuffle(columns_to_swap)

    # Reorder the columns in the new matrix based on the shuffled indices
    new_matrix = new_matrix[:, columns_to_swap]

    return new_matrix


import random

def generate_random_integers(n, m):
    """
    Generates a list of n random integers from 0 to m (inclusive).

    Args:
    - n (int): Number of random integers to generate.
    - m (int): Upper bound for the random integers.

    Returns:
    - list: List of n random integers.
    """
    random_integers = [random.randint(0, m) for _ in range(n)]
    return random_integers

def mutation2(matrix):
    """
    Performs a mutation operation on the matrix by swapping 'visitor' and 'home' attributes in Match objects.

    Args:
    - matrix (list of lists): The input matrix containing Match objects.

    Returns:
    - list of lists: The modified matrix after attribute swapping.
    """

    new_matrix = [row[:] for row in matrix]  # Make a copy of the matrix
    
    # Generate a random number of columns and rows for swapping
    n = generate_random_integers(1, len(matrix[0]) - 1)[0]
    columns = generate_random_integers(n, len(matrix[0]) - 1)
    rows = generate_random_integers(n, len(matrix) - 1)
    
    for col in columns:
        for r in rows:  # Loop through rows
            # Assuming Match is a class and you're swapping 'visitor' and 'home' attributes
            new_matrix[r][col] = Match(matrix[r][col].visitor, matrix[r][col].home)
    
    return new_matrix  # Return the modified matrix


def evolutionary_algorythm(teams=nba_teams, size=5):
    """
    Executes an evolutionary algorithm to generate the best schedule for given teams.

    Args:
    - teams (list): List of Team objects representing NBA teams (default: nba_teams).
    - size (int): Size of the population for the evolutionary algorithm (default: 5).

    Returns:
    - tuple: A tuple containing the best schedule matrix and its grade.
    """

    c = 0  # Initialize iteration count
    population = []  # Initialize the population list
    
    # Generate an initial population of schedules
    for n in range(size):
        population.append(generate_schedule(teams))
    
    best = 0  # Initialize the best grade
    
    # Evaluate the grades for each schedule in the initial population
    for j in range(len(population)):
        grade = heuristic(population[j])  # Calculate grade/heuristic value for the schedule
        cal = Calendar(population[j], grade)  # Create a Calendar object with the schedule and its grade
        population[j] = cal  # Store the Calendar object back in the population list
    
    # Evolutionary algorithm iterations
    for e in range(50):
        # Select subsets of the population randomly
        m1 = generate_random_integers(int(size / 2), size - 1)
        m2 = generate_random_integers(int(size / 2), size - 1)
        
        # Apply mutation1 for the schedules in subset m1
        for x in m1:
            new_matrix = mutation1(population[x].schedule)
            grade = heuristic(new_matrix)  # Calculate grade for the mutated schedule
            new_calendar = Calendar(new_matrix, grade)  # Create a Calendar object for the mutated schedule
            population.append(new_calendar)  # Add the new schedule to the population
        
        # Apply mutation2 for the schedules in subset m2
        for y in m2:
            new_matrix = mutation2(population[y].schedule)
            grade = heuristic(new_matrix)  # Calculate grade for the mutated schedule
            new_calendar = Calendar(new_matrix, grade)  # Create a Calendar object for the mutated schedule
            population.append(new_calendar)  # Add the new schedule to the population
        
        new_schedule = generate_schedule(teams)  # Generate a new random schedule
        # Calculate grade for the new schedule and create a Calendar object
        population.append(Calendar(new_schedule, heuristic(new_schedule)))
        
        # Sort the population based on grade in descending order
        sorted_population = sorted(population, key=lambda x: x.grade, reverse=True)
        population = sorted_population[0:size]  # Select the top schedules from the population
        
        best = population[0].grade  # Update the best grade
        c += 1  # Increment the iteration count
        print(c, ": ", best)  # Print the iteration count and the best grade
    
    # Return the best schedule matrix and its grade
    return population[0].schedule, best

# Execute the evolutionary algorithm and retrieve the best schedule
res = evolutionary_algorythm()
best_schedule = schedule_to_dataframe(res[0])  # Convert the best schedule to a DataFrame

                              
        
    
    
    
            

            