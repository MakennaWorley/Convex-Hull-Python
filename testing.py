from bridges.data_src_dependent import data_source
from math import atan2
from sympy import Rational

import matplotlib.pyplot as plt
import time

EPSILON = 1e-9 # helps with any floating point errors


# This draws all the points the graph with matplotlib
def show_graph(cities):
    points = [(city.lon, city.lat) for city in cities]
    lats = [point[1] for point in points]
    lons = [point[0] for point in points]

    plt.figure(figsize=(10, 8))
    plt.scatter(lons, lats, s=1, color='red', alpha=1)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Cities Plotted by Latitude and Longitude")

    plt.grid(True, alpha=1)
    plt.show()


def number_of_hull_cities(cities, algorithm_name):
    print(f"The number of hull cities is: {len(cities)} for algorithm {algorithm_name}.")


# This draws the convex hull and the data points on the graph without labels
def draw_convex_hull(cities, hull, algorithm_name):
    points = [(city.lon, city.lat) for city in cities]
    lats = [point[1] for point in points]
    lons = [point[0] for point in points]

    plt.figure(figsize=(10, 8))
    plt.scatter(lons, lats, s=1, color="red", label="Cities", alpha=0.5)

    hull = sort_hull_cities(hull)

    hull_lats = [city.lat for city in hull + [hull[0]]]
    hull_lons = [city.lon for city in hull + [hull[0]]]
    plt.plot(hull_lons, hull_lats, linewidth=2, color="blue", label="Convex Hull", alpha=0.5)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{algorithm_name} Convex Hull of Cities")
    plt.legend()
    plt.show()

    number_of_hull_cities(hull, algorithm_name)


# This draws the convex hull and the data points on the graph with city name, city state labels
def draw_convex_hull_coordinates(cities, hull, algorithm_name):
    points = [(city.lon, city.lat) for city in cities]
    lats = [point[1] for point in points]
    lons = [point[0] for point in points]

    plt.figure(figsize=(10, 8))
    plt.scatter(lons, lats, s=1, color="red", label="Cities", alpha=0.5)

    hull = sort_hull_cities(hull)

    hull_lats = [city.lat for city in hull + [hull[0]]]
    hull_lons = [city.lon for city in hull + [hull[0]]]
    plt.plot(hull_lons, hull_lats, linewidth=2, color="blue", label="Convex Hull", alpha=0.5)

    for i, city in enumerate(hull):
        plt.text(city.lon, city.lat, f"({city.lon:.2f}, {city.lat:.2f})", fontsize=8, color="green")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{algorithm_name} Convex Hull of Cities")
    plt.legend()
    plt.show()

    number_of_hull_cities(hull, algorithm_name)


# This draws the convex hull and the data points on the graph with coordinate labels
def draw_convex_hull_names(cities, hull, algorithm_name):
    points = [(city.lon, city.lat) for city in cities]
    lats = [point[1] for point in points]
    lons = [point[0] for point in points]

    plt.figure(figsize=(10, 8))
    plt.scatter(lons, lats, s=1, color="red", label="Cities", alpha=0.5)

    hull = sort_hull_cities(hull)

    hull_lats = [city.lat for city in hull + [hull[0]]]
    hull_lons = [city.lon for city in hull + [hull[0]]]
    plt.plot(hull_lons, hull_lats, linewidth=2, color="blue", label="Convex Hull", alpha=0.5)

    for i, city in enumerate(hull):
        plt.text(city.lon, city.lat, f"{city.city}, {city.state}", fontsize=8, color="green")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{algorithm_name} Convex Hull of Cities")
    plt.legend()
    plt.show()

    number_of_hull_cities(hull, algorithm_name)

# Helper function for the construction of the hull
# some transformation of the points in relation to a center point via rotation around theta?
def sort_hull_cities(hull):
    centroid_lat = sum(city.lat for city in hull) / len(hull)
    centroid_lon = sum(city.lon for city in hull) / len(hull)

    def angle_from_centroid(city):
        lat, lon = city.lat, city.lon
        return atan2(lat - centroid_lat, lon - centroid_lon)

    return sorted(hull, key=angle_from_centroid)

def choose_cities(choice):
    us_cities = data_source.get_us_cities_data()

    match choice:
        case 1:
            return us_cities
        case 2:
            return [c for c in us_cities if c.state not in ("AK", "HI")]
        case 3:
            east_coast_states = {
                "ME", "NH", "MA", "RI", "CT", "NY", "NJ",
                "DE", "MD", "VA", "NC", "SC", "GA", "FL"
            }
            west_coast_states = {"CA", "OR", "WA"}

            coast_name = input("East or west coast? Example: east ").strip().lower()

            if coast_name == "east":
                return [c for c in us_cities if c.state in east_coast_states]
            elif coast_name == "west":
                return [c for c in us_cities if c.state in west_coast_states]
            else:
                print("Could not parse input. Please use 'east' or 'west'.")
                return
        case 4:
            state_abbreviation = input(
                "Enter the state abbreviation (e.g., 'NC' for North Carolina): ").strip().upper()
            state_cities = [c for c in us_cities if c.state == state_abbreviation]

            if not state_cities:
                print(f"No cities found for state: {state_abbreviation}.")
                return
            return state_cities
        case 5:
            type_input = input(
                "Population above or below population_number (e.g., 'above 1000000' or 'below 50000'): ").strip().lower()

            try:
                direction, number = type_input.split()
                number = int(number)
            except ValueError:
                print("Could not parse input. Please use 'above <number>' or 'below <number>'.")
                return

            if direction == "above":
                return [c for c in us_cities if c.population > number]
            elif direction == "below":
                return [c for c in us_cities if c.population < number]
            else:
                print("Invalid direction. Please use 'above' or 'below'.")
                return
        case _:
            print("Invalid choice. Please select a valid option.")
            return

def get_slope(city1, city2):
    if abs(city1.lon - city2.lon) < EPSILON:
        slope = float('inf')
    else:
        slope = (city2.lat - city1.lat) / (city2.lon - city1.lon)

    intercept = city1.lat - slope * city1.lon

    return slope, intercept

def get_point_placement_on_slope(slope, intercept, x, city):
    # y = mx + b rewritten to _ = mx + b - y
    if slope is float('inf'):
        return city.lon - x

    return slope * city.lon + intercept - city.lat

def cross_product_determinant(city1, city2, city3):
    lon1, lat1 = city1.lon, city1.lat
    lon2, lat2 = city2.lon, city2.lat
    lon3, lat3 = city3.lon, city3.lat

    return (lon2 - lon1) * (lat3 - lat1) - (lat2 - lat1) * (lon3 - lon1)

def cross_product_simple(city1, city2, city3):
    lon1, lat1 = Rational(city1.lon), Rational(city1.lat)
    lon2, lat2 = Rational(city2.lon), Rational(city2.lat)
    lon3, lat3 = Rational(city3.lon), Rational(city3.lat)

    value = (lon2 - lon1) * (lat3 - lat1) - (lat2 - lat1) * (lon3 - lon1)

    if value == 0:
        return 0
    return 1 if value > 0 else -1


def brute_force_convex_hull_slope(cities):
    n = len(cities)
    if n < 3: # all 2D shapes must have at least 3 points
        return cities

    hull = set()

    for i in range(n):
        for j in range(i+1, n):
            city1, city2 = cities[i], cities[j]
            x = city1.lon
            left_set, right_set = False, False

            try: # Catching undefined slopes
                slope, intercept = get_slope(city1, city2)
            except Exception as e:
                print(f"Error calculating slope between {city1} and {city2}: {e}")
                continue

            for k in range(n):
                if k == i or k == j:
                    continue

                city3 = cities[k]
                cross = get_point_placement_on_slope(slope, intercept, x, city3)

                if cross > EPSILON:
                    left_set = True
                if cross < EPSILON:
                    right_set = True

                if left_set and right_set:
                    break

            if (not left_set and right_set) or (not right_set and left_set):
                hull.add(city1)
                hull.add(city2)

    return list(hull)


def brute_force_convex_hull_determinant(cities):
    n = len(cities)
    if n < 3: # all 2D shapes must have at least 3 points
        return cities

    hull = set()

    for i in range(n):
        for j in range(i+1, n):
            city1, city2 = cities[i], cities[j]
            left_set, right_set = False, False

            for k in range(n):
                if k == i or k == j:
                    continue

                city3 = cities[k]
                cross = cross_product_determinant(city1, city2, city3)

                if cross > EPSILON:
                    left_set = True
                if cross < EPSILON:
                    right_set = True

                if left_set and right_set:
                    break

            if (not left_set and right_set) or (not right_set and left_set):
                hull.add(city1)
                hull.add(city2)

    return list(hull)


def distance_from_line(city1, city2, city3):
    lon1, lat1 = Rational(city1.lon), Rational(city1.lat)
    lon2, lat2 = Rational(city2.lon), Rational(city2.lat)
    lon3, lat3 = Rational(city3.lon), Rational(city3.lat)

    return abs((lon2 - lon1) * (lat3 - lat1) - (lat2 - lat1) * (lon3 - lon1))


def find_farthest_city(city1, city2, cities):
    max_distance = -1
    farthest_city = None
    for city3 in cities:
        distance = distance_from_line(city1, city2, city3)
        if distance > max_distance:
            max_distance = distance
            farthest_city = city3
    return farthest_city


def quickhull_rec(city1, city2, cities):
    if not cities:
        return []

    farthest_city = find_farthest_city(city1, city2, cities)
    hull = [farthest_city]

    left_set1 = [city for city in cities if cross_product_determinant(city1, farthest_city, city) > 0]
    left_set2 = [city for city in cities if cross_product_determinant(farthest_city, city2, city) > 0]

    hull += quickhull_rec(city1, farthest_city, left_set1)
    hull += quickhull_rec(farthest_city, city2, left_set2)

    return hull


def divide_and_conquer_quick_convex_hull(cities):
    cities = sort(cities)

    city1, city2 = cities[0], cities[-1]

    upper_set = [city for city in cities if cross_product_simple(city1, city2, city) > 0]
    lower_set = [city for city in cities if cross_product_simple(city2, city1, city) > 0]

    upper_hull = quickhull_rec(city1, city2, upper_set)
    lower_hull = quickhull_rec(city2, city1, lower_set)

    return [city1] + upper_hull + [city2] + lower_hull


def sort(cities):
    return sorted(cities, key=lambda city: (Rational(city.lon), Rational(city.lat)))


def is_close_to_zero(value):
    return abs(value) < EPSILON


def distance_squared(p1, p2):
    return (p1.lon - p2.lon)**2 + (p1.lat - p2.lat)**2


def find_upper_tangent(left_hull, right_hull, right_most_left, left_most_right):
    count = 0
    i, j = right_most_left, left_most_right
    while True:
        moved = False
        # Move `i` clockwise on the left hull
        while True:
            cross = cross_product_simple(right_hull[j], left_hull[i], left_hull[(i + 1) % len(left_hull)])
            if cross < -EPSILON:
                i = (i + 1) % len(left_hull)
                moved = True
                count += 1
            elif is_close_to_zero(cross):
                # Handle collinear points by checking distances
                next_i = (i + 1) % len(left_hull)
                if distance_squared(right_hull[j], left_hull[next_i]) > distance_squared(right_hull[j], left_hull[i]):
                    i = next_i
                    moved = True
                else:
                    break
            else:
                break
            if count > 1000:
                print("help")
                break

        count = 0
        # Move `j` counterclockwise on the right hull
        while True:
            cross = cross_product_simple(left_hull[i], right_hull[j], right_hull[(j - 1) % len(right_hull)])
            if cross > EPSILON:
                j = (j - 1) % len(right_hull)
                moved = True
                count += 1
            elif is_close_to_zero(cross):
                # Handle collinear points by checking distances
                prev_j = (j - 1) % len(right_hull)
                if distance_squared(left_hull[i], right_hull[prev_j]) > distance_squared(left_hull[i], right_hull[j]):
                    j = prev_j
                    moved = True
                else:
                    break
            else:
                break
            if count > 1000:
                print("help")
                break

        if not moved:
            break
    return i, j


def find_lower_tangent(left_hull, right_hull, right_most_left, left_most_right):
    count = 0
    i, j = right_most_left, left_most_right
    while True:
        moved = False
        # Move `j` clockwise on the right hull
        while True:
            cross = cross_product_simple(left_hull[i], right_hull[j], right_hull[(j + 1) % len(right_hull)])
            if cross < -EPSILON:
                j = (j + 1) % len(right_hull)
                moved = True
                count += 1
            elif is_close_to_zero(cross):
                # Handle collinear points by checking distances
                next_j = (j + 1) % len(right_hull)
                if distance_squared(left_hull[i], right_hull[next_j]) > distance_squared(left_hull[i], right_hull[j]):
                    j = next_j
                    moved = True
                else:
                    break
            else:
                break
            if count > 1000:
                print("help")
                break

        count = 0
        # Move `i` counterclockwise on the left hull
        while True:
            cross = cross_product_simple(right_hull[j], left_hull[i], left_hull[(i - 1) % len(left_hull)])
            if cross > EPSILON:
                i = (i - 1) % len(left_hull)
                moved = True
                count += 1
            elif is_close_to_zero(cross):
                # Handle collinear points by checking distances
                prev_i = (i - 1) % len(left_hull)
                if distance_squared(right_hull[j], left_hull[prev_i]) > distance_squared(right_hull[j], left_hull[i]):
                    i = prev_i
                    moved = True
                else:
                    break
            else:
                break
            if count > 1000:
                print("help")
                break

        if not moved:
            break
    return i, j


def merge_hulls(left_hull, right_hull):
    len_left = len(left_hull)
    len_right = len(right_hull)

    # Find rightmost point of left hull and leftmost point of right hull
    right_most_left = max(range(len_left), key=lambda i: Rational(left_hull[i].lon))
    left_most_right = min(range(len_right), key=lambda i: Rational(right_hull[i].lon))

    upper_left, upper_right = find_upper_tangent(left_hull, right_hull, right_most_left, left_most_right)
    lower_left, lower_right = find_lower_tangent(left_hull, right_hull, right_most_left, left_most_right)

    # Merge hulls, left and right side
    merged_hull = []

    # Traverse upper hull from upper_left to lower_left
    index = upper_left
    while True:
        merged_hull.append(left_hull[index])
        if index == lower_left:
            break
        index = (index + 1) % len_left

    # Traverse lower hull from lower_right to upper_right
    index = lower_right
    while True:
        merged_hull.append(right_hull[index])
        if index == upper_right:
            break
        index = (index + 1) % len_right

    return merged_hull


def divide_and_conquer_merge_convex_hull(cities):
    if len(cities) < 3:
        return sort(cities)

    sorted_cities = sort(cities)
    mid = len(cities) // 2

    left = sorted_cities[:mid]
    right = sorted_cities[mid:]

    left_hull = divide_and_conquer_merge_convex_hull(left)
    right_hull = divide_and_conquer_merge_convex_hull(right)

    return merge_hulls(left_hull, right_hull)


def main():
    start_time = time.time()

    print("Choose which cities to graph:")
    print("1. All cities")
    print("2. Continental US cities")
    print("3. East or West Coast cities")
    print("4. Cities in a single state")
    print("5. Cities filtered by population")

    try:
        choice = int(input("Enter your choice (1-5): ").strip())
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 5.")
        return

    cities = choose_cities(choice)
    print(len(cities))

    sorting_time = time.time()
    prework_time = sorting_time - start_time
    print(f"\tExecution time for prep work was: {prework_time:.6f} seconds")

    if cities:
        start_graph_time = time.time()
        show_graph(cities)
        end_graph_time = time.time()
        print(f"\tExecution time for drawing points was: {(end_graph_time-start_graph_time):.6f} seconds")

        start_graph_time = time.time()
        draw_convex_hull(cities, brute_force_convex_hull_slope(cities), "Brute Force Slope")
        end_graph_time = time.time()
        print(f"\tExecution time for brute force with slope convex hull was: {(end_graph_time - start_graph_time):.6f} seconds")

        start_graph_time = time.time()
        draw_convex_hull(cities, brute_force_convex_hull_determinant(cities), "Brute Force Determinant")
        end_graph_time = time.time()
        print(f"\tExecution time for brute force with determinant convex hull was: {(end_graph_time - start_graph_time):.6f} seconds")

        start_graph_time = time.time()
        draw_convex_hull(cities, divide_and_conquer_quick_convex_hull(cities), "Divide and Conquer Quick")
        end_graph_time = time.time()
        print(f"\tExecution time for divide and conquer convex hull was: {(end_graph_time - start_graph_time):.6f} seconds")

        start_graph_time = time.time()
        draw_convex_hull(cities, divide_and_conquer_merge_convex_hull(cities), "Divide and Conquer Merge")
        end_graph_time = time.time()
        print(f"\tExecution time for divide and conquer convex hull was: {(end_graph_time - start_graph_time):.6f} seconds")

    else:
        print("No cities to display.")

    print()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for all was: {execution_time:.6f} seconds")


# Run the main function
if __name__ == "__main__":
    main()