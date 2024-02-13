from utils import random, np

def get_solution(distances_to_airport, distance_matrix):
    print("Inside the function get solution")
    capacity = 7
    speed_limit = 60/60  # Convertir a km/min
    time_limit = 75  # Límite de tiempo en minutos
    iterations = 500
    tabu_size = 30
    best_solution, best_cost = tabu_search(distance_matrix, capacity, speed_limit, time_limit, iterations, tabu_size, distances_to_airport)
    print("Best solution:", best_solution)
    print("Best cost:", best_cost)
    max = 0
    for i in best_solution:
        if(len(i) > 0):
            max = max+1

    print(f"Factor de ocupación: {distance_matrix.shape[0]/max}")
    return best_solution, best_cost

def initial_solution(distance_matrix, capacity):
    print("Inside the function initial solution")
    n = len(distance_matrix)
    passengers = list(range(n))
    random.shuffle(passengers)
    return [passengers[i:i+capacity] for i in range(0, n, capacity)]


# This function calculate the cost of the solution. The greater is the value, then worst solution, the objetive is to find the smallest cost
def cost(solution, distance_matrix, distances_to_airport, time_limit, speed_limit, capacity):
    total_cost = 0
    num_vehicles = 0
    vehicle_penalty = 1000
    # Calculate distances as a haversine
    factor_distance_each_person = 1.3
    # Calculate distances as a haversine, so is not 100% real, we added this factor in order to "fix". Also to add some risk of the traffic
    factor_distance_to_airport = 1.5
    # This factor will increase the distance between each passenger (like a risk of traffic), for now it is fixed, but could be dynamic.
    # So for every passenger in a vehicule will be penalize for their distances between them, in order to find more closer
    factor_circunstancias_viales = 2.5  

    for route in solution:
        if not route:  # Skip empty routes
            continue

        num_vehicles += 1
        route_cost = 0
        current_capacity = 0

        for i in range(len(route) - 1):
            dist = distance_matrix[route[i]][route[i + 1]]
            route_cost += dist * (factor_distance_each_person + factor_circunstancias_viales)
            current_capacity += 1  # Assuming each route[i] is one unit of capacity

        # Distance to the airport for the last passenger
        route_cost += distances_to_airport[route[-1]] * factor_distance_to_airport

        # Calculate and check the route time
        route_time = route_cost / speed_limit
        if route_time > time_limit or current_capacity > capacity:
            return float('inf')

        total_cost += route_cost

    total_cost += num_vehicles * vehicle_penalty

    # Add a new penalty for underutilized vehicle capacity
    for route in solution:
        if route:
            route_capacity_utilization = len(route) / capacity
            if route_capacity_utilization < 1.0:  # Penalize underutilized capacity
                total_cost += (1 - route_capacity_utilization) * vehicle_penalty

    return total_cost
    
def select_new_centroid(route, distance_matrix):
    # Logic for selecting a new centroid
    return route[0]  

def reassign_to_new_centroid(route, new_centroid, distance_matrix):
    # Logic for reassigning data points based on the new centroid
    # This returns the route with the new_centroid at the beginning
    reordered_route = [new_centroid] + [p for p in route if p != new_centroid]
    return reordered_route
    
# Get neighbors 
def get_neighbors(solution, distance_matrix, capacity):
    neighbors = []
    for route_index, route in enumerate(solution):
        new_centroid = select_new_centroid(route, distance_matrix)
        new_solution = [list(r) for r in solution]  # Deep copy of the solution
        new_solution[route_index] = reassign_to_new_centroid(route, new_centroid, distance_matrix)
        neighbors.append(new_solution)
    return neighbors
    
    # Assign to a new vehicle
    for i in range(len(solution)):
        for k in range(len(solution[i])):
            passenger = solution[i][k]
            # Create a new solution to assign the passenger to a new vehicle
            new_solution = [route[:] for route in solution]
            new_solution[i].pop(k)
            new_solution.append([passenger])
            neighbors.append(new_solution)
    
    return neighbors

# Function that allows performing the search for the best solution based on cost and permutations}
# https://www.baeldung.com/cs/tabu-search
def tabu_search(distance_matrix, capacity, speed_limit, time_limit, iterations, tabu_size, distances_to_airport):
    print("Starting tabu search")
    current_solution = initial_solution(distance_matrix, capacity)
    best_solution = current_solution
    best_cost = cost(current_solution, distance_matrix, distances_to_airport, time_limit, speed_limit, capacity)
    tabu_list = []

    for _ in range(iterations):
        neighbors = get_neighbors(current_solution, distance_matrix, capacity)
        for neighbor in neighbors:
            if neighbor in tabu_list:
                continue  
            neighbor_cost = cost(neighbor, distance_matrix, distances_to_airport, time_limit, speed_limit, capacity)
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost
                break  # Exit the loop early if a better solution is found

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)  # Keep the tabu list within the specified size

        current_solution = best_solution  # Moves to the best solution found in this iteration

    print("Best solution found:", best_solution)
    print("With cost:", best_cost)
    return best_solution, best_cost