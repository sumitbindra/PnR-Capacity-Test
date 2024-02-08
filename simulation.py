import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

# Initialize parameters and structures
trips_per_zone = np.array([50, 1000, 1000, 4000, 5000])
n_zones = len(trips_per_zone)
total_trips = trips_per_zone.sum()
n_iterations = 30
lot_capacities = np.array([100, 100, 1000])
parking_lots = len(lot_capacities)
min_costs = np.array([2, 2, 2])  # Minimum cost for parking at each lot
cost_increase_factor = 0.8
cost_decrease_factor = 1
fixed_driving_cost = np.array([[5, 4, 6, 3, 7]])  # Fixed cost to drive from each zone
max_cost = 50  # Max cost to avoid a shadow price that's too high
cost_coeff = -0.05
mode_choice_temperature = 2
lot_choice_temperature = 3

# Initialize last_costs as a matrix where each row is a zone and each column is a parking lot
last_costs = np.tile(min_costs, (n_zones, 1))

# Initialize probabilities with an equal chance for each parking lot out of the 50% total for parking
probabilities = np.zeros((n_zones, parking_lots + 1))  # +1 for driving option


def random_initial_probabilities(zone_index, n_parking_lots):
    parking_probs = np.random.rand(n_parking_lots)
    parking_probs /= parking_probs.sum() + 2  # Adjust to ensure sum < 1
    # Round new_probabilities to 3 decimal places before returning
    parking_probs = np.round(parking_probs, 3)
    driving_prob = 1 - parking_probs.sum()
    total_probs = np.append(parking_probs, driving_prob)
    # print(f"Init Prob for zone {zone_index+1}: {total_probs}")
    return total_probs


for i in range(n_zones):
    probabilities[i, :] = random_initial_probabilities(i, parking_lots)


def update_probabilities(costs, probabilities, lot_capacity, total_demand):
    new_probabilities = np.zeros_like(probabilities)
    utilization = total_demand / lot_capacity

    util_allmode = [0] * 5
    util_PNR_bestLot = [0] * 5
    lot_util = costs[:, :-1] * cost_coeff
    util_allmode = costs[:, -1] * cost_coeff
    util_PNR_bestLot = (costs[:, :-1].min(axis=1) + 10) * cost_coeff
    x = np.array([util_PNR_bestLot, util_allmode]) * mode_choice_temperature
    # calculate driving probability
    new_probabilities[:, -1] = softmax(x, axis=0).T[:, -1]
    # calculate lot probability based on costs
    new_probabilities[:, :-1] = softmax(lot_util * lot_choice_temperature, axis=1)
    # scale lot probability
    new_probabilities[:, :-1] = (
        new_probabilities[:, :-1] * (1 - new_probabilities[:, -1])[:, None]
    )
    return new_probabilities


# Initialize costs with an intial parking cost as lot fee and driving as fixed fee
def initialize_costs(n_zones, n_parking_lots, min_costs, fixed_driving_cost):
    # Generate initial costs based on min_costs, adding randomness
    initial_costs = np.random.rand(n_zones, n_parking_lots) + min_costs
    # Append a column with fixed driving cost for each zone
    initial_costs = np.append(initial_costs, fixed_driving_cost.T, axis=1)
    # print(f"Initial costs: \n {initial_costs}")
    return initial_costs


# Initialize costs including a column for fixed driving costs
costs = initialize_costs(n_zones, parking_lots, min_costs, fixed_driving_cost)


def update_costs(
    costs,
    total_demand,
    lot_capacity,
    min_costs,
    max_cost,
    cost_increase_factor,
    cost_decrease_factor,
    fixed_driving_cost,
):
    new_costs = np.copy(costs)  # Start with current costs
    utilization = total_demand / lot_capacity

    for lot in range(parking_lots):
        # (EXP(beta_1 * (demand)/capacity)-EXP(Beta_1))*Beta_2
        if utilization[lot] > 1:
            # Demand exceeds capacity, increase cost
            adjustment_factor = (
                np.exp(cost_increase_factor * utilization[lot])
                - np.exp(cost_increase_factor)
            ) * cost_decrease_factor  # (utilization[lot] - 1) * cost_increase_factor
        # elif utilization[lot] < 0.5:
        #    # If the lot is less than 50% full, make parking a lot more attractive by reducing cost significantly
        #    adjustment_factor = (utilization[lot] - 1) * cost_decrease_factor * 10
        else:
            # Demand is below capacity, decrease cost to encourage more parking
            adjustment_factor = (
                np.exp(cost_increase_factor * utilization[lot])
                - np.exp(cost_increase_factor)
            ) * cost_decrease_factor  # (utilization[lot] - 1) * cost_decrease_factor

        # Apply adjustment factor to parking costs, not affecting the fixed driving cost
        new_costs[:, lot] = np.clip(
            costs[:, lot] + adjustment_factor, min_costs[lot], max_cost
        )

    # Ensure driving cost remains unchainged fixed after these calculations
    new_costs[:, -1] = fixed_driving_cost

    # Round the costs to 2 decimal places for clarity
    new_costs = np.round(new_costs, 2)

    # print utilization and new cost
    print(f"Starting costs: \n {costs}")
    print(f"Starting demand: \n {total_demand}")
    print(f"Lot capacity: \n {lot_capacity}")
    print(f"Starting utilization: \n {utilization}")
    print(f"Updated costs: \n {new_costs}")
    return new_costs


# Additional code for visualization
utilization_history = np.zeros((n_iterations, parking_lots))
cost_history = np.zeros(
    (n_iterations, parking_lots)
)  # Store only parking costs for simplicity
demand_history = np.zeros((n_iterations, parking_lots))
drive_vs_park = np.zeros((n_iterations, 2))

for iteration in range(n_iterations):
    total_demand = np.zeros(parking_lots)
    zone_demand = np.zeros(
        (n_zones, parking_lots)
    )  # Demand from each zone for each parking lot

    for zone in range(n_zones):
        zone_probabilities = probabilities[zone, :-1]  # Exclude driving probability
        zone_demand[zone, :] = np.random.binomial(
            trips_per_zone[zone], zone_probabilities
        )
        total_demand += zone_demand[zone, :]

    print(f"Zone demand after iteration {iteration+1} : \n {np.round(zone_demand,0)}")

    demand_history[iteration] = total_demand

    costs = update_costs(
        costs,
        total_demand,
        lot_capacities,
        min_costs,
        max_cost,
        cost_increase_factor,
        cost_decrease_factor,
        fixed_driving_cost,
    )

    # Update probabilities based on the new costs and demand
    probabilities = update_probabilities(
        costs, probabilities, lot_capacities, total_demand
    )

    # Calculate utilization for visualization
    drive_vs_park[iteration, 0] = total_trips - np.sum(zone_demand)  # Those who drive
    drive_vs_park[iteration, 1] = np.sum(zone_demand)  # Those who park
    utilization_history[iteration] = total_demand / lot_capacities
    cost_history[iteration] = costs[:, :-1].mean(
        axis=0
    )  # Average cost per lot, excluding driving

# Chart 1: Utilization, Demand, and Parking Cost for Each Lot by Iteration
for lot in range(parking_lots):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.plot(utilization_history[:, lot], label="Utilization")
    plt.title(f"Lot {lot+1} Utilization")
    plt.xlabel("Iteration")
    plt.ylabel("Utilization")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(demand_history[:, lot], label="Demand")
    plt.title(f"Lot {lot+1} Demand")
    plt.xlabel("Iteration")
    plt.ylabel("Demand")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(cost_history[:, lot], label="Cost")
    plt.title(f"Lot {lot+1} Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Chart 2: Percentage of Driving and Parking Trips by Iteration
plt.figure(figsize=(10, 5))
plt.plot(drive_vs_park[:, 0] / total_trips * 100, label="Percentage Drive")
plt.plot(drive_vs_park[:, 1] / total_trips * 100, label="Percentage Park")
plt.title("Percentage of Driving vs. Parking Trips")
plt.xlabel("Iteration")
plt.ylabel("Percentage")
plt.legend()
plt.show()
