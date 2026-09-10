import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from geopy.distance import geodesic

sns.set(style="whitegrid")


class Phase3Prescriptive:

    def __init__(self, db_path="supply_chain.db", visual_dir="visuals"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.visual_dir = visual_dir
        os.makedirs(self.visual_dir, exist_ok=True)

    # Inventory Optimization

    def inventory_optimization(self, holding_cost=2.0, ordering_cost=50.0, service_level=0.95):
        print("\n=== INVENTORY OPTIMIZATION ===")
        inv = pd.read_sql("SELECT * FROM inventory_engineered", self.conn)

        demand_mean = inv["Demand"].mean()
        demand_std = inv["Demand"].std()

        # EOQ
        EOQ = np.sqrt((2 * demand_mean * ordering_cost) / holding_cost)

        # Safety Stock (assuming normal demand)
        from scipy.stats import norm
        z = norm.ppf(service_level)
        safety_stock = z * demand_std

        baseline_cost = (holding_cost * demand_mean) + ordering_cost
        optimized_cost = (holding_cost * EOQ) + ordering_cost

        results = {
            "EOQ": EOQ,
            "SafetyStock": safety_stock,
            "BaselineCost": baseline_cost,
            "OptimizedCost": optimized_cost,
            "Savings": baseline_cost - optimized_cost
        }

        pd.DataFrame([results]).to_sql("inventory_optimization", self.conn, if_exists="replace", index=False)

        plt.bar(["Baseline", "Optimized"], [baseline_cost, optimized_cost], color=["red", "green"])
        plt.title("Inventory Cost Comparison")
        plt.ylabel("Cost")
        plt.tight_layout()
        plt.savefig(os.path.join(self.visual_dir, "inventory_cost_comparison.png"), dpi=300)
        plt.close()

        print("✔ Inventory optimization completed")
        print(results)

    
    # Logistics Optimization (VRP with OR-Tools + CO2 Emissions)
    
    def logistics_optimization(self, vehicle_count=3, max_points=200):
        print("\n=== LOGISTICS OPTIMIZATION (VRP + CO₂) ===")
        logi = pd.read_sql("SELECT Latitude, Longitude, Asset_ID FROM logistics_engineered", self.conn)

        # ⚡ Optional sampling to avoid solver overload
        if len(logi) > max_points:
            logi = logi.sample(n=max_points, random_state=42)
            print(f"⚠ Too many locations, sampled {max_points} for optimization.")

        coords = list(zip(logi["Latitude"], logi["Longitude"]))
        num_locations = len(coords)

        if num_locations <= 1:
            print("⚠ Not enough locations for VRP optimization.")
            return

        dist_matrix = [[geodesic(coords[i], coords[j]).km for j in range(num_locations)] for i in range(num_locations)]

        # Emission factor (kg CO2 per km per vehicle)
        EMISSION_FACTOR = 0.25

        # Baseline: naive sequential route
        baseline_routes = []
        step = int(np.ceil(num_locations / vehicle_count))
        for v in range(vehicle_count):
            route_nodes = list(range(v * step, min((v + 1) * step, num_locations)))
            if len(route_nodes) > 1:
                dist = sum(geodesic(coords[route_nodes[i]], coords[route_nodes[i + 1]]).km
                           for i in range(len(route_nodes) - 1))
            else:
                dist = 0
            co2 = dist * EMISSION_FACTOR
            baseline_routes.append((v, "->".join(map(str, route_nodes)), dist, co2))

        # OR-Tools VRP
        manager = pywrapcp.RoutingIndexManager(num_locations, vehicle_count, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.time_limit.seconds = 30   # ⏱ limit search to 30 seconds

        solution = routing.SolveWithParameters(search_params)

        if solution:
            routes = []
            for v in range(vehicle_count):
                index = routing.Start(v)
                route = []
                route_distance = 0
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route.append(node)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, v)
                dist_km = route_distance / 1000
                co2 = dist_km * EMISSION_FACTOR
                routes.append((v, "->".join(map(str, route)), dist_km, co2))

            df_opt = pd.DataFrame(routes, columns=["Vehicle", "Opt_Route", "Opt_Distance_km", "Opt_CO2_kg"])
            df_base = pd.DataFrame(baseline_routes, columns=["Vehicle", "Base_Route", "Baseline_Distance_km", "Baseline_CO2_kg"])
            df_routes = pd.merge(df_base, df_opt, on="Vehicle", how="outer")

            # Save detailed routes
            df_routes.to_sql("logistics_optimization", self.conn, if_exists="replace", index=False)

            # Comparison (wide format)
            df_compare = df_routes[["Vehicle", "Baseline_Distance_km", "Opt_Distance_km", "Baseline_CO2_kg", "Opt_CO2_kg"]]

            # Convert to long format for distances
            df_compare_dist = df_compare.melt(
                id_vars="Vehicle",
                value_vars=["Baseline_Distance_km", "Opt_Distance_km"],
                var_name="Scenario",
                value_name="Total_Distance"
            )
            df_compare_dist["Scenario"] = df_compare_dist["Scenario"].replace({
                "Baseline_Distance_km": "Baseline",
                "Opt_Distance_km": "Optimized"
            })

            # Convert to long format for CO2
            df_compare_co2 = df_compare.melt(
                id_vars="Vehicle",
                value_vars=["Baseline_CO2_kg", "Opt_CO2_kg"],
                var_name="Scenario",
                value_name="Total_CO2"
            )
            df_compare_co2["Scenario"] = df_compare_co2["Scenario"].replace({
                "Baseline_CO2_kg": "Baseline",
                "Opt_CO2_kg": "Optimized"
            })

            # Save in schema Phase 5 expects
            df_compare_dist.to_sql("logistics_cost_comparison", self.conn, if_exists="replace", index=False)
            df_compare_co2.to_sql("logistics_co2_comparison", self.conn, if_exists="replace", index=False)

            # Plot Distance Comparison
            summary = df_compare_dist.groupby("Scenario")["Total_Distance"].sum().reset_index()
            plt.bar(summary["Scenario"], summary["Total_Distance"], color=["red", "green"])
            plt.title("Baseline vs Optimized Total Distances")
            plt.ylabel("Total Distance (km)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "logistics_cost_comparison.png"), dpi=300)
            plt.close()

            # Plot CO2 Comparison
            summary_co2 = df_compare_co2.groupby("Scenario")["Total_CO2"].sum().reset_index()
            plt.bar(summary_co2["Scenario"], summary_co2["Total_CO2"], color=["red", "green"])
            plt.title("Baseline vs Optimized CO₂ Emissions")
            plt.ylabel("Total CO₂ (kg)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "logistics_co2_comparison.png"), dpi=300)
            plt.close()

            print("✔ Logistics optimization completed (with CO₂ emissions)")
            print(df_compare)
        else:
            print("⚠ No solution found for VRP (within time limit)")

    
    # Close

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    p3 = Phase3Prescriptive(db_path="supply_chain.db", visual_dir="visuals")
    p3.inventory_optimization()
    p3.logistics_optimization(vehicle_count=5, max_points=200) 
    p3.close()
    print("\nPHASE 3 COMPLETE: Inventory + Logistics Optimization (with CO₂) done.")
