import cv2
import numpy as np
import matplotlib.pyplot as plt
from Grid import grid_generation, get_masked_grid_points, classify_grid_points, visualise_classified_grid_points
# Use python library for Dijkstra
from dijkstar import Graph, find_path
from Grid import GridPoint

class DijkstraResult:
    def __init__(self, plugged_gp, path_info, masked_gp_id):
        self.plugged_gp = plugged_gp
        self.path = path_info.nodes
        self.total_cost = path_info.total_cost
        self.masked_gp_id = masked_gp_id

    def __repr__(self):
        return f"DijkstraResult(plugged_gp={self.plugged_gp.get_index()}, masked_gp_id={self.masked_gp_id}, total_cost={self.total_cost}, path={self.path})"


class GridDijkstraSolver:
    def __init__(self, gps):
        # Createa a Map with key as Grid Coordinate and Value being Grid Point Object
        self.grid_map = {}
        self.gps = gps
        # Find the list of Grid Points that are plugged in
        self.plugged_gps = []
        for gp in gps:
            self.grid_map[gp.get_index()] = gp
            if gp.is_plugged():
                self.plugged_gps.append(gp)

        self.graph = Graph()

        # Check 8 directions
        # directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for gp in gps:
            if gp.is_masked:
                # Add the masked GP as node
                self.graph.add_node(gp.get_index())
                continue
            if gp.type == 1 or gp.type == 2:  # Pass Over or Plugged
                r, c = gp.get_index()

                # Get the borders the wire passes through
                borders_r_indexed, borders_c_indexed = gp.borders

                # Based on the borders, organise the directions into two sets,
                # one set has higher priority (weight 1) and the other set has lower priority (weight 3)
                # e.g. the wire passes through the top and right borders so, borders_r_indexed = [-1] and borders_c_indexed = [1]
                # The directions with higher priority are: [(-1, 0), (0, 1)]
                # The directions with lower priority are: [(-1, 1), (-1, 1), (1, 1)]
                # Basically the directions that directly align with the borders have higher priority
                # The directions that are diagonal to the borders have lower priority
                high_priority_set = set()
                low_priority_set = set()
                for dr in borders_r_indexed:
                    high_priority_set.add((dr, 0))
                    low_priority_set.add((dr, -1))
                    low_priority_set.add((dr, 1))
                for dc in borders_c_indexed:
                    high_priority_set.add((0, dc))
                    low_priority_set.add((-1, dc))
                    low_priority_set.add((1, dc))


                # Create edges to the neighbours based on the directions

                for dr, dc in high_priority_set:
                    neighbor_coords = (r + dr, c + dc)

                    try:
                        neighbor_gp = self.grid_map[neighbor_coords]
                        if neighbor_gp.type == 1 or neighbor_gp.type == 2 or neighbor_gp.is_masked:  # Pass Over or Plugged
                            self.graph.add_edge(gp.get_index(), neighbor_gp.get_index(), 1)  # weight 1 for high priority
                    except KeyError:
                        pass

                for dr, dc in low_priority_set:
                    neighbor_coords = (r + dr, c + dc)

                    try:
                        neighbor_gp = self.grid_map[neighbor_coords]
                        if neighbor_gp.type == 1 or neighbor_gp.type == 2:  # Pass Over or Plugged
                            self.graph.add_edge(gp.get_index(), neighbor_gp.get_index(), 3)  # weight 3 for low priority
                    except KeyError:
                        pass

    def run_dijkstra(self) -> list[DijkstraResult]:
        # For each plugged GP, find the nearest masked GP using Dijkstra
        # return a list of DijkstraResult objects
        results: list[DijkstraResult] = []
        for gp in self.gps:
            if gp.is_plugged():
                start = gp.get_index()
                shortest_path = None
                shortest_path_length = float('inf')
                nearest_masked_gp_id = None
                for masked_gp in self.gps:
                    if masked_gp.is_masked:
                        end = masked_gp.get_index()
                        try:
                            path_info = find_path(self.graph, start, end)
                            path_length = path_info.total_cost
                            if path_length < shortest_path_length:
                                shortest_path_length = path_length
                                shortest_path = path_info
                                nearest_masked_gp_id = masked_gp.mask_id
                        except Exception as e:
                            # No path found
                            pass
                if shortest_path is not None:
                    results.append(DijkstraResult(gp, shortest_path, nearest_masked_gp_id))
                else:
                    print(f"No path found from Plugged GP at {start} to any Masked GP.")

        return results


    """
    Find all pairs of masked Grid Points that are directly connected by wires
    Given a list of Grid Points, return a list of tuples of connected plugged Grid Points
    Tuple found by running Dijkstra to check connctivity and choosing the one with lowest cost
    """
    def find_wire_connected_gps(self, gps: list[GridPoint]) -> list[(GridPoint, GridPoint)]:
        connected_gps = []
        while len(gps) > 1:
            gp1 = gps.pop(0)
            start = gp1.get_index()
            shortest_path = None
            shortest_path_length = float('inf')
            connected_gp2 = None
            for gp2 in gps:
                end = gp2.get_index()
                try:
                    path_info = find_path(self.graph, start, end)
                    path_length = path_info.total_cost
                    if path_length < shortest_path_length:
                        shortest_path_length = path_length
                        shortest_path = path_info
                        connected_gp2 = gp2
                except Exception as e:
                    # No path found
                    pass
            if shortest_path is not None and connected_gp2 is not None:
                connected_gps.append((gp1, connected_gp2))
                gps.remove(connected_gp2)
        return connected_gps