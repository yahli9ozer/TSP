import numpy as np
import tsp_viewer

class TSPInstance:
    """
    Represents a TSP problem instance.
    Handles the parsing of city coordinates and pre-computes a distance matrix
    to ensure O(1) distance lookups during the algorithm's execution.
    """
    def __init__(self, filepath: str):
        # Use the provided viewer's function to parse the coordinates
        self.coords = tsp_viewer.parse_tsplib_coords(filepath)
        self.num_cities = len(self.coords)
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Pre-computes the pairwise distance matrix.
        This is a critical optimization step that allows the local search (2-opt) 
        to run in seconds rather than minutes by avoiding repetitive distance calculations.
        """
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                # Use the provided viewer's Euclidean distance function
                dist = tsp_viewer.euclidean_distance(self.coords[i], self.coords[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
                
        return matrix

    def get_distance(self, u: int, v: int) -> float:
        """
        Retrieves the distance between city 'u' and city 'v' in O(1) time.
        """
        return self.distance_matrix[u][v]