import numpy as np
import tsp_viewer  # <-- מייבאים את הקובץ של הקורס (לא לשכוח לשנות לו את השם לקו תחתון!)

class TSPInstance:
    def __init__(self, filepath: str):
        # שימוש בפונקציה של הקורס לקריאת הקואורדינטות!
        self.coords = tsp_viewer.parse_tsplib_coords(filepath)
        self.num_cities = len(self.coords)
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        """בונה מטריצת מרחקים מראש כדי שהחיפוש המקומי ירוץ בשניות ולא בדקות"""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                # שימוש בפונקציית המרחק של הקורס!
                dist = tsp_viewer.euclidean_distance(self.coords[i], self.coords[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def get_distance(self, u: int, v: int) -> float:
        return self.distance_matrix[u][v]