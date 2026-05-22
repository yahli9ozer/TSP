import numpy as np
import math
from typing import List, Tuple

class TSPInstance:
    """
    מחלקה האחראית על טעינת נתוני ה-TSPLIB וניהול מטריצת המרחקים בין הערים.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        # 1. טעינת הקואורדינטות מתוך הקובץ
        self.coords = self._parse_tsplib_coords(filepath)
        self.num_cities = len(self.coords)
        
        # 2. חישוב מראש של מטריצת המרחקים לביצועים מהירים
        self.distance_matrix = self._compute_distance_matrix()

    def _parse_tsplib_coords(self, filepath: str) -> List[Tuple[float, float]]:
        """קורא קובץ .tsp ומחלץ את מיקומי הערים (X, Y)"""
        coords = []
        with open(filepath, 'r') as f:
            reading_nodes = False
            for line in f:
                cleaned_line = line.strip()
                if cleaned_line == "NODE_COORD_SECTION":
                    reading_nodes = True
                    continue
                if cleaned_line == "EOF" or cleaned_line == "-1":
                    break
                if reading_nodes:
                    parts = cleaned_line.split()
                    if len(parts) >= 3:
                        # parts[0] הוא האינדקס (1, 2, 3...), parts[1] הוא X, parts[2] הוא Y
                        coords.append((float(parts[1]), float(parts[2])))
        
        if not coords:
            raise ValueError(f"לא נמצאו קואורדינטות חוקיות בקובץ: {filepath}")
        return coords

    def _compute_distance_matrix(self) -> np.ndarray:
        """מחשב מטריצה ריבועית שבה תא [i][j] מכיל את המרחק האוקלידי בין עיר i לעיר j"""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                p1 = self.coords[i]
                p2 = self.coords[j]
                # חישוב מרחק אוקלידי (EUC_2D) בהתאם לנוסחת ה-TSPLIB וקוד ה-viewer
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def get_distance(self, u: int, v: int) -> float:
        """שולף את המרחק בין עיר u לעיר v ב-O(1)"""
        return self.distance_matrix[u][v]