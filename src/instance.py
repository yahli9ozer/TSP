import math

class TSPInstance:
    """
    Handles parsing and distance calculations for TSPLIB files.
    Automatically caches all coordinate-based files (EUC_2D, GEO) into a 
    pre-calculated distance matrix for O(1) lightning-fast lookups during Local Search.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.num_cities = 0
        self.coords = []
        self.distance_matrix = []
        
        self.is_matrix_format = False
        self.matrix_format_type = "FULL_MATRIX" # default
        self.is_geo_format = False 
        
        self._load_instance()

    def _load_instance(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        reading_coords = False
        reading_matrix = False
        matrix_values = []

        for line in lines:
            line = line.strip()
            if not line or line == "EOF":
                continue

            # Parse metadata
            if line.startswith("DIMENSION"):
                self.num_cities = int(line.split()[-1].strip(': '))
                
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                weight_type = line.split()[-1].strip(': ')
                if weight_type == "GEO":
                    self.is_geo_format = True
                    
            elif line.startswith("EDGE_WEIGHT_FORMAT"):
                format_type = line.split()[-1].strip(': ')
                if format_type in ["FULL_MATRIX", "UPPER_ROW", "UPPER_DIAG_ROW", "LOWER_DIAG_ROW"]:
                    self.is_matrix_format = True
                self.matrix_format_type = format_type
            
            # Detect section changes
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                reading_matrix = False
                continue
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                reading_matrix = True
                reading_coords = False
                continue
            elif line.startswith("DISPLAY_DATA_SECTION"):
                reading_matrix = False
                reading_coords = False
                continue

            # Read the actual data
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    self.coords.append((float(parts[1]), float(parts[2])))
            
            elif reading_matrix:
                matrix_values.extend([float(val) for val in line.split()])

        n = self.num_cities
        
        # 1. Handle Explicit Matrix Files
        if self.is_matrix_format and matrix_values:
            self.distance_matrix = [[0.0] * n for _ in range(n)]
            idx = 0
            if self.matrix_format_type == "FULL_MATRIX":
                for i in range(n):
                    for j in range(n):
                        if idx < len(matrix_values):
                            self.distance_matrix[i][j] = matrix_values[idx]
                            idx += 1
                            
            elif self.matrix_format_type == "UPPER_ROW":
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if idx < len(matrix_values):
                            val = matrix_values[idx]
                            self.distance_matrix[i][j] = val
                            self.distance_matrix[j][i] = val
                            idx += 1
                            
            elif self.matrix_format_type == "UPPER_DIAG_ROW":
                for i in range(n):
                    for j in range(i, n):
                        if idx < len(matrix_values):
                            val = matrix_values[idx]
                            self.distance_matrix[i][j] = val
                            self.distance_matrix[j][i] = val
                            idx += 1

            elif self.matrix_format_type == "LOWER_DIAG_ROW":
                for i in range(n):
                    for j in range(i + 1):  
                        if idx < len(matrix_values):
                            val = matrix_values[idx]
                            self.distance_matrix[i][j] = val
                            self.distance_matrix[j][i] = val 
                            idx += 1

        # 2. OPTIMIZATION: Handle Coordinate Files 
        # Pre-calculate the entire matrix so 2-opt doesn't compute square roots!
        elif not self.is_matrix_format and self.coords:
            self.distance_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.distance_matrix[i][j] = self._calculate_raw_distance(i, j)
            
            # Force the class to use the lightning-fast matrix lookup from now on
            self.is_matrix_format = True

    def _calculate_raw_distance(self, city1: int, city2: int) -> float:
        """ Helper method to calculate real math distance just once during caching. """
        if self.is_geo_format:
            PI = 3.141592
            def to_rad(coord):
                deg = int(coord)
                minute = coord - deg
                return PI * (deg + 5.0 * minute / 3.0) / 180.0
                
            lat1, lon1 = to_rad(self.coords[city1][0]), to_rad(self.coords[city1][1])
            lat2, lon2 = to_rad(self.coords[city2][0]), to_rad(self.coords[city2][1])
            
            RRR = 6378.388
            q1 = math.cos(lon1 - lon2)
            q2 = math.cos(lat1 - lat2)
            q3 = math.cos(lat1 + lat2)
            
            inner_val = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
            inner_val = max(-1.0, min(1.0, inner_val)) 
            
            return float(int(RRR * math.acos(inner_val) + 1.0))
            
        else:
            # Standard Euclidean
            x1, y1 = self.coords[city1]
            x2, y2 = self.coords[city2]
            return math.dist((x1, y1), (x2, y2))

    def get_distance(self, city1: int, city2: int) -> float:
        """
        Always runs in O(1) time because all distances are now pre-calculated into the matrix!
        """
        return self.distance_matrix[city1][city2]