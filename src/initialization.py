import random
from typing import List, Tuple, Set
from src.instance import TSPInstance

class PopulationInitializer:
    """
    מחלקה האחראית על בניית פתרונות התחלתיים לאוכלוסייה.
    מטרתה לייצר שני מסלולים המילטוניאנים עם מינימום חפיפת קשתות.
    """
    @staticmethod
    def generate_random_permutation(num_cities: int) -> List[int]:
        """מייצר מסלול אקראי חוקי (תמורה של הערים)"""
        path = list(range(num_cities))
        random.shuffle(path)
        return path

    @staticmethod
    def generate_disjoint_path(instance: TSPInstance, forbidden_edges: Set[Tuple[int, int]]) -> List[int]:
        """
        בונה מסלול שני בצורה גרידית-הסתברותית (Nearest Neighbor משולב רולטה),
        תוך הענשה כבדה של קשתות שנמצאות ב-forbidden_edges (הקשתות של המסלול הראשון).
        """
        num_cities = instance.num_cities
        unvisited = set(range(num_cities))
        
        # בחירת עיר התחלה אקראית
        current_city = random.choice(list(unvisited))
        path = [current_city]
        unvisited.remove(current_city)
        
        while unvisited:
            candidates = list(unvisited)
            weights = []
            
            for next_city in candidates:
                # הגדרת הקשת כזוג לא מסודר (הקטן תמיד ראשון)
                edge = (min(current_city, next_city), max(current_city, next_city))
                base_dist = instance.get_distance(current_city, next_city)
                
                # אם הקשת אסורה (חופפת למסלול 1), נוסיף לה עלות מדומה חריפה מאוד
                penalty = 10000.0 if edge in forbidden_edges else 0.0
                total_cost = base_dist + penalty
                
                # נהפוך את העלות למשקל חיובי לבחירה (ככל שהעלות קטנה, המשקל גדול יותר)
                # נוסיף אפסילון קטן למניעת חלוקה באפס
                weight = 1.0 / (total_cost + 1e-6)
                weights.append(weight)
            
            # בחירת השכן הבא באמצעות הגרלת רולטה משוקללת
            total_w = sum(weights)
            if total_w == 0:
                next_city = random.choice(candidates)
            else:
                r = random.uniform(0, total_w)
                current_sum = 0
                for idx, w in enumerate(weights):
                    current_sum += w
                    if current_sum >= r:
                        next_city = candidates[idx]
                        break
            
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
            
        return path