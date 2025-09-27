import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class LeaderboardManager:
    """Manages the leaderboard data storage and retrieval"""
    
    def __init__(self, data_file: str = "data/leaderboard.json"):
        self.data_file = data_file
        # Ensure data directory exists
        data_dir = os.path.dirname(self.data_file)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        self.leaderboard_data = self.load_leaderboard()
    
    def load_leaderboard(self) -> List[Dict]:
        """Load leaderboard data from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def save_leaderboard(self) -> bool:
        """Save leaderboard data to JSON file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.leaderboard_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving leaderboard: {e}")
            return False
    
    def add_score(self, player_name: str, score: int, game_duration: float = 0.0) -> bool:
        """Add a new score entry to the leaderboard"""
        if not player_name.strip():
            return False
            
        entry = {
            "player_name": player_name.strip()[:20],  # Limit name length
            "score": score,
            "game_duration": round(game_duration, 1),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": datetime.now().timestamp()
        }
        
        self.leaderboard_data.append(entry)
        
        # Sort by score (descending), then by game duration (ascending for ties)
        self.leaderboard_data.sort(key=lambda x: (-x["score"], x["game_duration"]))
        
        # Keep only top 50 entries to prevent file from growing too large
        self.leaderboard_data = self.leaderboard_data[:50]
        
        return self.save_leaderboard()
    
    def get_top_scores(self, limit: int = 10) -> List[Dict]:
        """Get top N scores from leaderboard"""
        return self.leaderboard_data[:limit]
    
    def get_player_rank(self, player_name: str) -> Optional[int]:
        """Get the rank of a specific player's best score"""
        for i, entry in enumerate(self.leaderboard_data):
            if entry["player_name"].lower() == player_name.lower():
                return i + 1
        return None
    
    def get_player_best_score(self, player_name: str) -> Optional[Dict]:
        """Get a player's best score entry"""
        for entry in self.leaderboard_data:
            if entry["player_name"].lower() == player_name.lower():
                return entry
        return None
    
    def clear_leaderboard(self) -> bool:
        """Clear all leaderboard data"""
        self.leaderboard_data = []
        return self.save_leaderboard()
    
    def get_total_players(self) -> int:
        """Get total number of unique players"""
        unique_players = set()
        for entry in self.leaderboard_data:
            unique_players.add(entry["player_name"].lower())
        return len(unique_players)