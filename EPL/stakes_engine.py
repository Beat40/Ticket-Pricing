import numpy as np

class StakesEngine:
    """
    Computes the dynamic stakes score (0.0 to 1.0) for an EPL match based on
    live standings, points gaps, and the stage of the season.
    """

    THRESHOLDS = {
        "title": 1,              # Position 1
        "champions_league": 4,   # Top 4
        "europa": 6,             # Top 6  
        "conference": 7,         # Top 7
        "relegation": 17,        # Bottom 3 starts at 18
    }

    def compute_stakes_score(
        self,
        home_position: int,
        away_position: int,
        home_points: int,
        away_points: int,
        home_gd: int,
        away_gd: int,
        matchweek: int,
        standings: list  # Full 20-club standings, sorted by points then GD
    ) -> dict:
        """
        Computes a continuous stakes score (0.0 - 1.0) and associated labels.
        """
        # games_remaining should include the current match for contention logic
        games_remaining = 38 - matchweek + 1
        
        # Urgency multiplier: stakes mean more as season ends
        # At matchweek 1: urgency = 0.026 (approx 0.1 as per briefing's spirit)
        # At matchweek 38: urgency = 1.0
        # The briefing says MW 1: 0.1, MW 38: 1.0. 
        # A simple linear map: 0.1 + (matchweek - 1) * (0.9 / 37)
        if matchweek <= 1:
            urgency = 0.1
        else:
            urgency = 0.1 + (matchweek - 1) * (0.9 / 37)
        
        # Points available remaining
        max_points_remaining = games_remaining * 3
        
        # Ensure we have enough clubs in standings
        if len(standings) < 20:
            # Fallback for early simulation if standings aren't fully populated
            return {
                "match_stakes_score": 0.1,
                "match_stakes_label": "Standard",
                "title_score": 0.0,
                "top4_score": 0.0,
                "relegation_score": 0.0,
                "euro_score": 0.0,
                "games_remaining": games_remaining,
                "urgency": urgency
            }

        # --- TITLE RACE ---
        leader_points = standings[0]["points"]
        home_title_gap = leader_points - home_points
        away_title_gap = leader_points - away_points
        
        # Both clubs in title contention (briefing: <= max_points_remaining * 0.5)
        home_title_alive = home_title_gap <= max_points_remaining * 0.5
        away_title_alive = away_title_gap <= max_points_remaining * 0.5
        
        title_score = 0.0
        if home_title_alive and away_title_alive:
            # Direct title clash
            closeness = 1 - (home_title_gap + away_title_gap) / (max_points_remaining * 2 + 1)
            title_score = closeness * urgency * 1.0
        
        # --- TOP FOUR BATTLE ---
        fourth_place_points = standings[3]["points"]
        
        home_top4_gap = fourth_place_points - home_points
        away_top4_gap = fourth_place_points - away_points
        
        # Home/Away top 4 contender (briefing: pos <= 6 and gap <= max * 0.6)
        home_top4_contender = (home_position <= 6 and 
                               home_top4_gap <= max_points_remaining * 0.6)
        away_top4_contender = (away_position <= 6 and 
                               away_top4_gap <= max_points_remaining * 0.6)
        
        top4_score = 0.0
        if home_top4_contender or away_top4_contender:
            relevance = 1 - min(max(0, home_top4_gap), max(0, away_top4_gap)) / (max_points_remaining + 1)
            top4_score = relevance * urgency * 0.80
        
        # --- RELEGATION BATTLE ---
        eighteenth_points = standings[17]["points"]
        
        # Gaps relative to 18th place
        home_relg_gap = home_points - eighteenth_points
        away_relg_gap = away_points - eighteenth_points
        
        # Home/Away threatened (briefing: pos >= 14 and gap <= max * 0.5)
        home_relg_threatened = (home_position >= 14 and 
                                home_relg_gap <= max_points_remaining * 0.5)
        away_relg_threatened = (away_position >= 14 and 
                                away_relg_gap <= max_points_remaining * 0.5)
        
        relg_score = 0.0
        if home_relg_threatened and away_relg_threatened:
            # Six-pointer — maximum intensity
            closeness = 1 - (abs(home_relg_gap) + abs(away_relg_gap)) / (max_points_remaining * 2 + 1)
            relg_score = closeness * urgency * 0.85
        elif home_relg_threatened or away_relg_threatened:
            closeness = 1 - min(abs(home_relg_gap), abs(away_relg_gap)) / (max_points_remaining + 1)
            relg_score = closeness * urgency * 0.65
        
        # --- EUROPEAN SPOTS ---
        seventh_points = standings[6]["points"]
        home_euro_gap = seventh_points - home_points
        away_euro_gap = seventh_points - away_points
        
        euro_score = 0.0
        home_euro_contender = (6 <= home_position <= 10 and 
                               home_euro_gap <= max_points_remaining * 0.5)
        away_euro_contender = (6 <= away_position <= 10 and 
                               away_euro_gap <= max_points_remaining * 0.5)
        
        if home_euro_contender or away_euro_contender:
            relevance = 1 - min(max(0, home_euro_gap), max(0, away_euro_gap)) / (max_points_remaining + 1)
            euro_score = relevance * urgency * 0.55
        
        # --- COMBINE ---
        # Take maximum of all stake dimensions
        raw_score = max(title_score, top4_score, relg_score, euro_score)
        
        # Clip and scale
        final_score = float(np.clip(raw_score, 0.0, 1.0))
        
        # Label for reporting
        if final_score >= 0.75:
            if title_score == raw_score:
                label = "Title Decider"
            elif relg_score == raw_score:
                label = "Relegation Six-Pointer"
            else:
                label = "Top Four Decider"
        elif final_score >= 0.50:
            label = "High Stakes"
        elif final_score >= 0.30:
            label = "European Spot"
        else:
            label = "Standard"
        
        return {
            "match_stakes_score": final_score,
            "match_stakes_label": label,
            "title_score": title_score,
            "top4_score": top4_score,
            "relegation_score": relg_score,
            "euro_score": euro_score,
            "games_remaining": games_remaining,
            "urgency": urgency
        }
