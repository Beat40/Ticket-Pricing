from stakes_engine import StakesEngine
import json

def test_stakes():
    engine = StakesEngine()
    
    # Mock standings
    # 1. Man City: 85 pts
    # 2. Arsenal: 84 pts
    # 4. Liverpool: 70 pts
    # 17. Everton: 34 pts
    # 18. Burnley: 33 pts
    
    standings = []
    for i in range(20):
        points = 85 - i * 3
        standings.append({"position": i+1, "points": max(0, points)})

    # Case 1: Final matchweek title decider (MW 38)
    # Man City (1st, 85) vs Arsenal (2nd, 84)
    # MW 38 means games_remaining = 0. Urgency = 1.0.
    res1 = engine.compute_stakes_score(
        home_position=1,
        away_position=2,
        home_points=85,
        away_points=84,
        home_gd=40,
        away_gd=38,
        matchweek=38,
        standings=standings
    )
    print("Case 1 (Title Decider MW 38):")
    print(json.dumps(res1, indent=2))
    assert res1["match_stakes_label"] == "Title Decider"
    assert res1["match_stakes_score"] >= 0.75

    # Case 2: Mid-season standard match (MW 19)
    # 10th vs 11th
    # MW 19. Urgency = ~0.5
    res2 = engine.compute_stakes_score(
        home_position=10,
        away_position=11,
        home_points=25,
        away_points=24,
        home_gd=0,
        away_gd=-2,
        matchweek=19,
        standings=standings
    )
    print("\nCase 2 (Standard MW 19):")
    print(json.dumps(res2, indent=2))
    assert res2["match_stakes_label"] == "Standard"

    # Case 3: Relegation six-pointer MW 35
    # 17th vs 18th
    # MW 35. Urgency = ~0.9
    # 17th: 34 pts, 18th: 33 pts. 
    # 18th in mock standings has points = 85 - 17*3 = 85 - 51 = 34.
    # Let's adjust standings for this case.
    standings_relg = []
    for i in range(20):
        if i == 17: points = 33 # 18th
        elif i == 16: points = 34 # 17th
        else: points = 85 - i * 3
        standings_relg.append({"position": i+1, "points": points})

    res3 = engine.compute_stakes_score(
        home_position=17,
        away_position=18,
        home_points=34,
        away_points=33,
        home_gd=-20,
        away_gd=-25,
        matchweek=35,
        standings=standings_relg
    )
    print("\nCase 3 (Relegation 6-pointer MW 35):")
    print(json.dumps(res3, indent=2))
    assert res3["match_stakes_label"] == "Relegation Six-Pointer"
    assert res3["match_stakes_score"] >= 0.75

if __name__ == "__main__":
    test_stakes()
