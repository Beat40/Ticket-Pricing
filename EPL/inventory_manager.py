class InventoryManager:
    """
    Handles the segmentation of EPL stadium capacity into fixed and dynamic zones.
    """
    
    ZONE_SPLITS = {
        "hospitality": 0.08,
        "lower_tier": 0.35,
        "upper_tier": 0.28,
        "away_end": 0.05,
        "family_community": 0.05,
        "general_sale": 0.19
    }

    def get_dynamic_inventory(self, club: dict) -> dict:
        """
        Calculates the slice of inventory available for dynamic pricing.
        """
        cap = club["capacity"]
        inv_pct = club["dynamic_inventory_pct"]
        
        # Section 4 formulas
        # Dynamic Lower:   dynamic_inventory_pct × 0.55 × capacity
        # Dynamic Upper:   dynamic_inventory_pct × 0.35 × capacity
        # Dynamic General: dynamic_inventory_pct × 0.10 × capacity
        
        return {
            "Dynamic Lower": int(cap * inv_pct * 0.55),
            "Dynamic Upper": int(cap * inv_pct * 0.35),
            "Dynamic General": int(cap * inv_pct * 0.10),
            "Away End": int(cap * 0.05), # Non-dynamic but tracked
            "Hospitality": int(cap * 0.08) # Non-dynamic but tracked
        }

    def get_hospitality_base_price(self, club: dict) -> float:
        """
        Returns the fixed hospitality price based on club tier (Section 8).
        """
        tier = club["tier"]
        if tier == "big_six": return 250.0
        if tier == "established_mid": return 140.0
        return 100.0
