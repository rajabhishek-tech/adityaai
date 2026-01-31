def generate_executive_summary(health_score, domain):
    if health_score >= 80:
        quality = "high"
    elif health_score >= 50:
        quality = "moderate"
    else:
        quality = "poor"

    return (
        f"The dataset quality is {quality}, making it suitable for {domain.lower()} analysis. "
        "Key metrics show mixed performance with areas of stability and risk. "
        "Addressing high-risk columns and volatility will improve decision confidence "
        "and business outcomes."
    )
