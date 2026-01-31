def generate_next_actions(domain, health_score):
    actions = []

    if health_score < 50:
        actions.append("Prioritize data cleaning before making strategic decisions.")
    elif health_score < 80:
        actions.append("Address high-risk columns to improve insight reliability.")
    else:
        actions.append("Dataset is reliable for advanced analytics and modeling.")

    if domain == "Retail":
        actions.append("Analyze sales volatility and assess promotion dependency.")
        actions.append("Segment customers based on stable revenue drivers.")

    elif domain == "Finance":
        actions.append("Review high-risk metrics and apply risk controls.")
        actions.append("Stress-test volatile financial indicators.")

    elif domain == "Marketing":
        actions.append("Optimize underperforming campaigns.")
        actions.append("Reallocate budget toward improving conversion metrics.")

    else:
        actions.append("Explore deeper feature relationships and trends.")

    return actions
