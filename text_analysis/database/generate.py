from data import abbrevations

def pattern_generator():
    patterns = "patterns = [\n"
    for company in abbrevations:
        patterns+="{\"label\": \"ORG\", \"pattern\": " + company.lower() + "},\n"
        patterns+="{\"label\": \"ORG\", \"pattern\": " + abbrevations[company].lower() + "},\n"
    patterns += "]"