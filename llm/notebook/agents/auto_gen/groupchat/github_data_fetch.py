# filename: github_data_fetch.py
import requests
from datetime import datetime, timedelta
import csv

# Function to fetch the number of issues and pull requests for a given date range
def fetch_data(repo, start_date, end_date):
    # Endpoint for issues and pull requests
    base_url = f"https://api.github.com/repos/{repo}"
    issues_url = f"{base_url}/issues"
    pulls_url = f"{base_url}/pulls"

    # Parameters for filtering by date range and state
    params = {
        'since': start_date.isoformat(),
        'state': 'all',
    }

    # Fetch issues and pull requests
    issues_response = requests.get(issues_url, params=params)
    pulls_response = requests.get(pulls_url, params=params)

    issues = issues_response.json()
    pulls = pulls_response.json()

    return issues, pulls

def filter_data_by_week(issues, pulls, weeks):
    # Dictionary to hold week data
    data_by_week = {week: {'issues': 0, 'pull_requests': 0} for week in weeks}

    # Function to determine the week key for a given date
    def get_week_key(date, weeks):
        for week in weeks:
            if week[0] <= date <= week[1]:
                return week
        return None

    # Count issues and pull requests per week
    for issue in issues:
        created_at = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        week_key = get_week_key(created_at, weeks)
        if week_key:
            data_by_week[week_key]['issues'] += 1

    for pr in pulls:
        created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        week_key = get_week_key(created_at, weeks)
        if week_key:
            data_by_week[week_key]['pull_requests'] += 1

    return data_by_week

def main():
    # Constants
    repo = 'microsoft/autogen'
    today = datetime.now()
    three_weeks_ago = today - timedelta(weeks=3)
    
    # Create a list of week ranges
    weeks = [(three_weeks_ago + timedelta(weeks=i), three_weeks_ago + timedelta(weeks=i+1)-timedelta(seconds=1)) for i in range(3)]

    # Fetch data
    issues, pulls = fetch_data(repo, three_weeks_ago, today)
    
    # Filter the data by week
    data_by_week = filter_data_by_week(issues, pulls, weeks)

    # CSV output
    csv_filename = 'github_data.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Week', 'Issues', 'Pull Requests'])
        
        for week, data in data_by_week.items():
            start_date = week[0].strftime('%Y-%m-%d')
            end_date = week[1].strftime('%Y-%m-%d')
            writer.writerow([f"{start_date} to {end_date}", data['issues'], data['pull_requests']])

main()