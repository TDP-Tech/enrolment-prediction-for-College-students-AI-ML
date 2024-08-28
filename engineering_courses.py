import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(0)

# Generate data for 20 engineering programs from 2015 to 2024
years = [f"{year}/{year+1}" for year in range(2015, 2025)]
courses = [
    "DIPLOMA IN ARCHITECTURE",
    "DIPLOMA IN ELECTRICAL AND ELECTRONIC ENGINEERING",
    "DIPLOMA IN LABORATORY SCIENCE AND TECHNOLOGY",
    "DIPLOMA IN ELECTRONICS AND TELECOMMUNICATION ENGINEERING",
    "DIPLOMA IN HIGHWAY ENGINEERING",
    "DIPLOMA IN COMPUTER SCIENCE",
    "DIPLOMA IN MECHATRONICS ENGINEERING",
    "DIPLOMA IN MECHANICAL ENGINEERING",
    "DIPLOMA IN CIVIL ENGINEERING",
    "BACHELOR IN CIVIL ENGINEERING",
    "BACHELOR IN MECHANICAL ENGINEERING",
    "BACHELOR IN COMPUTER ENGINEERING",
    "BACHELOR IN ELECTRICAL ENGINEERING",
    "BACHELOR IN ELECTRONICS AND TELECOMMUNICATION ENGINEERING",
    "BACHELOR IN SOFTWARE ENGINEERING",
    "BACHELOR IN ENVIRONMENTAL ENGINEERING",
    "BACHELOR IN CHEMICAL ENGINEERING",
    "BACHELOR IN BIOMEDICAL ENGINEERING",
    "BACHELOR IN INDUSTRIAL ENGINEERING",
    "BACHELOR IN AEROSPACE ENGINEERING"
]

data = []

for year in years:
    for course in courses:
        capacity = np.random.randint(50, 300)
        selected = np.random.randint(int(capacity * 0.6), capacity)
        registered = np.random.randint(int(selected * 0.2), selected)
        data.append([year, course, selected, capacity, registered])

df = pd.DataFrame(data, columns=["YEAR", "COURSE", "SELECTED", "CAPACITY", "REGISTERED"])

# Save the dataframe to a CSV file
df.to_csv('engineering_programs_data.csv', index=False)
