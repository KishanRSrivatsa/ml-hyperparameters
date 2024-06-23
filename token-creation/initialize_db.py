import sqlite3
import secrets
import string

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('tokens.db')

# Create a cursor object
cursor = conn.cursor()

# Create a table to store tokens
cursor.execute('''
CREATE TABLE IF NOT EXISTS tokens (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL UNIQUE
)
''')

# Function to generate a random token
def generate_token(length=12):
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

# Generate and insert 20 unique tokens
tokens = [generate_token() for _ in range(20)]

for token in tokens:
    cursor.execute('''
    INSERT OR IGNORE INTO tokens (token) VALUES (?)
    ''', (token,))

# Commit the changes and close the connection
conn.commit()
conn.close()

# Print generated tokens
print("Generated tokens:")
for token in tokens:
    print(token)
