import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('tokens.db')

# Create a cursor object
cursor = conn.cursor()

# Fetch all tokens
cursor.execute('SELECT token FROM tokens')
tokens = cursor.fetchall()

# Close the connection
conn.close()

# Print the tokens
print("Tokens in database:")
for token in tokens:
    print(token[0])
