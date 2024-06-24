import sqlite3
from cryptography.fernet import Fernet

# Load the encryption key from the database
def get_encryption_key():
    conn = sqlite3.connect('tokens.db')
    cursor = conn.cursor()
    cursor.execute('SELECT key FROM encryption_key WHERE id = 1')
    key = cursor.fetchone()[0]
    conn.close()
    return key

encryption_key = get_encryption_key()
cipher_suite = Fernet(encryption_key)

# Connect to SQLite database
conn = sqlite3.connect('tokens.db')
cursor = conn.cursor()

# Fetch all tokens
cursor.execute('SELECT token FROM tokens')
tokens = cursor.fetchall()
conn.close()

# Print the decrypted tokens
print("Tokens in database:")
for encrypted_token in tokens:
    try:
        decrypted_token = cipher_suite.decrypt(encrypted_token[0].encode()).decode()
        print(decrypted_token)
    except Exception as e:
        print(f"Error decrypting token: {encrypted_token[0]}, error: {e}")
