from cryptography.fernet import Fernet
import sqlite3

# Function to fetch the encryption key from the database
def load_key_from_db():
    conn = sqlite3.connect('token-creation/tokens.db')
    cursor = conn.cursor()
    cursor.execute("SELECT key FROM tokens WHERE id = 1")  # Assuming there is only one key
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0].encode()  # Convert string to bytes
    else:
        raise ValueError("Encryption key not found in database")

key = load_key_from_db()
cipher_suite = Fernet(key)

def encrypt_token(token):
    encrypted_token = cipher_suite.encrypt(token.encode('utf-8'))
    return encrypted_token

def decrypt_token(encrypted_token):
    decrypted_token = cipher_suite.decrypt(encrypted_token)
    return decrypted_token.decode('utf-8')
