from cryptography.fernet import Fernet

# Generate a key for encryption/decryption
# You must save this key securely and use it consistently across client and server
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_token(token):
    encrypted_token = cipher_suite.encrypt(token.encode('utf-8'))
    return encrypted_token

def decrypt_token(encrypted_token):
    decrypted_token = cipher_suite.decrypt(encrypted_token)
    return decrypted_token.decode('utf-8')
