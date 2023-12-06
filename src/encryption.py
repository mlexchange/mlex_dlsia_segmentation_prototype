from cryptography.fernet import Fernet


def encrypt(
        message: bytes,
        key: bytes
        ) -> bytes:
    return Fernet(key).encrypt(message)


def decrypt(
        token: bytes,
        key: bytes
        ) -> bytes:
    return Fernet(key).decrypt(token)