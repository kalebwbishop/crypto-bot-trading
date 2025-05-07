import os
import time
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

from dotenv import load_dotenv

load_dotenv()

def build_jwt(method, path):
    private_key_pem = os.getenv('COINBASE_PRIVATE_KEY')
    if not private_key_pem:
        raise ValueError("COINBASE_PRIVATE_KEY environment variable is not set")
    
    name = os.getenv('COINBASE_API_KEY_ID') 
    if not name:
        raise ValueError("COINBASE_API_KEY_ID environment variable is not set")
    
    current_time = int(time.time())

    header = {
        "alg": "ES256",
        "typ": "JWT",
        "kid": name,
        "nonce": str(current_time)
    }

    if not path.startswith("https://api.coinbase.com"):
        raise ValueError("Path must start with https://api.coinbase.com")
    
    path = path.replace("https://", "")
    uri = f"{method} {path}"

    data = {
        "iss": "coinbase-cloud",
        "nbf": current_time,
        "exp": current_time + 120,
        "sub": name,
        "uri": uri
    }

    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None,
        backend=default_backend()
    )

    token = jwt.encode(
        data,
        private_key,
        algorithm="ES256",
        headers=header
    )

    return token

if __name__ == "__main__":
    print(build_jwt("GET", "api/v3/brokerage/accounts"))
