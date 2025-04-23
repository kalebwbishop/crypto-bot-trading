import time
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def build_jwt(method, path):
    private_key_pem = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIK+lDAB7pXdghQxse5G28KShs8UMivLNcf+CVffcVZx4oAoGCCqGSM49
AwEHoUQDQgAETls6JQJ7EsokIG8HxItAreK2SpRwKgmepFYRn+6gA2LoGBSyFITS
fdeWToG/CWHBIZbqasyNxUKPCtRov7O64A==
-----END EC PRIVATE KEY-----"""

    name = "organizations/b7d428cd-41bb-466d-8351-33c4d83b08c6/apiKeys/9bb60e8c-6ed5-4ba1-95c4-34f5b2406aca"
    current_time = int(time.time())

    header = {
        "alg": "ES256",
        "typ": "JWT",
        "kid": name,
        "nonce": str(current_time)
    }

    host = "api.coinbase.com"
    uri = f"{method} {host}/{path}"

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
