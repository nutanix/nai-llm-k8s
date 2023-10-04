"""
Script to get auth session cookie
"""
import argparse
import os
import requests
import sys


def get_auth_session_cookie(email: str, password: str):
    ingress_host = os.getenv("INGRESS_HOST")
    if not ingress_host:
        print("INGRESS_HOST env variable is not set")
        sys.exit(1)

    ingress_port = os.getenv("INGRESS_PORT")
    if not ingress_host:
        print("INGRESS_PORT env variable is not set")
        sys.exit(1)

    HOST = f"http://{ingress_host}:{ingress_port}/"
    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"login": email, "password": password}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]
    print(session_cookie)


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to get auth session cookie.')

    # Add arguments
    parser.add_argument('--email', type=str, help='name of the deployment')
    parser.add_argument('--password', type=str, help='name of the deployment')

    # Parse the command-line arguments
    args = parser.parse_args()

    if not args.email or not args.password:
        print("Email/Password not provided")
        sys.exit(1)

    get_auth_session_cookie(args.email, args.password)
