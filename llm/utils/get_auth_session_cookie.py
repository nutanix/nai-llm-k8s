"""
Script to get auth session cookie

Usage:
python get_auth_session_cookie.py --email <EMAIL> --password <PASSWORD>
"""
import argparse
import os
import requests
import sys


def get_auth_session_cookie(email: str, password: str) -> str:
    """
    Function to get authservice_session cookie from dex idp.
    It requires INGRESS_HOST and INGRESS_HOST to be set as env variables.
    """
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
    return session.cookies.get_dict()["authservice_session"]


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to get auth session cookie.')

    # Add arguments
    parser.add_argument('--email', type=str, help='user email', required=True)
    parser.add_argument('--password', type=str, help='user password', required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    session_cookie = get_auth_session_cookie(args.email, args.password)
    print(session_cookie)
