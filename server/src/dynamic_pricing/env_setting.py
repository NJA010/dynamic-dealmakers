from google.cloud import secretmanager
import os
import json

def get_secret(project_id: str, secret_id: str, version_id: str):
    """
    Get information about the given secret. This only returns metadata about
    the secret container, not any secret material.
    """

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # Get the secret.
    response = client.access_secret_version(request={"name": name})

    # Print the secret payload.
    payload = response.payload.data.decode("UTF-8")

    # Print data about the secret.
    return json.loads(payload)

def define_app_creds():

    DEBUG = os.getenv('DEBUG', True) in ['true', 'True', True]
    PROJECT_ID = os.getenv('PROJECT_ID')
    SECRET_ID_SA = os.getenv('SECRET_ID_SA')
    VERSION_ID = os.getenv('VERSION_ID')
    print(f'{DEBUG=}. If true, running locally, else in a container.')
    if not DEBUG:
        config = get_secret(
            PROJECT_ID,
            SECRET_ID_SA,
            VERSION_ID,
        )
        with open('/app/config.json', 'w') as out_file:
            json.dump(config, out_file)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/config.json'
    else:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pricing-prd-11719402-69eaf79e6222.json'

