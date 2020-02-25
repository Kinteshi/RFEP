import onedrivesdk
from onedrivesdk.helpers import GetAuthCodeServer

redirect_uri = 'http://localhost:8080/'
client_secret = 'your_app_secret'
scopes=['wl.signin', 'wl.offline_access', 'onedrive.readwrite']

client = onedrivesdk.get_default_client(
    client_id='your_client_id', scopes=scopes)

auth_url = client.auth_provider.get_auth_url(redirect_uri)

#this will block until we have the code
code = GetAuthCodeServer.get_auth_code(auth_url, redirect_uri)

client.auth_provider.authenticate(code, redirect_uri, client_secret)