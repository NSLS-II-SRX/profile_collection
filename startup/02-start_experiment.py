# /// script
# dependencies = ["ldap3", "nslsii", "httpx"]
# ///

from ldap3 import Server, Connection, NTLM
from ldap3.core.exceptions import LDAPInvalidCredentialsResult

import re
import httpx
import warnings
from getpass import getpass

data_session_re = re.compile(r"^pass-(?P<proposal_number>\d+)$")

nslsii_api_client = httpx.Client(
    base_url="https://api.nsls2.bnl.gov"
)

def validate_proposal(data_session_value):

    data_session_match = data_session_re.match(data_session_value)
    
    if data_session_match is None:
        raise ValueError(
            f"RE.md['data_session']='{data_session_value}' "
            f"is not matched by regular expression '{data_session_re.pattern}'"
        )

    try:
        proposal_number = data_session_match.group("proposal_number")
        
        proposal_response = nslsii_api_client.get(
            f"/v1/proposal/{proposal_number}"
        )
        proposal_response.raise_for_status()
        if "error_message" in proposal_response.json():
            raise ValueError(
                f"while verifying data_session '{data_session_value}' "
                f"an error was returned by {proposal_response.url}: "
                f"{proposal_response.json()}"
            )
        else:
            # data_session is valid!
            pass

    except httpx.RequestError as rerr:
        # give the user a warning
        # but allow the run to start
        warnings.warn(
            f"while verifying data_session '{data_session_value}' "
            f"the request {rerr.request.url!r} failed with "
            f"'{rerr}'"
        )


def md_validator(md):
    
    if "data_session" in md:
        validate_proposal(md["data_session"])



def authenticate(username):

    auth_server = Server('dc2.bnl.gov', use_ssl=True)

    try:
        connection = Connection(
            auth_server, user=f'BNL\\{username}',
            password=getpass("Password : "), authentication=NTLM,
            auto_bind=True, raise_exceptions=True)    
        print(f'\nAuthenticated as : {connection.extend.standard.who_am_i()}')

    except LDAPInvalidCredentialsResult:
        raise RuntimeError(f"Invalid credentials for user '{username}'.") from None


def prove_who_you_are():
    
    username = input("username: ")
    authenticate(username)
    
    return username
    
    
def should_they_be_here(username, new_data_session, beamline):
    
    user_access_json = nslsii_api_client.get(f"/v1/data_session/{username}").json()
    
    if "nsls2" in user_access_json["facility_all_access"]:
        return True
    
    elif beamline in user_access_json["beamline_all_access"]:
        return True
    
    elif new_data_session in user_access_json["data_sessions"]:
        return True
    
    return False


class AuthorizationError(Exception):
    ...
        
        
def _start_experiment(RE, proposal_number, beamline):
    
    new_data_session = f"pass-{proposal_number}"
    
    if new_data_session != RE.md.get("data_session"):
        
        validate_proposal(new_data_session)
        
        username = prove_who_you_are()
        ok = should_they_be_here(username, new_data_session, beamline)
        
        if not ok:
            raise AuthorizationError(
                        f"User '{username}' is not allowed to take data "
                        f"on proposal {new_data_session}"
                    )
        else:
            print("Successfully stared experiment!")
            
    else:
        print("Experiment already started!") 
    
    RE.md["data_session"] = f"pass-{proposal_number}"

    return RE
    
def start_experiment(proposal_number, RE=RE, beamline='srx')
    return _start_experiment(RE, proposal_number, beamline)

# e.g. start_experiment(RE, proposal_number=314062, beamline="chx")
