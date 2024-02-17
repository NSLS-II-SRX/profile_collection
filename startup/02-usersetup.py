print(f"Loading {__file__}...")

import os
import shutil
import time as ttime

import httpx

### Proposal information put into the metadata
# proposal_num = None
# proposal_title = None
# PI_lastname = None
# saf_num = None

# proposal_num = 312933
data_session = RE.md["data_session"]
proposal_num = data_session.split("-")[-1]

# proposal_title = 'SRX Beamline Commissioning'
# PI_lastname = 'Kiss'
# saf_num = 311284

nslsii_api_client = httpx.Client(base_url="https://api-staging.nsls2.bnl.gov")

proposal_response = nslsii_api_client.get(f"/proposal/{proposal_num}")
proposal_response.raise_for_status()
proposal = proposal_response.json()

PI_lastname = "whoami"
for user in proposal["users"]:
    if user["is_pi"]:
        PI_lastname = user["last_name"]
        break

saf_num = proposal["safs"][0]["saf_id"]

md_proposal = {
    "proposal_num": proposal_num,
    "proposal_title": proposal["title"],
    "PI_lastname": PI_lastname,
    "saf_num": saf_num,
    "cycle": proposal["cycles"][-1],
}

# proposal_num = 313507
# proposal_title = 'Data security at SRX'
# PI_lastname = 'Kiss'
# saf_num = 312779


cycle = "2024_cycle1"

# Set user data in bluesky
# RE.md['data_session'] = data_session
RE.md["proposal"] = md_proposal
RE.md["cycle"] = proposal["cycles"][-1]
