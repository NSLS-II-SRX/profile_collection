print(f"Loading {__file__}...")

import os
import shutil
import time as ttime

import httpx

# proposal_num = 312933
data_session = RE.md["data_session"]
proposal_num = data_session.split("-")[-1]

# proposal_title = 'SRX Beamline Commissioning'
# PI_lastname = 'Kiss'
# saf_num = 311284

nslsii_api_client = httpx.Client(base_url="https://api.nsls2.bnl.gov")

proposal_response = nslsii_api_client.get(f"/v1/proposal/{proposal_num}")
proposal_response.raise_for_status()
proposal = proposal_response.json()["proposal"]

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


cycle = "2024_cycle2"


RE.md["proposal"] = md_proposal
RE.md["cycle"] = proposal["cycles"][-1]
# RE.md["cycle"] = "2024-2"

cycle = RE.md["cycle"]
cycle = cycle.replace("-", "_cycle")
RE.md["proposal"]["cycle"] = "2024_cycle2"
