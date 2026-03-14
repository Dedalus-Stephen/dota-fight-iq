import sys
import gevent
from gevent import monkey
monkey.patch_all()

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(message)s')

from steam.client import SteamClient
from dota2.client import Dota2Client

STEAM_USER = "dotafightiq"
STEAM_PASS = "828588ss"
MATCH_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 8726201423

client = SteamClient()
dota = Dota2Client(client)

@client.on("error")
def on_error(result):
    print(f"Steam error: {result}")

@client.on("channel_secured")
def on_channel_secured():
    print("Channel secured")

@client.on("connected")
def on_connected():
    print("Connected to Steam")

@client.on("logged_on")
def on_logged_on():
    print(f"Logged into Steam as {client.user.name}")
    dota.launch()

@client.on("disconnected")
def on_disconnected():
    print("Disconnected from Steam")

@dota.on("ready")
def on_dota_ready():
    print(f"Dota 2 GC ready — requesting match {MATCH_ID}...")
    dota.request_match_details(MATCH_ID)

@dota.on("match_details")
def on_match_details(match_id, eresult, match):
    if eresult != 1:
        print(f"ERROR: GC returned eresult={eresult}")
    else:
        print(f"\nMatch:       {match_id}")
        print(f"Cluster:     {match.cluster}")
        print(f"Replay Salt: {match.replay_salt}")
        if match.cluster and match.replay_salt:
            print(f"Replay URL:  http://replay{match.cluster}.valve.net/570/{match_id}_{match.replay_salt}.dem.bz2")
    client.disconnect()

print("Connecting to Steam...")
result = client.login(STEAM_USER, STEAM_PASS)
print(f"Login result: {result}")
client.run_forever()