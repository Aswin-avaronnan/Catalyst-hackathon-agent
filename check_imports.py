import sys
sys.path.insert(0, '/home/aswin/hackathon-agent')
from agent.core import TalentScoutAgent
from agent.matcher import Matcher
from agent.conversation import ConversationSimulator
from agent.candidate_discovery import CandidateDiscovery, Candidate
from api.server import app
print("ALL IMPORTS OK")
