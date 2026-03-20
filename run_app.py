import asyncio
import sys

# Python 3.14 changed the default asyncio event loop policy and get_event_loop behaviour.
# We must explicitly set up a loop for the main thread before importing Streamlit.
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

import streamlit.web.cli as stcli
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv() # Load the .env file containing the GROQ API Key
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
