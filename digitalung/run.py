"""Run streamlit application

Necessary script for using run command with streamlit applications
"""

import sys

from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "digitalung.py"]
    sys.exit(stcli.main())
