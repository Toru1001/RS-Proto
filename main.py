import os
from pathlib import Path
from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_path = Path(__file__).parent / "src" / "app.py"
    
    # Set the script path for Streamlit
    os.sys.argv = ["streamlit", "run", str(app_path)]
    
    # Run streamlit
    os.sys.exit(stcli.main())