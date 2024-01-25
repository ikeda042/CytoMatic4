import os

def init():
    try:
        os.system("git pull origin main")
    except:
        pass
    os.system("pip install -r app/requirements.txt")


init()

# This demo file is for testing purposes only (Read the Quick Overview section in the README.md file for more information)
from app.main import main
main(
    "app/test_database.db",
    85,
    255,
    img_size=200,
    mode="data_analysis",
    layer_mode="normal",
)
