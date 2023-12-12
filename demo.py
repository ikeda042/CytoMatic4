from main import main
import subprocess
def install_requirements(filename='requirements.txt') -> None:
    with open(filename, 'r') as file:
        packages = [line.strip() for line in file if line.strip()]
    for package in packages:
        subprocess.run(['pip', 'install', package])

install_requirements()
#This demo file is for testing purposes only (Read the Quick Overview section in the README.md file for more information)
main("test_database.db", 85, 255, img_size = 200, mode="data_analysis", layer_mode="normal")
