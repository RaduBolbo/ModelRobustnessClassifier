# ModelRobustnessClassifier

## Linux setup

To setup the project run the following commands:

```bash
git clone https://github.com/RaduBolbo/ModelRobustnessClassifier.git
cd ModelRobustnessClassifier/
```

The script setup.sh does the following:
- Installs python3, pip, kaggle and python venv if it does not exist
- Downloads the dataset from kaggle (one needs to set the username and API key for this to work. Go to kaggle at Settings -> API -> Create New Token. This will download a json file where you can find the username and key. Then go to setup.sh and set the Kaggle credentials)
- Creates checkpoints directory -> model checkpoints can be found here (to be added)
- Creates python virtual environment and installs the packages from *requirements.txt*

To run the script:
```bash
bash setup.sh
```

Activate the virtual environment:
```bash
source myenv/bin/activate
```

To train the baseline model:
```bash
python src/train_loop_baseline.py 
```