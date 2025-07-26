# Tiny GPT

### Installation 
```
python3 -m venv env; source env/bin/activate 
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio

python -m pip install requests tiktoken numpy

# Go to shakespear folder to prepare data
cd shakespear
python shakespear.py

# Then, start training
cd ..
python transfomer.py
```
