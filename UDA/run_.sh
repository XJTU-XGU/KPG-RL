#!/usr/bin/env bash
python train_KPG-RL.py --source Art --target Clipart
python train_KPG-RL.py --source Art --target Product
python train_KPG-RL.py --source Art --target Real_World
python train_KPG-RL.py --source Clipart --target Art
python train_KPG-RL.py --source Clipart --target Product
python train_KPG-RL.py --source Clipart --target Real_World
python train_KPG-RL.py --source Product --target Art
python train_KPG-RL.py --source Product --target Clipart
python train_KPG-RL.py --source Product --target Real_World
python train_KPG-RL.py --source Real_World --target Art
python train_KPG-RL.py --source Real_World --target Clipart
python train_KPG-RL.py --source Real_World --target Product
