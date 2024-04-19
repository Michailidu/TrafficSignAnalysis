# Analýza dopravních značek

## Spuštění

```bash
git clone https://github.com/Michailidu/TrafficSignAnalysis.git
cd TrafficSignAnalysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Data by měla být umístěna do složky `TrafficSignAnalysis/datasets/data`.


Pro změnu typu modelu a batch size je možné použít parametry:
```bash
python main.py --model-type <model_type> --batch-size <batch_size>
```

Defaultní hodnoty jsou `yolov9c.pt` a `16`.

## Výstup
Výstupem by měl být soubor `time.txt` a složka pro každý běh ve formátu `config_<velikost_datasetu>_<počet_epoch>_<velikost_obrázku>`. 