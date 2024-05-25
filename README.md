# Analýza dopravních značek
Implementace k semestrálnímu projektu Analýza dopravních značek za pomocí obrazů.

K řešení jsou použity modely YOLO pomocí balíčku Ultralytics. Vlastní kódy jsou inspirovány kódy uvedenými v [dokumentaci balíčku](https://docs.ultralytics.com/).

## Nastavení prostředí

```bash
git clone https://github.com/Michailidu/TrafficSignAnalysis.git
cd TrafficSignAnalysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data by měla být umístěna do složky `TrafficSignAnalysis/datasets/data`.

## Trénování modelu
Trénování je možné provést spuštěním skriptu `train.py` s následujícími parametry (výchozí hodnoty jsou uvedeny v závorkách):
* `config-path` - cesta k .yaml souboru s popisem datasetu
* `model-type` - typ modelu (yolov8n.pt)
* `batch-size` - velikost dávky (8)
* `epochs` - počet epoch (30)
* `img-size` - velikost obrázků (640)
* `name` - název běhu (run)

```bash
python train.py --help
```

Jako .yaml soubory s popisem datasetu lze použít soubory z adresáře `TrafficSignAnalysis/datasets`. 
V případě změny adresářové struktury je nutné upravit cesty.

## Testování modelu
Pro určení přesnosti byl vytvořen Jupyter notebook `test.ipynb`, který umožňuje zjištění přesnosti modelu na testovacím datasetu. Jako popisný .yaml soubor je použit `TrafficSignAnalysis/datasets/config_test.yaml`.