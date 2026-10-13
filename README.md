# Samsung Final

Восстановление повреждённых 3D-сеток сосудов на данных RBWH. Сравнение L1, L2 и SeMIGCN с исходной геометрией.

## Установка

```bash
git clone https://github.com/fffrrreeekkss/Samsung-Final.git
cd Samsung-Final
python3 -m venv .venv-experiment
.venv-experiment/bin/python -m pip install -r requirements.txt
```

## Запуск

Один пациент:

```bash
.venv-experiment/bin/python -c 'from benchmark.experiment import run; run([1], severities=("medium",), methods=("L1", "L2", "SeMIGCN"))'
```

Все 20 пациентов:

```bash
.venv-experiment/bin/python -c 'from benchmark.experiment import run; run(list(range(1, 21)), severities=("medium",), methods=("L1", "L2", "SeMIGCN"))'
```

Результаты: `experiments/rbwh_results/metrics.csv`.

