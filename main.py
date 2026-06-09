"""
Plik uruchomieniowy projektu zaliczeniowego z przedmiotu Python - analiza topologii rynku.

Autorzy:
    Kamil Czyżowicz
    Dawid Szymulewicz
"""

import logging
from topologia import TopologiaRynku
from config import TICKERS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

def main():
    bt = TopologiaRynku(TICKERS, '2025-01-01', '2025-12-31')
    
    bt.pobierz_dane()
    bt.oblicz_macierz_odleglosci()
    bt.buduj_graf()
    bt.wyznacz_mst()

    bt.rysuj_graf()
    bt.rysuj_mst()

if __name__ == "__main__":
    main()