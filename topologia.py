import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging

class TopologiaRynku:
    def __init__(self, symbole, data_start, data_koniec):
        self.symbole = symbole
        self.start = data_start
        self.end = data_koniec
        self.dane = None
        self.stopy_zwrotu = pd.DataFrame()
        self.macierz_odl = pd.DataFrame()
        self.G = None
        self.mst_graph = None
        self.logger = logging.getLogger("TopologiaRynku")

    def pobierz_dane(self):
        self.logger.info(f"Pobieranie danych dla {len(self.symbole)} instrumentów z yFinance.")
        try:
            df = yf.download(self.symbole, start=self.start, end=self.end, auto_adjust=True, progress=False)
            if df.empty:
                self.logger.error("Pobrano pusty DataFrame!")
                return None
            
            self.dane = df['Close'].copy() if 'Close' in df else df.copy()
            if self.dane.shape[1] == 0:
                self.logger.error("Brak kolumn z cenami zamknięcia.")
                self.dane = None
                return None
            
            self.logger.info(f"Pobrano dane pomyślnie. Rozmiar macierzy: {self.dane.shape}")
            return self.dane
        except Exception as e:
            self.logger.critical(f"Błąd podczas pobierania danych: {e}")
            return None

    def oblicz_macierz_odleglosci(self):
        if self.dane is None or self.dane.empty:
            self.logger.error("Brak danych")
            return None
        
        self.logger.info("Obliczam logarytmiczne stopy zwrotu i macierz odległości.")
        aktywne_tickery = [t for t in self.symbole if t in self.dane.columns]

        for t in aktywne_tickery:
            self.stopy_zwrotu[t] = np.log(self.dane[t] / self.dane[t].shift(1))

        self.stopy_zwrotu.dropna(inplace=True)
        korelacja = self.stopy_zwrotu.corr()
        self.macierz_odl = np.sqrt(2 * (1 - korelacja))

        return self.macierz_odl

    def buduj_graf(self):
        if self.macierz_odl.empty:
            return None
        self.G = nx.from_pandas_adjacency(self.macierz_odl)
        return self.G

    def wyznacz_mst(self):
        if self.G is None:
            return None
        self.mst_graph = nx.minimum_spanning_tree(self.G)
        self.logger.info("Wyznaczono Minimalne Drzewo Rozpinające (MST).")
        return self.mst_graph

    def _daj_kolor_wezla(self, ticker):
        if not isinstance(ticker, str):
            return 'gray'
        if ticker.endswith('.WA'):
            return 'crimson'
        elif '-USD' in ticker:
            return 'gold'
        elif '=F' in ticker:
            return 'chocolate'
        return 'skyblue'

    def rysuj_graf(self, prog_korelacji=0.2):
        if self.macierz_odl.empty:
            return
        macierz_kor = self.stopy_zwrotu.corr()
        G_rys = nx.Graph()
        tickery = macierz_kor.columns
        grubosc_krawedzi = []

        for i in range(len(tickery)):
            for j in range(i + 1, len(tickery)):
                t1, t2 = tickery[i], tickery[j]
                r = macierz_kor.loc[t1, t2]
                if abs(r) >= prog_korelacji:
                    G_rys.add_edge(t1, t2, weight=abs(r))
                    grubosc_krawedzi.append(abs(r) * 3)

        if not G_rys.nodes():
            return
        plt.figure(figsize=(12, 10))
        kolory = [self._daj_kolor_wezla(n) for n in G_rys.nodes()]
        pos = nx.circular_layout(G_rys)
        nx.draw_networkx_nodes(G_rys, pos, node_size=1200, node_color=kolory, alpha=0.9)
        nx.draw_networkx_labels(G_rys, pos, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(G_rys, pos, width=grubosc_krawedzi, edge_color='gray', alpha=0.4)
        plt.title(f"Graf Rynku (Korelacje > {prog_korelacji})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("graf_rynku.png")
        plt.show()

    def rysuj_mst(self):
        if self.mst_graph is None:
            return
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(self.mst_graph, k=0.6, seed=42, iterations=100)
        stopnie = dict(self.mst_graph.degree())
        max_stopien = max(stopnie.values()) if stopnie else 1
        kolory = [self._daj_kolor_wezla(n) for n in self.mst_graph.nodes()]
        rozmiary = [600 + (stopnie[n] / max_stopien) * 2000 for n in self.mst_graph.nodes()]
        najwazniejszy = max(stopnie, key=stopnie.get)
        obwodki = ['black' if n == najwazniejszy else 'none' for n in self.mst_graph.nodes()]
        nx.draw_networkx_edges(self.mst_graph, pos, edge_color='gainsboro', width=1.5)
        nx.draw_networkx_nodes(self.mst_graph, pos, node_size=rozmiary, node_color=kolory, edgecolors=obwodki, linewidths=2)
        nx.draw_networkx_labels(self.mst_graph, pos, font_size=9)
        plt.title(f"Topologia MST ({self.start} - {self.end})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("graf_mst.png")
        plt.show()