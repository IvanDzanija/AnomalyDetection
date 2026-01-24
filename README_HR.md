# Detekcija Anomalija u Potrošnji Energije Stambenih Objekata

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English Version / Engleska verzija](README.md)

## Sadržaj
- [Pregled](#pregled)
- [Cilj](#cilj)
- [Skup Podataka](#skup-podataka)
- [Metodologija](#metodologija)
- [Pokušane Metode](#pokušane-metode)
- [Izazovi](#izazovi)
- [Rezultati](#rezultati)
- [Budući Napredak](#budući-napredak)
- [Instalacija](#instalacija)
- [Korištenje](#korištenje)
- [Struktura Projekta](#struktura-projekta)
- [Akademski Kontekst](#akademski-kontekst)
- [Reference](#reference)
- [Doprinosi](#doprinosi)
- [Licenca](#licenca)

## Pregled

Ovaj projekt implementira **sustav strojnog učenja bez nadzora** za detekciju anomalija u potrošnji energije stambenih objekata. Sustav analizira podatke o komunalnom obračunu iz stambenih zgrada, fokusirajući se na potrošnju energije za grijanje (ENESGR) i tople vode (ENESTV) kako bi identificirao neobične obrasce koji mogu ukazivati na kvarove sustava, pogreške u naplati ili neobično ponašanje stanara.

Pristup kombinira **segmentaciju temeljenu na grupiranju** s **detekcijom anomalija temeljenoj na rezidualima**, koristeći prednosti nenadziranog učenja za otkrivanje prirodnih grupa u obrascima potrošnje bez potrebe za označenim podacima za treniranje.

## Cilj

**Što predviđamo i zašto?**

Primarni ciljevi ovog projekta su:

1. **Predviđanje mjesečne potrošnje energije** za pojedinačne stanove na temelju:
   - Fizičkih karakteristika (površina, instalirana snaga)
   - Informacija o stanovalnicima (broj osoba)
   - Vremenskih obrazaca (sezonalnost, godina)

2. **Detekcija anomalija u potrošnji** identificiranjem odstupanja od predviđenih vrijednosti koja premašuju statističke pragove (Z-score > 3)

3. **Razumijevanje obrazaca potrošnje** kroz analizu grupiranja radi segmentacije stanova u grupe sa sličnim ponašanjem potrošnje energije

**Zašto je ovo važno?**

- **Energetska Učinkovitost**: Identificiranje abnormalne potrošnje pomaže u otkrivanju neučinkovitih sustava ili gubitka energije
- **Točnost Naplate**: Neobični obrasci mogu ukazivati na kvarove brojila ili pogreške u naplati
- **Prediktivno Održavanje**: Rano otkrivanje anomalija može spriječiti kvarove sustava
- **Planiranje Resursa**: Razumijevanje obrazaca potrošnje pomaže u planiranju kapaciteta i alokaciji resursa
- **Smanjenje Troškova**: Za upravitelje zgrada i stanare kroz rano otkrivanje problema

## Skup Podataka

**Izvor Podataka**: TV_dataset_202511060849.csv - Zapisi o komunalnom obračunu stambenih stanova (2010-2024)

**Ključne Značajke**:
- `ID_STANA`: Jedinstveni identifikator stana
- `POVRSINA`: Površina stana (m²)
- `BR_OSOBA`: Broj stanovnika
- `S_SNAGA`: Instalirana snaga grijanja
- `ENESGR`: Potrošnja energije za grijanje
- `ENESTV`: Potrošnja energije za toplu vodu
- `MJESEC`: Mjesec obračuna
- `GODINA`: Godina obračuna

**Vremenski Raspon**: Mjesečni podaci o naplati koji obuhvaćaju više od 14 godina (2010-2024)

**Izazovi Podataka**:
- Nedostajuće vrijednosti u atributima stana (POVRSINA, BR_OSOBA)
- Rijetki zapisi obračuna za neke stanove
- Mješoviti tipovi podataka koji zahtijevaju pretprocesiranje
- Sezonske varijacije koje zahtijevaju posebno kodiranje

## Metodologija

Projekt koristi **pristup nenadziranog učenja** koji kombinira grupiranje i regresiju za detekciju anomalija:

### 1. Inženjerstvo Značajki
- **Kružno kodiranje** mjesečne sezonalnosti korištenjem sinus/kosinus transformacija za hvatanje cikličkih obrazaca
- Normalizacija numeričkih značajki pomoću StandardScaler
- Obrada nedostajućih podataka kroz imputaciju i filtriranje

### 2. Segmentacija Temeljena na Grupiranju
- **K-Means grupiranje** za grupiranje stanova sa sličnim obrascima potrošnje
- Optimalan odabir klastera korištenjem:
  - **Elbow Method**: Identificiranje točke smanjenih prinosa u redukciji varijance
  - **Silhouette Score**: Mjerenje kohezije i razdvajanja klastera (rezultati: 0.26-0.31)
- Zasebna analiza za tri energetske skupine (1EG, 2EG, 3EG)

### 3. Prediktivni Modeli Specifični za Klastere
- **Linearna regresija** modeli trenirani neovisno za svaki klaster
- Značajke: Površina, broj stanovnika, sezonsko kodiranje, instalirana snaga
- Zasebni modeli poboljšavaju točnost predviđanja uzimajući u obzir različita ponašanja potrošnje

### 4. Detekcija Anomalija
- **Analiza reziduala**: Računanje razlika između stvarne i predviđene potrošnje
- **Z-score prag**: Označavanje anomalija gdje je |Z-score| > 3 (99.7% interval pouzdanosti)
- Vizualizacija normalnih naspram anomalnih podatkovnih točaka

## Pokušane Metode

### Primarne Metode

#### 1. K-Means Grupiranje
**Zašto odabrano**: Nenadzirana metoda idealna za otkrivanje prirodnih grupa bez označenih podataka

**Implementacija**:
- Testirani brojevi klastera: k = 3, 4, 5, 6
- Optimalan odabir k na temelju Silhouette Score
- Rezultati:
  - **1EG**: k=4 (Silhouette: 0.2664)
  - **2EG**: k=5 (Silhouette: 0.3154)
  - **3EG**: k=3-4 (Silhouette: ~0.27)

**Nalazi**: Pojavili su se jasni sezonski i obrasci temeljeni na zauzetosti, s jasnim klasterima zimske/ljetne potrošnje

#### 2. Linearna Regresija (Po Klasteru)
**Zašto odabrano**: Jednostavan, interpretabilan model za predviđanje potrošnje unutar homogenih grupa

**Implementacija**:
- Zasebni modeli za svaki klaster
- Značajke: POVRSINA, BR_OSOBA, MJESEC_sin, MJESEC_cos, S_SNAGA
- Performanse evaluirane putem analize reziduala

**Nalazi**: Modeli specifični za klastere značajno su nadmašili globalnu regresiju zbog heterogenosti ponašanja

#### 3. Statistička Detekcija Anomalija (Z-Score)
**Zašto odabrano**: Standardni statistički pristup za detekciju outliera temeljen na standardnoj devijaciji

**Implementacija**:
- Izračunavanje reziduala: `rezidual = stvarno - predviđeno`
- Računanje Z-scorova kroz reziduala
- Označavanje anomalija gdje je |Z| > 3

**Nalazi**: Uspješno identificirani outlieri uključujući greške brojila, prazne stanove i neobične događaje potrošnje

### Razmatrane Alternativne Metode

#### 4. LOWESS Izglađivanje (Statsmodels)
**Status**: Istraženo za analizu trendova i sezonsku dekompoziciju

**Primjena**: Izglađivanje vremenskih serija potrošnje za identificiranje temeljnih trendova

**Ishod**: Korisno za vizualizaciju ali ne primarna metoda detekcije

## Izazovi

### Problemi Kvalitete Podataka
1. **Nedostajući Metapodaci**: ~15-20% stanova nema informacije o površini ili broju stanovnika
   - **Utjecaj**: Smanjen skup značajki za pogođene stanove
   - **Ublažavanje**: Imputacija korištenjem medijana temeljenih na klasterima

2. **Rijetki Podaci o Naplati**: Nepravilni ciklusi naplate za neke stanove
   - **Utjecaj**: Nepotpune vremenske serije za treniranje
   - **Ublažavanje**: Fokusirana analiza na stanove s dosljednim zapisima

3. **Nekonzistentnost Tipova Podataka**: Mješoviti formati u CSV stupcima
   - **Utjecaj**: DtypeWarnings tijekom učitavanja
   - **Ublažavanje**: Eksplicitna specifikacija dtype tijekom uvoza

### Metodološki Izazovi
1. **Sezonska Složenost**: Snažne sezonske varijacije u potrošnji grijanja
   - **Rješenje**: Kružno kodiranje (sinus/kosinus) za hvatanje mjesečnih cikličkih obrazaca

2. **Validacija Klastera**: Umjereni Silhouette Scores (0.26-0.31)
   - **Interpretacija**: Obrasci potrošnje imaju postupne granice, ne diskretne klastere
   - **Implikacija**: Fuzzy metode grupiranja mogu biti prikladnije

3. **Odabir Praga Anomalije**: Z-score > 3 može biti previše strog ili previše blag ovisno o klasteru
   - **Rješenje**: Razmotreni pragovi specifični za klaster temeljeni na distribucijama reziduala

4. **Važnost Značajki**: Ograničene značajke dostupne za predviđanje
   - **Izazov**: Ne može uhvatiti faktore ponašanja (npr. razdoblja godišnjeg odmora, preferencije termostata)
   - **Utjecaj**: Neke legitimne varijacije u ponašanju označene kao anomalije

## Rezultati

### Performanse Grupiranja
- Uspješno identificirano **3-5 različitih obrazaca potrošnje** po energetskoj skupini
- Klasteri usklađeni s:
  - Sezonskim varijacijama (zimsko grijanje naspram ljetne osnovne razine)
  - Kategorije veličine stana (mali/srednji/veliki)
  - Razine zauzetosti (samac/obitelj/zajednički)

### Točnost Predviđanja
- Linearni modeli specifični za klastere postigli **bolje uklapanje** od globalnih modela
- Analiza reziduala otkrila sustavne obrasce u greškama predviđanja
- Performanse modela varirale po klasteru (R² vrijednosti: 0.45-0.75)

### Detekcija Anomalija
- Identificirane vrste anomalija:
  - **Greške u Naplati**: Vrijednosti potrošnje reda veličine različite od susjeda
  - **Kvarovi Brojila**: Nagli padovi na nulu ili skokovi
  - **Prazni Stanovi**: Produljena gotovo nulta potrošnja
  - **Problemi Sustava**: Odstupanja na razini klastera koja ukazuju na probleme sa zgradnim sustavom

### Uvidi
- **Sezonska Ovisnost**: Potrošnja grijanja 3-5x viša u zimskim mjesecima
- **Korelacija Veličine**: Snažna pozitivna korelacija između veličine stana i potrošnje
- **Učinak Zauzetosti**: Broj stanovnika pokazuje slabu korelaciju od očekivane
- **Ocjena Snage**: Instalirana snaga (S_SNAGA) korelira s varijabilnošću potrošnje

## Budući Napredak

### Kratkoročna Poboljšanja
1. **Alternativne Metode Grupiranja**:
   - **DBSCAN**: Grupiranje temeljeno na gustoći za rukovanje nepravilnim oblicima klastera i identifikaciju šuma
   - **Gaussian Mixture Models (GMM)**: Probabilističko grupiranje s mekim dodjelama
   - **Hijerarhijsko Grupiranje**: Istraživanje ugniježđenih obrazaca potrošnje

2. **Napredni Regresijski Modeli**:
   - **Random Forest**: Nelinearni obrasci i interakcije značajki
   - **Gradient Boosting (XGBoost/LightGBM)**: Poboljšana točnost predviđanja
   - **Polinomska Regresija**: Hvatanje nelinearnih odnosa

3. **Metode Vremenskih Serija**:
   - **ARIMA/SARIMA**: Uključivanje vremenske autokorelacije
   - **Prophet**: Dekompozicija trenda, sezonalnosti i praznika
   - **LSTM/GRU**: Duboko učenje za složene vremenske obrasce

4. **Ensemble Detekcija Anomalija**:
   - **Isolation Forest**: Detekcija anomalija temeljena na stablu
   - **Local Outlier Factor (LOF)**: Detekcija anomalija temeljena na gustoći
   - **Autoencoders**: Greška rekonstrukcije temeljena na neuronskoj mreži
   - **One-Class SVM**: Učenje granica temeljeno na potpornim vektorima

### Srednjoročni Pravci Istraživanja
1. **Inženjerstvo Značajki**:
   - Integracija vremenskih podataka (vanjska temperatura, stupnjevi grijanja)
   - Karakteristike zgrade (godina izgradnje, kvaliteta izolacije)
   - Rasporedi zauzetosti (rad od kuće, razdoblja godišnjeg odmora)
   - Ekonomski pokazatelji (cijene energije, sezonska prilagodba naplate)

2. **Multi-Modalna Analiza**:
   - Zajedničko modeliranje grijanja (ENESGR) i tople vode (ENESTV)
   - Međustanova komparativna analiza
   - Agregacijski obrasci na razini zgrade

3. **Objašnjivost**:
   - SHAP vrijednosti za važnost značajki
   - LIME za lokalnu interpretabilnost
   - Kontrafaktualna objašnjenja za anomalije

4. **Okvir za Validaciju**:
   - Stručno označavanje poznatih anomalija
   - Metrike preciznosti/odziva
   - ROC krivulje i AUC analiza

### Dugoročna Vizija
1. **Praćenje u Stvarnom Vremenu**: Procesiranje streaming podataka i online detekcija anomalija
2. **Prediktivna Upozorenja**: Sustav ranog upozoravanja za potencijalne kvarove sustava
3. **Automatizirana Dijagnoza**: Klasifikacija tipova anomalija s analizom temeljnih uzroka
4. **Sustav Optimizacije**: Preporuke za poboljšanje energetske učinkovitosti
5. **Web Nadzorna Ploča**: Interaktivna platforma za vizualizaciju i izvještavanje

## Instalacija

### Preduvjeti
- Python 3.12 ili viši
- pip ili uv upravitelj paketa

### Postavljanje

1. **Klonirajte repozitorij**:
```bash
git clone https://github.com/IvanDzanija/AnomalyDetection.git
cd AnomalyDetection
```

2. **Stvorite virtualno okruženje** (preporučeno):
```bash
python -m venv .venv
source .venv/bin/activate  # Na Windows: .venv\Scripts\activate
```

3. **Instalirajte ovisnosti**:

Korištenje pip:
```bash
pip install -r requirements.txt
```

Ili korištenje uv:
```bash
uv sync
```

### Ovisnosti
- `jupyter>=1.1.1` - Jupyter bilježnice
- `matplotlib>=3.10.7` - Crtanje i vizualizacija
- `numpy>=2.3.4` - Numeričke operacije
- `scikit-learn>=1.8.0` - Algoritmi strojnog učenja
- `scipy>=1.16.3` - Statističke funkcije
- `seaborn>=0.13.2` - Statistička vizualizacija
- `statsmodels>=0.14.6` - Statističko modeliranje

## Korištenje

### Pokretanje Analize

1. **Pokrenite Jupyter**:
```bash
jupyter lab
```

2. **Glavne Bilježnice**:
   - `nova_biljeznica.ipynb`: Potpuni cjevovod analize
   - `notebooks/Grupiranje.ipynb`: Sveobuhvatna analiza grupiranja
   - `notebooks/1EG.ipynb`, `2EG.ipynb`, `3EG.ipynb`: Analize specifične za energetske skupine

### Tijek Rada

```python
# 1. Učitavanje i pretprocesiranje podataka
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/TV_dataset_202511060849.csv')

# 2. Inženjerstvo značajki
data['MJESEC_sin'] = np.sin(2 * np.pi * data['MJESEC'] / 12)
data['MJESEC_cos'] = np.cos(2 * np.pi * data['MJESEC'] / 12)

# 3. Grupiranje
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# 4. Regresija specifična za klaster
from sklearn.linear_model import LinearRegression
for cluster_id in range(4):
    cluster_data = data[clusters == cluster_id]
    model = LinearRegression()
    model.fit(X_train, y_train)
    
# 5. Detekcija anomalija
from scipy.stats import zscore
residuals = actual - predicted
z_scores = zscore(residuals)
anomalies = np.abs(z_scores) > 3
```

## Struktura Projekta

```
AnomalyDetection/
│
├── README.md                    # Engleska dokumentacija
├── README_HR.md                 # Hrvatska dokumentacija (ova datoteka)
├── pyproject.toml               # Konfiguracija projekta
├── uv.lock                      # Datoteka zaključavanja ovisnosti
│
├── notebooks/                   # Jupyter bilježnice
│   ├── 1EG.ipynb               # Analiza Energetske Skupine 1
│   ├── 2EG.ipynb               # Analiza Energetske Skupine 2
│   ├── 3EG.ipynb               # Analiza Energetske Skupine 3
│   └── Grupiranje.ipynb        # Sveobuhvatno grupiranje
│
├── nova_biljeznica.ipynb       # Glavna analitička bilježnica
│
└── docs/                        # Dokumentacija
    ├── assets/                  # Slike i figure
    └── plan_projekta.docx      # Plan projekta
```

## Akademski Kontekst

Ovaj projekt razvijen je kao dio akademskog istraživanja u **primjenama strojnog učenja za energetske sustave**. Rad demonstrira praktičnu primjenu tehnika nenadziranog učenja na stvarne podatke o komunalijama, rješavajući izazove uobičajene u domenama pametnih zgrada i IoT-a.

### Ključni Doprinosi
1. **Hibridni pristup** koji kombinira grupiranje i regresiju za poboljšanu detekciju anomalija
2. **Kružno kodiranje** sezonskih obrazaca za bolju reprezentaciju vremenskih serija
3. **Modeli specifični za klastere** prepoznajući heterogenost u ponašanjima potrošnje
4. **Praktična validacija** na više od 14 godina stvarnih stambenih podataka o komunalijama

### Istraživačka Pitanja na Koja Je Odgovoreno
- Može li nenadzirano učenje učinkovito segmentirati potrošače energije u stambenim objektima?
- Kako se mogu integrirati sezonski obrasci u okvire detekcije anomalija?
- Što je trade-off između složenosti modela i interpretabilnosti u detekciji anomalija u komunalijama?
- Kako se modeli specifični za klastere uspoređuju s globalnim pristupima?

## Reference

### Strojno Učenje i Grupiranje
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*.
- Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*.

### Detekcija Anomalija
- Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly detection: A survey." *ACM Computing Surveys*, 41(3), 1-58.
- Hodge, V., & Austin, J. (2004). "A survey of outlier detection methodologies." *Artificial Intelligence Review*, 22(2), 85-126.

### Energetski Sustavi i Pametne Zgrade
- Ahmad, T., et al. (2018). "A review on renewable energy and electricity requirement forecasting models for smart grid and buildings." *Sustainable Cities and Society*, 55, 102052.
- Seem, J. E. (2007). "Using intelligent data analysis to detect abnormal energy consumption in buildings." *Energy and Buildings*, 39(1), 52-58.

### Vremenske Serije i Predviđanje
- Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: forecasting and control* (5. izdanje). John Wiley & Sons.
- Cleveland, W. S. (1979). "Robust locally weighted regression and smoothing scatterplots." *Journal of the American Statistical Association*, 74(368), 829-836.

## Doprinosi

Doprinosi su dobrodošli! Slobodno podnesite Pull Request. Za veće izmjene, molimo prvo otvorite issue kako bi se raspravljalo što želite promijeniti.

### Smjernice za Razvoj
1. Slijedite PEP 8 stilske smjernice za Python kod
2. Dokumentirajte sve funkcije i metode
3. Dodajte testove za nove značajke
4. Po potrebi ažurirajte dokumentaciju

## Licenca

Ovaj projekt licenciran je pod MIT licencom - pogledajte LICENSE datoteku za detalje.

---

**Kontakt**: Za pitanja ili mogućnosti suradnje, molimo otvorite issue na GitHub-u.

**Zadnje Ažuriranje**: Siječanj 2025
