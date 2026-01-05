# Izveštaj o projektu — Zadatak 1 i Zadatak 2

## Pregled

Kratak rezime implementiranih rešenja:

* **Zadatak 1:** Učenje parametara SVM modela () sa linearnim jezgrom (kernelom) koristeći **primalnu formulaciju**.
  Model je obučen, izvršen je odabir hiperparametra (), a rezultati su vizuelno prikazani.
* **Zadatak 2:** Učenje dualnih parametara SVM-a () sa nelinearnim jezgrom (**RBF**). Korišćen je dualni QP rešavač (
  kvadratno programiranje), dok su parametri jezgra i konstanta odabrani putem pretrage po mreži (grid search).
  Rezultati su detaljno vizuelizovani.

**Ulazne tačke koda:** `src/task-1.py`, `src/task-2.py`.

**Učitavanje podataka:** `src/shared.py` (učitava datoteku `svmData.csv`).

### Zadatak 1 — Linearni SVM (primalni problem)

Sažeti prikaz realizacije:

* Rešen je primalni problem SVM-a kako bi se dobili parametri i  (implementirano pomoću odabranog optimizatora).
* Sprovedena je unakrsna validacija (cross-validation) i pretraga po mreži za različite vrednosti parametra kako bi se
  odredila optimalna kazna za *hinge* gubitak.
* Vizuelizovani su: trening skup, granica razdvajanja (decision boundary), linije margina, dok su potporni vektori (
  support vectors) posebno istaknuti.
* Anotirani su *hinge* gubici za nekoliko potpornih vektora i iscrtan je grafik zavisnosti prosečnog *hinge* gubitka od
  parametra .

**Grafici:**

* Trening tačke, granica razdvajanja (isprekidana linija), zaokruženi potporni vektori i anotacije *hinge*
  gubitka. *Figure 1*

![linea-svm-primal-sgd](out/task-1/linear_svm_primal_sgd.png)

* Prikaz prosečnog *hinge* gubitka u zavisnosti od vrednosti korišćenog za selekciju hiperparametara. *Figure 2*

![c-selection](out/task-1/c_selection.png)

### Zadatak 2 — Dualni SVM (nelinearno jezgro, RBF)

Sažeti prikaz realizacije:

* Rešen je dualni problem SVM-a za dobijanje koeficijenata koristeći QP rešavač sa **RBF** jezgrom.
* Izvršena je pretraga po mreži za parametre jezgra (gamma / sigma) i kaznu ; rezultati su prikazani toplotnom mapom (
  heatmap) radi lakšeg odabira optimalnih hiperparametara.
* Vizuelizovane su konture odlučivanja, konture margina i potporni vektori (zaokruženi), uz anotacije odabranih
  vrednosti i mera gubitka.
* Kompletan proces je primenjen na skup podataka `res/svmData.csv`.

**Grafici:**

* Toplotna mapa pretrage po mreži (prosečan *hinge* gubitak) za odnos i parametra jezgra (sigma/gamma). *Figure 3*

![hyperparameter-search](out/task-2/hyperparameter_search.png)

* Konture odlučivanja, margine i potporni vektori za izabrani model sa RBF jezgrom. *Figure 4*

![nonlinear-svm-sv18](out/task-2/nonlinear_svm_sv18.png)

## Kako reprodukovati rezultate (ukratko)

* Instalirajte zavisnosti: `pip install -r requirements.txt`
* Postavite podatke na putanju `res/svmData.csv`.
* Pokrenite sledeće komande:
* `python src/task-1.py` — obučava primalni linearni SVM i čuva dve slike za Zadatak 1.
* `python src/task-2.py` — obučava dualni RBF SVM, pokreće pretragu po mreži i čuva dve slike za Zadatak 2.

## Napomene

* Grafikoni se čuvaju u direktorijumu `out/` (ili u direktorijumu koji je konfigurisan u kodu).
* Prikazi uključuju: boje klasa, označavanje potpornih vektora, anotacije *hinge* gubitka i vizuelizaciju procesa
  selekcije hiperparametara.
* Precizni detalji o rešavaču i samoj implementaciji nalaze se u komentarima unutar skripti `src/task-1.py` i
  `src/task-2.py`.
