# CIFAR-10 Image Classification with PyTorch

Detta projekt implementerar en komplett tränings- och utvärderingspipeline för bildklassificering på CIFAR-10 med PyTorch. Projektet täcker hela flödet från datahantering och modellimplementation till träningsloop, utvärdering, hyperparameter-experiment och analys av resultat.

---

## Dataset

Vi använder CIFAR-10, ett välkänt benchmark-dataset inom bildklassificering. Det består av 50 000 träningsbilder och 10 000 testbilder fördelade över 10 klasser, där varje bild har storleken 3 × 32 × 32 pixlar. Datasetet är balanserat med 5 000 bilder per klass i träningsmängden, vilket gör det lämpligt för att utvärdera modellers generella klassificeringsförmåga.

---

## Modell

I detta projekt används en enkel feedforward neural network (MLP)
bestående av fullt kopplade lager.

Bilderna (3×32×32) flattenas till en vektor med 3072 värden
och matas sedan genom:

```python
nn.Sequential(
    nn.Linear(3072, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```
Modellen valdes som en medveten baslinje för att tydligt kunna
fokusera på träningspipeline, hyperparametrar och experimentstruktur.

---

## Träning

Modellen tränas med CrossEntropyLoss och optimeras med Adam. Träningen körs i 5 epochs med en batch size på 64. Under varje epoch skrivs aktuell loss ut för att följa konvergensen.

---

## Experiment

För att hitta en lämplig learning rate testades tre olika värden med 5 epochs vardera:

| Learning Rate | Epochs | Test Accuracy |
|---------------|--------|---------------|
| 0.001         | 5      | 0.4363        |
| 0.0001        | 5      | 0.4169        |
| 0.01          | 5      | 0.1000        |

### Resultat

Learning rate 0.001 gav den högsta accuracy (~0.44) och valdes som bästa hyperparameter. En learning rate på 0.01 visade sig vara för hög och ledde till instabil träning, medan 0.0001 fungerade men konvergerade betydligt långsammare. Värdet 0.001 gav stabil och effektiv konvergens, vilket tydligt illustrerar hur stor påverkan learning rate har på modellens prestanda och stabilitet.

---

## Optimering

Modellen optimeras med gradient descent genom att iterativt uppdatera vikterna i riktning mot den negativa gradienten av loss-funktionen. I neurala nätverk sker detta i ett högdimensionellt parameterutrymme som inte kan visualiseras direkt, men principen motsvarar att röra sig nedför en förlustyta mot ett minimum.

---

## Struktur

```
project/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
├── experiments.ipynb
└── README.md
```

---

## Hur man kör projektet

För att träna och utvärdera modellen kör du helt enkelt `python main.py` från projektets rotkatalog. Vill du köra hyperparameter-experimenten öppnar du notebooken `experiments.ipynb` i Jupyter.

---

## Framtida förbättringar

En naturlig vidareutveckling vore att ersätta den fullt kopplade
modellen med ett konvolutionellt neuralt nätverk (CNN), som är
bättre anpassat för bildklassificering eftersom det bevarar
spatial information.

Ytterligare förbättringar kan inkludera data augmentation,
fler träningsepoker, learning rate scheduling och regularisering.