# 📝 Daglig Logg — Fordypningsoppgave

> **Periode:** 15. april – 25. april 2026
> **Tema:** Bildeklassifisering med maskinlæring
> **Fag:** Utvikling (fordypningsoppgave)

---

## Dag 1 — Onsdag 15. april
**Tema:** Kaggle-kurs — Intro til TensorFlow & Keras

**Fokus:**
- Gjennomgå Kaggle sin "Intro to Machine learning"
- Forstå hva TensorFlow og Keras er, og hvordan de fungerer sammen
- Kjøre eksempelkoder i Kaggle Notebook

**Hva jeg gjorde:**
- Intro to machine learning

**Hva jeg lærte:**
- forstå og laste opp data til model
- Underfitting og Overfitting for bedre performance
- Random Forest: ML algoritme 

**Utfordringer:**
- 

---



## Dag 2 — Torsdag 17. april
**Tema:** Forelesning om CNN 

**Fokus:**
- se youtube video om Convolutional Neural Network

**Hva jeg gjorde:**
- fikk et innblikk om hvordan CNN fungerer 
**Hva jeg lærte:**
- Hvordan mennesker gjenkjenner bilder 
- Hvordan data bruker filtere for å finne detaljer
- ReLU (Rectified Linear Unit)
- Pooling
**Utfordringer:**
-

---

## Dag 3 — Fredag 17. april
**Tema:** Kaggle-kurs — Intro to Deep Learning

**Fokus:**
- Starte med "Intro to deep learning" (DNN, Dropout, Batch Normalization)
- Gjøre øvelsene i Kaggle Notebooks

**Hva jeg gjorde:**
- gikk gjennom tre temaer, gjennom tutorials og exercises

**Hva jeg lærte:**
- lineære enheter og byggesteinene i deep learning
- Deep Neural nettworks 
- The loss function og optimizer

**Utfordringer:**
- 

---

## Dag 4 — Mandag 20. april
**Tema:** Kaggle-kurs — Intro to Deep Learning

**Fokus:**
- Fullføre "Intro to deep learning"
- Gjøre øvelsene i Kaggle Notebooks

**Hva jeg gjorde:**
- Fullførte Intro to Deep learning
- Startet på kurset "Computer Vision" 

**Hva jeg lærte:**
- Overfitting og Underfitting
- Dropout and Batch Normalization 
- Binary Classification
- The Convolutional Classifier 

**Utfordringer:**
- Hvordan man oppdager Overfitting eller Underfitting

---

## Dag 5 — Tirsdag 21. april
**Tema:** Kaggle-kurs — Computer vision 

**Fokus:**
- Computer Vision kurs som dekker CNN (Convolutional Neural Networks)
- Forstå hvorfor CNN er egnet for bildeklassifisering
- Lære om convolutional lag, pooling og feature detection
- Gjøre øvelsene i Kaggle Notebooks

**Hva jeg gjorde:**
- Fullførte kurset om Computer Vision

**Hva jeg lærte:**
- Convolution og ReLU
- Maximum Pooling (Feature Extraction)
- Padding og Stride 
- Designe en enkel Binær Bildeklassifiserer (skille mellom to kategorier)
- Data augmentation 
**Utfordringer:**
- 

---


## Dag 6 — Onsdag 22. april
**Tema:** Datasett & databehandling 

**Fokus:**
- Velge datasett (CIFAR-10 eller eget datasett)
- Laste inn og utforske dataene (former, klasser, eksempelbilder)
- Normalisere og forberede data (train/validation/test-split)
- Visualisere eksempelbilder med Matplotlib

**Hva jeg gjorde:**
- Utforsket et datasett fra Tensorflow (MNIST)
- Feilsøkte problemer med WSL, Vscode og python pakker 

**Hva jeg lærte:**
- Ikke ha prosjektet ditt i "mnt" mappen i WSL (Det går tregt)

**Utfordringer:**
- WSL. Brukte over halve dagen på å finne ut av et problem med virtual environment og vscode. 

---


## Dag 7 — Torsdag 23. april
**Tema:** Transfer learning og feature extraction

**Fokus:**
- Implementere data augmentation (rotasjon, flipping, zoom)
- Utforske transfer learning med en pretrained modell (f.eks. MobileNetV2)
- Sammenligne resultater med og uten forbedringer

**Hva jeg gjorde:**
- valgte en base model (MobileNetV2), fordi den er effektiv og lettvektig i forhold til andre modeller (som ResNet50). Passer fint hvis man skal mobile applikasjoner.
- Startet med fine tuning av modellen
**Hva jeg lærte:**
- litt mer om transfer learning, og analysere resultater 

**Utfordringer:**
-

---

## Dag 8 — Fredag 24. april
**Tema:** Fine-tune + dokumentasjon
**Fokus:**
- Fine tuning
- Dokumentasjon
- Lage et script som tar i mot et bilde og returnerer prediksjon
- Rydde opp i koden

**Hva jeg gjorde:**
- Eksprimenterte med ulike topplagskonfigurasjoner
- la til class weights for bedre håndtering av datasettet
- oppnådde ~87% val accuracy

**Hva jeg lærte:**
- Ubalanserte klasser påvirker accuracy betydlig
- For agressiv augmentation kan føre til mer overfitting


**Utfordringer:**
- redusere overfitting, kan hende at et større datasett ville ha vært løsningen

---

## 📚 Ressurser

- [Kaggle — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Kaggle — Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)
- [Kaggle — Computer Vision](https://www.kaggle.com/learn/computer-vision)
- [TensorFlow — Transfer Learning Documentation](https://www.tensorflow.org/tutorials/images/transfer_learning#data_preprocessing)