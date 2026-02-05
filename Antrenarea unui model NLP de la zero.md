# Antrenarea unui Model NLP de la Zero

Acest ghid vă prezintă pașii conceptuali și practici pentru antrenarea unui model Word2Vec, aliniați cu codul din `train.py`.

## Cuprins
1.  [Colectarea Datelor](#1-colectarea-datelor)
2.  [Curățarea Datelor](#2-curățarea-datelor)
3.  [Vectorizarea & "Cutia Neagră"](#3-vectorizarea--cutia-neagră)
    *   [Cum Funcționează Skip-Gram](#cum-funcționează-skip-gram)
    *   [Dimensiuni](#dimensiuni)
4.  [Antrenarea (Procesul)](#4-antrenarea-procesul)
    *   [Notă de Eficiență](#notă-de-eficiență)
5.  [Detalii Tehnice (Aprofundare)](#5-detalii-tehnice-aprofundare)
    *   [Vectorizarea Inițială (Start Aleatoriu)](#1-vectorizarea-inițială-start-aleatoriu)
    *   [Calcularea Pierderii (Cât de mult am greșit?)](#2-calcularea-pierderii-cât-de-mult-am-greșit)
    *   [Backpropagation (Ajustarea Neuronilor)](#3-backpropagation-ajustarea-neuronilor)
    *   [Bucla de Antrenare Pas cu Pas](#4-bucla-de-antrenare-pas-cu-pas)
6.  [Concepte Avansate & Tehnici Moderne](#6-concepte-avansate--tehnici-moderne)
    *   [N-Grame (Fraze)](#1-n-grame-fraze)
    *   [FastText (Informații Sub-cuvânt)](#2-fasttext-informații-sub-cuvânt)
    *   [Transformers (BERT/GPT)](#3-transformers--contextual-embeddings-bertgpt)

---

## 1. Colectarea Datelor

Datele pe care le oferim modelului determină înțelegerea acestuia asupra lumii. Principiul "Gunoi intră, gunoi iese" se aplică puternic aici. Pentru a construi un model robust, avem nevoie de date text curate din surse diverse:

*   **Text Formal (ex. Wikipedia, Cărți):** Oferă gramatică corectă și "împământare" în fapte obiective.
    *   *Scop Conceptual:* Datele Wiki sunt seci și dense în informații, ancorând modelul în "adevăr". Este excelent pentru cunoștințe enciclopedice, dar îi lipsește profunzimea emoțională.
*   **Text Narativ (ex. Romane):** Permite modelului să vadă cuvinte în contexte variate, captând relații tematice și emoționale.
    *   *Scop Conceptual:* Romanele oferă "EQ" (inteligență emoțională) sau intuiție. Cuvintele sunt legate în moduri efemere, non-obiective (ex. "dragoste" și "pierdere" apărând împreună), ajutând modelul să înțeleagă sentimentul uman.
*   **Text Informal (Știri/Chat):** Ajută modelul să preia argoul modern și utilizarea limbajului în evoluție (deși kit-ul nostru actual se concentrează pe primele două).

În `train.py`, automatizăm acest lucru descărcând un mix de literatură clasică (Sherlock Holmes, Frankenstein) și un subset de Wikipedia. Acest lucru ne oferă un amestec de vocabular și structură.

## 2. Curățarea Datelor

Înainte ca modelul să poată învăța, trebuie să simplificăm textul. Textul brut conține zgomot care poate confunda modelul (ex. "Cuvânt." și "cuvânt" ar putea fi tratate ca lucruri diferite dacă nu suntem atenți).

În scriptul nostru `train.py`, folosim un `MemoryFriendlyIterator` pentru a curăța datele din mers pe măsură ce sunt citite de pe disc. Acest lucru este eficient și menține utilizarea memoriei scăzută.

**Strategia Noastră de Curățare:**
1.  **Minuscule (Lowercasing):** Convertim "The" și "the" la același token, astfel încât modelul să învețe că reprezintă același concept.
2.  **Eliminarea Zgomotului:** Folosim o Expresie Regulată (`re.sub(r'[^a-zA-Z\s]', '', line.lower())`) pentru a păstra doar literele și spațiile. Numerele și punctuația sunt eliminate pentru a ne concentra pur pe relațiile dintre cuvinte.

*Notă: În mediile de producție, ați putea folosi tokenizatoare mai avansate (precum `spaCy` sau `nltk`) pentru a gestiona mai bine punctuația și entitățile speciale, dar pentru învățarea înglobărilor de cuvinte (word embeddings), această abordare simplă este foarte eficientă.*

## 3. Vectorizarea & "Cutia Neagră"

"Vectorizarea" înseamnă pur și simplu convertirea cuvintelor în liste de numere ([vectori](#1-vectorizarea-inițială-start-aleatoriu)) poziționate într-un spațiu multidimensional. Scopul este de a plasa cuvinte similare semantic aproape unele de altele în acest spațiu.

*   **Conceptul:** Imaginați-vă un grafic 2D. Vrem ca "Rege" și "Bărbat" să fie aproape, iar "Regină" și "Femeie" să fie aproape.
*   **Matematică pe Înțeles:** Plasând cuvintele în spațiu, putem efectua aritmetică pe sensurile lor.
    *   *Analogie Clasică:* `Rege - Bărbat + Femeie = Regină`.
    *   Dacă iei vectorul pentru "Rege", scazi "Bărbăția" din el și adaugi "Feminitatea", poziția rezultată în spațiu este cea mai apropiată de vectorul pentru "Regină". Aceasta demonstrează că modelul a învățat *conceptul* de gen, nu doar ce cuvinte stau unul lângă altul.
*   **Aproximare Universală:** Conceptual, creăm o funcție matematică care întruchipează poziția spațială (în vocabularul nostru) a cuvântului. Nu trebuie să vizualizăm cele 100 de dimensiuni; trebuie doar să avem încredere că matematica (Rețele Neuronale) poate aproxima această "formă" a sensului, permițându-ne să calculăm relațiile eficient.

*   **Matematica:** Folosim o abordare de rețea neuronală **Skip-Gram** (setată prin `sg=1` în cod).

### Cum Funcționează Skip-Gram
În loc să prezică un cuvânt bazat pe contextul său (completează spațiul liber), Skip-Gram face invers. Ia un cuvânt specific (cuvântul "central") și încearcă să prezică cuvintele care apar probabil în jurul său ("contextul").

*   **Exemplu de Propoziție:** "The **king** owns these lands"
*   **Intrare:** "king"
*   **Ieșire Țintă (Context):** "The", "owns", "these", "lands"

Încercând repetat să prezică cuvintele de context dintr-un cuvânt central, modelul își ajustează numerele interne ([greutăți/neuroni](#3-backpropagation-ajustarea-neuronilor)). În timp, aceste greutăți devin "vectorii" care reprezintă sensul cuvântului.

### Dimensiuni
În codul nostru, setăm `VECTOR_SIZE = 100`. Aceasta înseamnă că fiecare cuvânt este reprezentat de o listă de 100 de numere.
*   **De ce 100?** Este un punct optim pentru dimensiunea setului nostru de date.
*   **De ce nu 300?** Google folosește 300 pentru modele masive antrenate pe miliarde de cuvinte. Pentru biblioteca noastră mai mică de cărți, 100 captează suficiente detalii fără a supra-ajusta (overfitting) sau a rula prea încet.

## 4. Antrenarea (Procesul)

Antrenarea are loc în **Epoci**. O epocă este o citire completă a întregii noastre biblioteci de cărți.

1.  **Streaming:** Nu încărcăm toate cărțile în RAM. `MemoryFriendlyIterator` transmite fluxul linie cu linie.
2.  **Fereastra (Windowing):** Modelul privește o "Fereastră" de cuvinte. Codul nostru setează `WINDOW = 5`, ceea ce înseamnă că privește 5 cuvinte înainte și 5 cuvinte după cuvântul central.
3.  **Învățare:**
    *   Modelul face o predicție.
    *   Calculează [Pierderea](#2-calcularea-pierderii-cât-de-mult-am-greșit) (cât de mult am greșit?).
    *   Actualizează vectorii ușor pentru a reduce acea eroare (vezi [Backpropagation](#3-backpropagation-ajustarea-neuronilor)).
4.  **Repetare:** Facem acest lucru pentru `EPOCHS = 10`. Expunerea repetată întărește modelele.

### Notă de Eficiență
Biblioteca `gensim` pe care o folosim în `train.py` este extrem de optimizată. Folosește cod C în spate pentru a efectua aceste operații matematice incredibil de rapid, utilizând nuclee CPU multiple (`workers=WORKERS`) pentru a paraleliza sarcina. Acest lucru se aliniază perfect cu modul în care procesoarele gestionează tipurile standard de numere (floats/ints), făcând antrenarea extrem de eficientă.

Odată ce antrenarea este completă, modelul este salvat în `models/word2vec_simple.model`. Acest fișier conține "creierul" – vectorii învățați pentru fiecare cuvânt din vocabularul nostru.

---

## 5. Detalii Tehnice (Aprofundare)

### 1. Vectorizarea Inițială (Start Aleatoriu)
Cum transformăm un cuvânt în numere *înainte* de a învăța ceva? Trișăm! Începem cu zgomot aleatoriu.

*   **Matricea:** Creăm o matrice gigantică (tabel) de dimensiunea `Mărime Vocabular x Mărime Vector`.
*   **Aleatoriu:** Umplem acest tabel cu numere aleatorii mici (ex. între -0.01 și 0.01).
*   **Căutare:** Fiecărui cuvânt unic îi este atribuit un index (ex. "rege" = 42). Vectorul pentru "rege" este pur și simplu rândul 42 al acestei matrice.
*   **Scopul:** Antrenarea este pur și simplu procesul de actualizare a acestor numere aleatorii până când reprezintă un sens.

### 2. Calcularea Pierderii (Cât de mult am greșit?)
Pentru a repara o greșeală, trebuie mai întâi să o măsurăm. În Word2Vec, "Pierderea" (Loss) este calculată folosind **Negative Sampling** (o optimizare a Softmax).

*   **Problema:** Calcularea probabilității ca "rege" să prezică "deține" față de *orice alt cuvânt din dicționar* (aprox. 100.000 de cuvinte) este prea lentă.
*   **Soluția (Negative Sampling):** Întrebăm modelul: "Este 'deține' vecinul lui 'rege'?" (Da). Apoi alegem 5 cuvinte aleatorii care *nu* sunt vecini (ex. "taco", "albastru", "spaghete") și întrebăm: "Sunt acestea vecine?" (Nu).
*   **Matematica:** Folosim o funcție Sigmoid. Vrem ca ieșirea să fie `1` pentru cuvântul corect și `0` pentru cuvintele aleatorii "negative". Diferența dintre ceea ce am obținut (ex. 0.3 pentru cuvântul real) și ceea ce am dorit (1.0) este **Pierderea**.

### 3. Backpropagation (Ajustarea Neuronilor)
Odată ce avem Pierderea, trebuie să schimbăm vectorii pentru a face mai bine data viitoare. Aceasta este **Backpropagation** folosind **Gradient Descent**.

*   **Gradientul:** Imaginați-vă că stați pe un deal (eroare mare) și simțiți cu picioarele în ce direcție este "jos" (eroare mai mică). Gradientul este acea direcție.
*   **Actualizarea:** Luăm numerele vectorului curent și scădem o fracțiune mică din gradient.
    *   `Greutate Nouă = Greutate Veche - (Rată de Învățare * Gradient)`
*   **Rezultatul:** Vectorul pentru "rege" se mută ușor mai aproape (matematic) de "deține" și ușor mai departe de "taco".

### 4. Bucla de Antrenare Pas cu Pas
Iată exact ce se întâmplă pentru un singur pas de antrenare cu propoziția "The **king** owns...":

1.  **Căutare:** Modelul preia vectorul curent pentru cuvântul de intrare "king" (Rândul 42).
2.  **Proiecție:** Înmulțește acest vector cu o matrice de greutăți pentru a obține scoruri pentru potențialele cuvinte de context.
3.  **Predicție:** Aplică funcția Sigmoid acestor scoruri pentru a obține probabilități.
4.  **Calcularea Erorii:** Compară probabilitatea cuvântului țintă ("owns") cu 1.0, și a mostrelor negative ("taco") cu 0.0.
5.  **Backprop:** Calculează gradientul ("panta" erorii).
6.  **Actualizare:** Ajustează numerele în vectorul "king" și vectorul "owns" pentru a reduce eroarea.
7.  **Următorul Cuvânt:** Trece la următorul cuvânt din propoziție și repetă.

---

## 6. Concepte Avansate & Tehnici Moderne

Deși începem cu Word2Vec standard, NLP-ul modern a evoluat. Iată tehnici cheie folosite în producție:

### 1. N-Grame (Fraze)
"New York" implică un oraș, dar "New" și "York" separat înseamnă "nou" și "un oraș în Anglia".
*   **Tehnică:** Putem pre-procesa textul pentru a găsi cuvinte care apar statistic împreună mai des decât ar fi întâmplător.
*   **Rezultat:** Le fuzionăm într-un singur token: `new_york`. Modelul învață apoi un vector specific pentru acest concept unic, separat de "new" sau "york".

### 2. FastText (Informații Sub-cuvânt)
Ce se întâmplă dacă modelul vede cuvântul "unforgettableness" dar nu a fost niciodată antrenat pe el? Word2Vec se blochează (sau îl ignoră).
*   **Tehnică (FastText):** Dezvoltat de Facebook, tratează cuvintele ca saci de n-grame de caractere.
    *   "apple" devine `<ap`, `app`, `ppl`, `ple`, `le>`.
*   **Beneficiu:** Modelul poate construi un sens pentru "unforgettableness" combinând vectorii pentru "un-", "forget", "-able", și "-ness". Înțelege cuvinte pe care nu le-a mai văzut niciodată!

### 3. Transformers & Contextual Embeddings (BERT/GPT)
Word2Vec are un defect: "Bank" are același vector în "river bank" (mal de râu) și "bank deposit" (depozit bancar).
*   **Tehnică (Transformers):** Modele precum BERT nu au vectori statici. Ele generează un vector *din mers* bazat pe întreaga propoziție.
*   **Rezultat:** Vectorul pentru "bank" se schimbă complet în funcție de cuvintele din jur. Aceasta este fundația LLM-urilor (Large Language Models) moderne precum cel care vă ajută chiar acum.
