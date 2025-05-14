
# Laboratorio di Ottimizzazione – Metodi del Gradiente

**Autore**: Alessandro Bianco  
**Data**: Settembre 2024

**Esame**: Laboratorio di Optimization Methods

## Descrizione del progetto
In questo laboratorio sono state implementate e analizzate diverse varianti del metodo del gradiente per risolvere problemi di ottimizzazione non vincolata, sfruttando la libreria PyCutest per il caricamento dei benchmark.

## Obiettivi
- Confrontare l’efficienza e la robustezza delle strategie di selezione del passo.
- Valutare l’impatto dei criteri di arresto sul numero di iterazioni e sui tempi di calcolo.
- Individuare le migliori pratiche per problemi con differenti caratteristiche (dimensione, condizionamento).

## Varianti del metodo del gradiente

1. **Passo costante**
2. **Armijo** (backtracking con condizione di sufficiente decremento)
3. **Armijo–Goldstein** (intervallo accettabile di decremento)
4. **Wolfe** (condizioni di curvatura)
5. **Armijo non monotono** (consente risalite temporanee)
6. **Armijo non monotono + Barzilai–Borwein** (passo iniziale stimato)

## Metodologia
- **Setup**: per ogni variante, si eseguono ripetute esecuzioni sui problemi di test.
- **Registrazione**: ad ogni iterazione si misura il tempo trascorso, il valore obiettivo e la norma del gradiente.
- **Arresto**: quando la norma del gradiente scende al di sotto di una soglia 1e-6 o dopo un massimo di 10.000 iterazioni.

## Benchmark utilizzati
- BROYDN7D
- BEALE
- BARD
- BOX
- BRKMCC

## Risultati e Analisi
- Grafici comparativi del **numero di iterazioni** e del **tempo di esecuzione** per ciascuna variante.
- Tabella riassuntiva con media e deviazione standard delle metriche.
- Osservazioni su convergenza rapida vs. stabilità del metodo.


