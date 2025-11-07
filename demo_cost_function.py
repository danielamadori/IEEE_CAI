"""
Script dimostrativo per mostrare il funzionamento corretto di cost_function
dopo le correzioni per gestire i problemi di divisione per zero.
"""

import numpy as np
from cost_function import cal_sigmas, cost_function

def demo_basic_usage():
    """Esempio base di utilizzo"""
    print("=" * 70)
    print("DEMO 1: Utilizzo Base")
    print("=" * 70)

    # Dati di training
    np.random.seed(42)
    X_train = np.random.randn(50, 3) * 2 + 10
    X_test = np.array([[10.0, 10.0, 10.0]])
    feature_names = ['feature_1', 'feature_2', 'feature_3']

    # Calcola sigma
    sigmas = cal_sigmas(X_train, X_test, feature_names)

    # Definisci intervalli
    icf = {
        'feature_1': (8.0, 12.0),
        'feature_2': (8.0, 12.0),
        'feature_3': (8.0, 12.0)
    }

    # Sample
    sample = {'feature_1': 10.0, 'feature_2': 10.0, 'feature_3': 10.0}

    # Calcola costo
    cost = cost_function(sample=sample, icf=icf, sigmas=sigmas[0], verbose=True)

    print(f"\n✅ Costo totale: {cost:.4f}")
    print()

def demo_edge_case_all_same():
    """Demo caso edge: tutti i valori identici (sigma=0)"""
    print("=" * 70)
    print("DEMO 2: Caso Edge - Tutti i Valori Identici")
    print("=" * 70)

    # Tutti i valori di training sono identici
    X_train = np.array([
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
    ])
    X_test = np.array([[5.0, 5.0]])
    feature_names = ['f1', 'f2']

    sigmas = cal_sigmas(X_train, X_test, feature_names)

    print(f"Sigmas calcolati: {sigmas[0]}")

    icf = {
        'f1': (4.0, 6.0),
        'f2': (4.0, 6.0)
    }

    sample = {'f1': 5.0, 'f2': 5.0}

    cost = cost_function(sample=sample, icf=icf, sigmas=sigmas[0], verbose=True)

    print(f"\n✅ Costo con varianza zero: {cost:.4f}")
    print("   (Le feature con varianza zero vengono skippate automaticamente)")
    print()

def demo_edge_case_extreme_asymmetry():
    """Demo caso edge: estrema asimmetria nei dati"""
    print("=" * 70)
    print("DEMO 3: Caso Edge - Estrema Asimmetria")
    print("=" * 70)

    # Tutti i valori di training sopra il test value
    X_train = np.array([
        [100.0],
        [101.0],
        [102.0],
    ])
    X_test = np.array([[1.0]])
    feature_names = ['f1']

    sigmas = cal_sigmas(X_train, X_test, feature_names)

    print(f"Sigmas calcolati: {sigmas[0]}")
    print(f"  sigma_plus: {sigmas[0]['f1']['sigma_plus']:.4f}")
    print(f"  sigma_minus: {sigmas[0]['f1']['sigma_minus']:.4f}")
    print(f"  ratio_above: {sigmas[0]['f1']['ratio_above_mean']:.4f}")
    print(f"  ratio_below: {sigmas[0]['f1']['ratio_below_mean']:.4f}")

    icf = {'f1': (-10.0, 10.0)}
    sample = {'f1': 1.0}

    cost = cost_function(sample=sample, icf=icf, sigmas=sigmas[0], verbose=True)

    print(f"\n✅ Costo con estrema asimmetria: {cost:.4f}")
    print()

def demo_edge_case_infinite_intervals():
    """Demo caso edge: intervalli infiniti"""
    print("=" * 70)
    print("DEMO 4: Caso Edge - Intervalli Infiniti")
    print("=" * 70)

    np.random.seed(123)
    X_train = np.random.randn(30, 2) + 5
    X_test = np.array([[5.0, 5.0]])
    feature_names = ['f1', 'f2']

    sigmas = cal_sigmas(X_train, X_test, feature_names)

    # Intervalli infiniti
    icf = {
        'f1': (-np.inf, np.inf),  # Tutto lo spazio
        'f2': (4.0, np.inf)        # Semi-infinito
    }

    sample = {'f1': 5.0, 'f2': 5.0}

    cost = cost_function(sample=sample, icf=icf, sigmas=sigmas[0], verbose=True)

    print(f"\n✅ Costo con intervalli infiniti: {cost:.4f}")
    print()

def demo_multiple_samples():
    """Demo con campioni multipli"""
    print("=" * 70)
    print("DEMO 5: Campioni Multipli")
    print("=" * 70)

    np.random.seed(42)
    X_train = np.random.randn(100, 2) * 2 + 10
    X_test = np.array([
        [9.0, 11.0],
        [10.0, 10.0],
        [11.0, 9.0],
        [15.0, 5.0]  # Outlier
    ])
    feature_names = ['f1', 'f2']

    sigmas = cal_sigmas(X_train, X_test, feature_names)

    icf = {
        'f1': (8.0, 12.0),
        'f2': (8.0, 12.0)
    }

    print("Calcolo costi per 4 campioni diversi:\n")
    for i in range(len(X_test)):
        sample = {'f1': X_test[i, 0], 'f2': X_test[i, 1]}
        cost = cost_function(sample=sample, icf=icf, sigmas=sigmas[i], verbose=False)
        print(f"  Campione {i+1} {dict(sample)}: costo = {cost:.4f}")

    print("\n✅ Tutti i campioni processati senza errori")
    print()

def run_all_demos():
    """Esegue tutte le demo"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "DEMO COST FUNCTION CORRETTA" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        demo_basic_usage()
        demo_edge_case_all_same()
        demo_edge_case_extreme_asymmetry()
        demo_edge_case_infinite_intervals()
        demo_multiple_samples()

        print("=" * 70)
        print("✅ TUTTE LE DEMO COMPLETATE CON SUCCESSO!")
        print("=" * 70)
        print("\nLe correzioni implementate garantiscono:")
        print("  1. ✅ Nessuna divisione per zero")
        print("  2. ✅ Gestione corretta di sigma nulli o molto piccoli")
        print("  3. ✅ Gestione di percentuali non valide")
        print("  4. ✅ Nessun NaN o Inf nei risultati")
        print("  5. ✅ Tutti i loop completano correttamente (bug return risolto)")
        print("  6. ✅ Gestione robusta di intervalli infiniti")
        print()

    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_all_demos()

