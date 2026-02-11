import os
import glob
import numpy as np
import pandas as pd


def compute_state_changes(df):
    """Voeg kolommen toe die state changes en hun duur berekenen."""

    # Detecteer state changes (verschil van +2 of -2)
    df['state_change'] = ((df['state'].diff() == -2) | (df['state'].diff() == 2)).astype(int)

    # Initialiseer kolommen met NaN
    df['p_state_dur'] = np.nan
    df['trial_from_state_change'] = np.nan
    df['trial_to_state_change'] = np.nan
    df['only_standards_in_state'] = 1

    # Vind indices van veranderingen
    change_indices = np.where(df['state_change'] == 1)[0]

    # Zorg dat er minstens één begin- en eindpunt is
    if len(change_indices) == 0:
        print("⚠️ Geen state changes gevonden — controleer je data.")
        return df

    # Voeg -1 toe om laatste segment af te sluiten
    changes = df['trial_nr'].iloc[np.concatenate((change_indices, np.array([-1])))]

    # Bereken duur per state
    durs = changes.diff()
    for i in range(changes.shape[0] - 1):
        state_start = changes.iloc[i]
        state_end = changes.iloc[i + 1]
        df.loc[
            (df['trial_nr'] >= state_start) & (df['trial_nr'] <= state_end),
            'p_state_dur'
        ] = durs.iloc[i]

    # trial_to_state_change
    changes = df['trial_nr'].iloc[np.concatenate((np.array([0]), change_indices))]
    for i in range(changes.shape[0] - 1):
        state_start = changes.iloc[i]
        state_end = changes.iloc[i + 1]
        state_dur = state_end - state_start
        df.loc[
            (df['trial_nr'] >= state_start) & (df['trial_nr'] <= state_end),
            'trial_to_state_change'
        ] = np.arange(state_dur + 1, 0, -1) - 1

    # trial_from_state_change
    changes = df['trial_nr'].iloc[np.concatenate((change_indices, np.array([-1])))]
    for i in range(changes.shape[0] - 1):
        state_start = changes.iloc[i]
        state_end = changes.iloc[i + 1]
        state_dur = state_end - state_start
        df.loc[
            (df['trial_nr'] >= state_start) & (df['trial_nr'] <= state_end),
            'trial_from_state_change'
        ] = np.arange(0, state_dur + 1, 1)

    return df


if __name__ == "__main__":
    # Zoek in lokale 'data' map in dezelfde map als het script
    data_dir = os.path.join(os.getcwd(), "data")

    # Input met foutafhandeling
    subject_nr = input("Subject #: ").strip()
    session_nr = input("Session #: ").strip()
    block_nr = input("Block #: ").strip()

    # Zoek bestanden met datum/tijd en "_events.tsv" in de naam
    pattern = os.path.join(data_dir, f"{subject_nr}_{session_nr}_{block_nr}_*_events.tsv")
    tsv_files = glob.glob(pattern)

    # Toon feedback als er niets wordt gevonden
    if not tsv_files:
        print("⚠️ Geen bestanden gevonden. Controleer of je data hier staan:")
        print(f"Pad: {data_dir}")
        print(f"Zoekpatroon: {pattern}")
        print("Beschikbare TSV-bestanden:")
        print(glob.glob(os.path.join(data_dir, '*.tsv')))
        raise FileNotFoundError(f"Geen TSV-bestanden gevonden voor patroon: {pattern}")

    # Neem het nieuwste bestand als er meerdere matches zijn
    tsv_file = max(tsv_files, key=os.path.getctime)
    print(f"📂 Bestand geladen: {os.path.basename(tsv_file)}")

    # Lees en filter data
    df = pd.read_csv(tsv_file, sep="\t")

    if 'phase' not in df.columns:
        raise KeyError("Kolom 'phase' ontbreekt in de dataset.")

    df = df.loc[df['phase'] == 4, :]

    # Controleer of vereiste kolommen bestaan
    for col in ['trial_nr', 'state']:
        if col not in df.columns:
            raise KeyError(f"Vereiste kolom '{col}' ontbreekt in de data.")

    # Bereken state changes
    df = compute_state_changes(df)

    # Toon resultaat
    print("\n" + "-" * 30)
    print(f"Subj {subject_nr}, Ses {session_nr}, Block {block_nr}:")
    print(f"# state changes = {int(df['state_change'].sum())}")
    print("-" * 30 + "\n")
