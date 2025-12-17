
import pandas as pd
import re

# --- Lematización ---
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Descargar recursos de nltk si no están
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()



# --- Cargar dataset ---





def lemmatize_text(text):
    """
    Aplica lematización palabra a palabra.
    """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])




def remove_duplicates(df: pd.DataFrame,
                          text_col: str = 'text',
                          label_col: str = 'label',
                          n_examples: int = 3) -> pd.DataFrame:
    """
    - Troba textos duplicats.
    - Si un text té més d'una etiqueta diferent, s'eliminen TOTES les seves files.
    - Si un text té sempre la mateixa etiqueta, només es deixa una fila (text, label).
    - Imprimeix 3 exemples de cada cas.
    """
    # Files que comparteixen el mateix text
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    dup_df = df[dup_mask]

    if label_col not in df.columns or dup_df.empty:
        print("[INFO] No s'ha trobat columna de label o no hi ha duplicats.")
        # En aquest cas, simplement traiem duplicats per text
        return df.drop_duplicates(subset=[text_col], keep='first').reset_index(drop=True)

    # Comptem quantes etiquetes diferents té cada text duplicat
    label_counts = dup_df.groupby(text_col)[label_col].nunique()
    same_texts = label_counts[label_counts == 1].index      # textos duplicats amb una sola etiqueta
    conflict_texts = label_counts[label_counts > 1].index   # textos amb etiquetes diferents

    print(f"[INFO] Duplicats amb la mateixa etiqueta (textos): {len(same_texts)}")
    print(f"[INFO] Duplicats amb etiquetes diferents (textos): {len(conflict_texts)}")

    # --- Exemples: duplicats amb la mateixa etiqueta ---
    print("\n[EXEMPLES] Duplicats amb la mateixa etiqueta (abans d'eliminar):")
    for text_val in list(same_texts[:n_examples]):
        sample = dup_df[dup_df[text_col] == text_val][[text_col, label_col]].head()
        print("----")
        print(sample.to_string(index=False))

    # --- Exemples: duplicats amb etiquetes diferents ---
    print("\n[EXEMPLES] Duplicats amb etiquetes diferents (abans d'eliminar):")
    for text_val in list(conflict_texts[:n_examples]):
        sample = dup_df[dup_df[text_col] == text_val][[text_col, label_col]].drop_duplicates().head()
        print("----")
        print(sample.to_string(index=False))

    # 1) Eliminar del dataframe tots els textos conflictius (més d'una etiqueta)
    df_clean = df[~df[text_col].isin(conflict_texts)].copy()

    # 2) Per la resta (incloent els "same_texts"), eliminar duplicats text+label,
    #    de manera que en quedi només un per cada (text, etiqueta).
    df_clean = df_clean.drop_duplicates(subset=[text_col, label_col], keep='first')

    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def lowercase_strip(text: str) -> str:
    """ Convierte el texto a minúsculas y
      elimina espacios al inicio y al final. """
    return text.lower().strip()

def remove_empty_texts(df: pd.DataFrame, column='text') -> pd.DataFrame:
    """
    Remove rows where the text column is NaN or empty after stripping spaces.
    """
    # Remove NaNs
    df = df.dropna(subset=[column])
    # Remove empty strings
    df = df[df[column].str.strip() != '']
    df = df.reset_index(drop=True)
    return df

def remove_punctuation_space(text: str) -> str:
    """
    Elimina signos de puntuación y sustituye por espacios.
    """
    # Puntuación a eliminar: guiones, comas, puntos, signos de interrogación/exclamación
    PUNCTUATION = re.compile(r'[.,!?;:…\"\'\-_/\\()#]+')
    # Sustituimos por espacio y convertimos a minúsculas
    return PUNCTUATION.sub(" ", text.lower())

def fix_abbr_en(text: str) -> str:
    """
    Expande abreviaciones comunes en inglés de tweets/mensajes,
    basadas en las más frecuentes de tu dataset.
    """
    if isinstance(text, list):
        words = text
    elif isinstance(text, str):
        words = text.split()
    else:
        raise TypeError('Input must be a string or a list of words.')

    abbrevs = {
        # Pronoms / paraules curtes típiques
        "u": "you",
        "ur": "your",
        "r": "are",
        "ya": "you",
        "&":"and",

        # Contractions sense apòstrof
        "im": "i am",
        "ive": "i have",
        "dont": "do not",
        "cant": "can not",
        "wont": "will not",
        "isnt": "is not",
        "shes": "she is",
        "hes": "he is",

        # Slang / xat
        "lol": "laughing out loud",
        "lmao": "laughing my ass off",
        "rofl": "rolling on the floor laughing",
        "omg": "oh my god",
        "omfg": "oh my fucking god",
        "idk": "i do not know",
        "btw": "by the way",

        "thx": "thanks",
        "thks": "thanks",
        "pls": "please",
        "plz": "please",

        "gr8": "great",
        "b4": "before",
        "l8r": "later",

        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "tbh": "to be honest",
        "smh": "shaking my head",

        "ily": "i love you",
        "brb": "be right back",
        "gtg": "got to go",
        "rn": "right now",
        "ikr": "i know right",
        "idc": "i do not care",
    }

    return " ".join(
        abbrevs.get(word.lower(), word)
        for word in words
    )

import re

def replace_links(text: str) -> str:
    # Patrón para detectar cualquier link común
    url_pattern = r'(http[s]?://\S+|www\.\S+|\S+\.ly/\S+)'
    return re.sub(url_pattern, '{link}', text)



def normalize_repeated_chars(text: str,
                             min_repeats: int = 3,
                             keep: int = 2) -> str:
    """
    Normaliza repeticiones excesivas de caracteres:
    - Solo toca secuencias de >= min_repeats del mismo carácter.
    - Las reduce a 'keep' repeticiones (por defecto 2).

    Ejemplos:
        "holaaaa"   -> "holaa"
        "goooood"   -> "good"
        "soooo"     -> "soo"
        "hmmmmmm"   -> "hmm"
        "sisterrrr" -> "sisterr"

    No toca:
        "good", "see", "cool" (porque solo tienen 2 letras iguales)
    """
    # (.)\1{2,} = un carácter seguido de sí mismo al menos 2 veces más (total 3 o más)
    pattern = re.compile(r'(.)\1{' + str(min_repeats - 1) + ',}')
    return pattern.sub(lambda m: m.group(1) * keep, text)

def normalize_laughs_en(text: str) -> str:
    """
    Normaliza risas escritas en inglés en un texto.
    Ejemplos:
        hahaha, hahahaha -> haha
        hehe, hehehe -> hehe
        hoho, hohoho -> hoho
        lmao, rofl -> lol 
    """
    words = text.split()
    normalized = []

    for word in words:
        w = word.lower()
        if 'ha' in w and w.count('h') + w.count('a') > 3:
            normalized.append('haha')
        elif 'he' in w and w.count('h') + w.count('e') > 3:
            normalized.append('hehe')
        elif 'ho' in w and w.count('h') + w.count('o') > 3:
            normalized.append('hoho')
        elif w in ['lmao', 'rofl']:
            normalized.append('lol')
        else:
            normalized.append(word)
    
    return " ".join(normalized)


def remove_mentions(text: str) -> str:
    """
    Sustituye menciones de usuario por {mention}.
    Ejemplo: @john -> {mention}
    """
    return " ".join(['{mention}' if word.startswith('@') else word for word in text.split()])



def remove_currency(text: str) -> str:
    """
    Replaces mentions of money symbols or currency words with {money}.
    Detects: €, $, £, yen, pound, euro, dollar
    """
    currency_words = ['€', '$', '£', 'yen', 'pound', 'euro', 'dollar']
    wlist = ['{money}' if any(c in word.lower() for c in currency_words) else word for word in text.split()]
    return " ".join(wlist)


def remove_special_characters(text: str) -> str:
    # Mantener solo letras, números y espacios
    return re.sub(r'[^a-zA-Z0-9\s{}]', '', text)





def mark_obvious_spam(df: pd.DataFrame, column: str = 'text',
                      max_unique_words: int = 2, min_length: int = 20) -> pd.DataFrame:
    """
    Marca com a 'spam' els textos amb molt poques paraules úniques i longitud prou gran.
    En comptes de crear cap columna nova, afegeix el token '{spam}' al text original
    quan compleix el criteri.
    """
    def is_spam(t: str) -> bool:
        words = t.split()
        return (len(set(words)) <= max_unique_words) and (len(t) > min_length)

    df[column] = df[column].astype(str)
    df[column] = df[column].apply(
        lambda t: "{spam} " + t if is_spam(t) else t
    )
    df = df.reset_index(drop=True)
    return df







