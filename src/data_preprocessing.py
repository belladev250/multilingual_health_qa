import os
import re
import random
import unicodedata
import pandas as pd

def clean_text(text: str) -> str:
    """
    Cleans raw text by removing HTML tags, normalizing whitespace,
    and applying Unicode NFKC normalization.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize unicode (crucial for languages like Amharic with Ge'ez script)
    text = unicodedata.normalize('NFKC', text)
    
    # Normalize punctuation and spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def apply_language_prefix(question: str, language: str) -> str:
    """
    Formats the input prompt with a language prefix to help the multilingual
    seq2seq model understand which language is being processed (EXP-03).
    """
    lang_map = {
        'lug': 'Luganda',
        'luganda': 'Luganda',
        'swa': 'Kiswahili',
        'kiswahili': 'Kiswahili',
        'aka': 'Akan',
        'akan': 'Akan',
        'amh': 'Amharic',
        'amharic': 'Amharic',
        'eng': 'English',
        'english': 'English'
    }
    lang_name = lang_map.get(language.lower()[:3], language.title())
    return f"Language: {lang_name} | Topic: Maternal and Sexual Health | Question: {question}"

def simulate_back_translation(text: str, language: str) -> str:
    """
    Simulates a back-translation data augmentation process (EXP-04).
    In real usage, this would translate text -> English -> text using MarianMT.
    Here we simulate it by applying synonym substitutions and slight phrasing variations
    to avoid running slow translation models in testing, but keeping the pipeline identical.
    """
    if not isinstance(text, str) or len(text) < 5:
        return text
    
    # Simple dictionary of some words we can swap to simulate back-translation variations
    synonyms = {
        # Kiswahili
        "habari": "jambo",
        "mama": "mzazi wa kike",
        "afya": "uzima",
        "ujauzito": "mimba",
        "mtoto": "mchanga",
        # Amharic
        "ጤና": "ደህንነት",
        "እናት": "ወላጅ",
        "ልጅ": "ህፃን",
        # English / General placeholders
        "health": "wellness",
        "pregnant": "expecting",
        "child": "infant",
        "fever": "high temperature"
    }
    
    words = text.split()
    augmented_words = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word).lower()
        if clean_word in synonyms and random.random() < 0.4:
            # Swap word but keep punctuation/capitalization simple
            augmented_words.append(synonyms[clean_word])
        else:
            augmented_words.append(word)
            
    return " ".join(augmented_words)

def generate_mock_datasets(output_dir: str):
    """
    Generates realistic synthetic datasets for testing the pipeline.
    Covers four target low-resource/regional African languages:
    Luganda (lug), Kiswahili (swa), Akan (aka), and Amharic (amh).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Multilingual health QA templates
    qa_templates = [
        {
            "lang": "Kiswahili",
            "code": "swa",
            "q": "Je, ni dalili gani za hatari wakati wa ujauzito?",
            "a": "Dalili za hatari ni pamoja na kuvuja damu, maumivu makali ya tumbo, kuvimba uso au mikono, na homa kali. Tafadhali nenda kliniki mara moja."
        },
        {
            "lang": "Kiswahili",
            "code": "swa",
            "q": "Nifanye nini kumlinda mtoto mchanga dhidi ya malaria?",
            "a": "Hakikisha mtoto analala chini ya chandarua kilichotiwa dawa, na uweke mazingira ya nyumbani safi ili kuzuia mbu kuzaliana."
        },
        {
            "lang": "Luganda",
            "code": "lug",
            "q": "Obubonero obw'akabi mu lubuto kuki?",
            "a": "Obubonero obw'akabi mwe muli okutonnya omusaayi, omutwe okulumya ennyo, okuzimba ebigere, n'omusujja ogw'amaanyi. Genda mu ddwaaliro mangu."
        },
        {
            "lang": "Luganda",
            "code": "lug",
            "q": "Nnyonnyola ebikolebwa okutangira siriimu mu baana?",
            "a": "Okutangira okusiiga siriimu okuva ku maama okudda ku mwana (PMTCT) kusinga kukolebwa mu kukozesa eddagala lya ARVs mu budde bw'okusaba n'okuzaala."
        },
        {
            "lang": "Amharic",
            "code": "amh",
            "q": "በእርግዝና ወቅት አደገኛ ምልክቶች ምንድን ናቸው?",
            "a": "ከባድ የሆድ ህመም፣ ደም መፍሰስ፣ የእጅና የፊት እብጠት፣ እና ከፍተኛ ትኩሳት አደገኛ ምልክቶች ናቸው። በአስቸኳይ ወደ ህክምና ተቋም ይሂዱ።"
        },
        {
            "lang": "Amharic",
            "code": "amh",
            "q": "አዲስ የተወለደ ልጅን ከወባ በሽታ እንዴት መከላከል ይቻላል?",
            "a": "ህፃኑ በአልጋ አጎበር ውስጥ እንዲተኛ ማድረግ እና በአካባቢው ላይ የቆመ ውሃ እንዳይኖር በማድረግ የወባ ትንኝ መራባትን መከላከል ይቻላል።"
        },
        {
            "lang": "Akan",
            "code": "aka",
            "q": "Dɛn nkyerɛwee na ɛkyerɛ sɛ nyinsɛn mu bɔne bi reba?",
            "a": "Sɛ mogya fi wo mu, wo tirim yɛ wo yaw dendeenden, wo nan anaa w'anim hon, anaa wanya yaw kɛse a, kɔ ayaresabea ntɛm ara."
        },
        {
            "lang": "Akan",
            "code": "aka",
            "q": "Mɛyɛ dɛn asɔ m'abaa foforɔ no ho afiri asramm mmoawa ho?",
            "a": "Ma abofra no mda tumpan a yɛde nnuruyɛ ho mu, na tew wo fie ho na nsuo antan hɔ a mmoa bɛwo mu."
        }
    ]
    
    # Replicate templates to create a larger training set (e.g. 120 rows)
    train_rows = []
    for i in range(120):
        tpl = random.choice(qa_templates)
        # Add slight variations to queries/answers
        q_var = tpl["q"] + (" ?" if i % 2 == 0 else "")
        a_var = tpl["a"] + (" Asante." if tpl["code"] == "swa" and i % 3 == 0 else "")
        train_rows.append({
            "ID": f"TRAIN_{i:04d}",
            "Question": q_var,
            "Answer": a_var,
            "Language": tpl["lang"]
        })
        
    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    
    # Replicate templates for testing (e.g. 30 rows)
    test_rows = []
    for i in range(30):
        tpl = random.choice(qa_templates)
        test_rows.append({
            "ID": f"TEST_{i:04d}",
            "Question": tpl["q"] + (" !" if i % 2 == 0 else ""),
            "Language": tpl["lang"]
        })
        
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Generate SampleSubmission
    submission_rows = []
    for i in range(30):
        submission_rows.append({
            "ID": f"TEST_{i:04d}",
            "Answer": "Mfano wa jibu la afya."  # Standard placeholder
        })
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(os.path.join(output_dir, "SampleSubmission.csv"), index=False)
    
    print(f"Mock datasets created successfully in: {output_dir}")
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

if __name__ == "__main__":
    # Test generation
    generate_mock_datasets("/Users/pixeleyeblue/.gemini/antigravity/scratch/multilingual_health_qa/data/raw")
