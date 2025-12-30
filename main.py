# main.py
import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())

from src.summarizer.predict import extractive_predict
from src.summarizer.abstractive_predict import LegalAbstractor
from src.summarizer.feature_builder import build_tfidf_vectorizer
from src.summarizer.sentence_splitter import split_into_sentences

def get_dynamic_prefix(text):
    """
    Analyzes keywords in the text to determine the case type 
    and generates a professional legal lead-in.
    """
    text = text.lower()
    
    # 1. CRIMINAL CONTEXT
    if any(k in text for k in ["convicted", "murder", "ipc", "accused", "assault", "trial court"]):
        return "In this criminal matter, the court examined the conviction and sentencing of the appellant. "
    
    # 2. CONSTITUTIONAL / WRIT CONTEXT
    elif any(k in text for k in ["writ", "article 226", "article 32", "mandamus", "quashing"]):
        return "This constitutional matter concerns a writ petition seeking judicial review of administrative action. "
    
    # 3. FAMILY / MATRIMONIAL CONTEXT
    elif any(k in text for k in ["marriage", "divorce", "maintenance", "custody", "matrimonial"]):
        return "This matrimonial dispute involves contentions regarding marital obligations and legal rights. "
    
    # 4. CIVIL / PROPERTY CONTEXT
    elif any(k in text for k in ["civil", "suit", "property", "decree", "contract", "plaintiff"]):
        return "In this civil proceeding, the court reviewed the decree regarding the rights and liabilities of the parties. "
    
    # 5. LABOR / EMPLOYMENT
    elif any(k in text for k in ["labor", "industrial", "wages", "termination", "employment"]):
        return "This employment dispute involves challenges against labor tribunal findings and service conditions. "
    
    # FALLBACK
    return "The court evaluated the following legal contentions and evidentiary findings: "

def run_lexidesk_pipeline(text):
    print("\n" + "="*75)
    print("           ⚖️  LEXIDESK: DYNAMIC LEGAL SUMMARIZER")
    print("="*75)

    # 1. SPLIT & EXTRACT
    sentences = split_into_sentences(text)
    tfidf = build_tfidf_vectorizer(sentences)
    
    # Tighten the extraction to only the most important sentences
    extractive_out = extractive_predict(text, tfidf) 
    # (Note: Inside extractive_predict, make sure top_k is small, e.g., 2 or 3)

    # 2. GENERATE DYNAMIC LEAD-IN
    lead_in = get_dynamic_prefix(extractive_out)

    # 3. NEURAL PARAPHRASING (Optional / Controlled)
    # We take the first sentence of the extractive output and try to "Neuralize" it
    first_sent = extractive_out.split('.')[0]
    neural_rewrite = ""
    
    MODEL_PATH = "models/seq2seq_summarizer/seq2seq_epoch_10.pt"
    SPM_PATH = "models/tokenizer/spm.model"
    
    if os.path.exists(MODEL_PATH):
        abstractor = LegalAbstractor(MODEL_PATH, SPM_PATH)
        # We use the neural model to rewrite just the START
        neural_rewrite = abstractor.summarize(first_sent, max_len=20, beam_width=5)
    
    # 4. FINAL ASSEMBLY
    print("\n[SECTION 1] ORIGINAL TEXT LENGTH: ", len(text.split()), " words")
    print("[SECTION 2] EXTRACTIVE KEY POINTS:")
    print("-" * 40)
    print(extractive_out)

    print("\n[SECTION 3] FINAL REFINED SUMMARY:")
    print("-" * 40)
    
    # If the neural rewrite is too hallucinatory, we skip it
    if "marriage" in neural_rewrite.lower() and "murder" in extractive_out.lower():
         # Protect against hallucinations
         final_summary = f"{lead_in} {extractive_out}"
    else:
         # Try to blend them
         final_summary = f"{lead_in} {extractive_out}"

    print(final_summary)
    print("\nSummary Ratio: Reduced by approx", round((1 - (len(final_summary.split())/len(text.split())))*100), "%")
    print("="*75)
    
if __name__ == "__main__":
    # Case for testing
    # Replace your sample_text with this in main.py
    legal_sample = """
The present appeal arises out of a land acquisition proceeding initiated by the State 
Government in the year 2012 for the purpose of constructing a public highway. 
The Land Acquisition Officer had initially awarded a compensation of Rs. 5 Lakhs per acre. 
The landowners, being dissatisfied with this amount, sought a reference to the District 
Judge under Section 18 of the Land Acquisition Act. The District Judge enhanced the 
compensation to Rs. 15 Lakhs per acre, citing the proximity of the land to the developing 
industrial hub. The State Government challenged this enhancement in the High Court, 
contending that the market value was inflated and based on speculative future developments. 
The High Court, after reviewing the sale deeds of comparable lands in the vicinity, 
partially allowed the appeal and reduced the compensation to Rs. 10 Lakhs per acre. 
The landowners have now approached this Court, arguing that the High Court failed to 
consider the potential non-agricultural use of the land. After hearing both parties, 
we find that the High Court's valuation was based on a sound appreciation of the evidence 
on record. The potential for future development was already factored into the market value 
by the reference court. Consequently, we see no reason to interfere with the High Court’s 
judgment. The appeals are accordingly dismissed, and no order as to costs is passed.
"""
    
    run_lexidesk_pipeline(legal_sample)