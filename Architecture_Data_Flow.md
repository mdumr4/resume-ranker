This is the absolute core of our Trail 6 architecture. Understanding why we use 2 queries reveals why this system will outperform a standard RAG pipeline.

Here is the exact flow of data and the "Semantic Dilution" problem we are solving.

### Why 2 Queries? (The "Semantic Dilution" Problem)
Imagine you have a single, massive query that says: 
> *"Looking for a Backend Engineer with Python. Must be in Bangalore expecting 20 LPA."*

When FAISS (Dense Vector Search) embeds this paragraph, it mathematically averages the meaning of all the words together into one dense dot. The heavy technical words ("Backend", "Python") get mixed with the logistical words ("Bangalore", "20 LPA").

Because the meaning is averaged out, FAISS might accidentally retrieve a **Frontend React Developer** who lives in Bangalore and wants 20 LPA. Why? Because that candidate matched 3 out of 4 concepts perfectly (Developer, Bangalore, 20 LPA). The model got confused by the logistical noise and forgot that *Python* was the non-negotiable part.

**The Solution:** We force the LLM to output 2 separate queries!
1. **Query A (Career/Skills):** *"Backend Engineer with Python."*
2. **Query B (Edu/Profile):** *"In Bangalore expecting 20 LPA."*

By running FAISS *twice* (once for Query A, once for Query B), we guarantee that the technical vector is 100% pure technical skill, with zero logistical noise diluting it. We then add the scores together (`Score A + Score B`). A candidate must score highly in *both* pure categories to survive!

---

### The Complete Data Flow

**1. The Recruiter Input**
The recruiter pastes a messy, unstructured Job Description into the system.

**2. The LLM JD Analyzer (Pre-Processing)**
The LLM reads the JD and outputs our strict JSON format containing `Query A (Career)` and `Query B (Logistics)`, perfectly formatted to match our candidate grammar.

**3. Stage 1: Retrieval & Trust Math (The Gatekeeper)**
- We feed `Query A` into FAISS and SPLADE.
- We feed `Query B` into FAISS and SPLADE.
- We merge the similarity scores together.
- **The Executioner:** We multiply that merged score by the candidate's `trust_score` (which we calculated in `template_builder.py`). If they have a massive career gap or fake dates, their score drops to near zero and they are eliminated.
- We take the **Top 100** survivors.

**4. Stage 2: The Cross-Encoder (The Perfect Reader)**
We pass the Top 100 candidates to the Cross-Encoder. 
Because the Cross-Encoder has perfect Attention memory and doesn't average out vectors, we can safely combine `Query A + Query B` into one prompt for it. It reads the queries alongside the Candidate's massive text string, word-by-word, and outputs the final, flawless 1-to-100 ranking!
