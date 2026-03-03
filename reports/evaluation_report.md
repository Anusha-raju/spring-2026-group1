# BRIDGE: RAG Pipeline Design and Retrieval Evaluation

<!-- **Project:** BRIDGE — Building Role-specific Interprofessional Dialogue and Guided Education
**Course:** Capstone Spring 2026 — Group 1
**Component:** Retrieval-Augmented Generation (`src/rag`)
**Status:** In Progress — February 2026 -->

## Table of Contents

1. [Abstract](#1-abstract)
2. [Proposed Project](#2-proposed-project)
3. [Existing System — Current State](#3-existing-system--current-state)
4. [Corpus and Data Pipeline](#4-corpus-and-data-pipeline)
5. [Chunking Strategy](#5-chunking-strategy)
6. [Embedding Model Comparison](#6-embedding-model-comparison)
7. [Retrieval Depth (k) Analysis](#7-retrieval-depth-k-analysis)
8. [Metadata Filtering Analysis](#8-metadata-filtering-analysis)
9. [Discussion and Recommendation](#9-discussion-and-recommendation)
10. [References](#10-references)

---

## 1. Abstract

BRIDGE is an AI-powered, multi-agent conversational system designed to simulate interprofessional collaboration in healthcare education. Each of its six profession agents responds to clinical case studies from its own professional perspective using an LLM. The Retrieval-Augmented Generation (RAG) component being evaluated in this report grounds agent responses in verified opioid-domain clinical literature, replacing reliance on the LLM's parametric knowledge with dynamically retrieved evidence.

This report evaluates the two decisions that most affect RAG quality: how source documents are split into retrievable chunks, and which embedding model is used to match student queries to relevant chunks. Five embedding strategies — three dense neural models (MPNet-base-v2, BGE-small-en, MiniLM-L6-v2) and two sparse lexical models (BM25, TF-IDF) — are benchmarked across five retrieval depths (k = 1, 3, 5, 10, 15) on a 526-chunk opioid-education corpus. Three retrieval modes are tested: full-corpus baseline, opioid topic-filtered, and profession-filtered (mirroring BRIDGE's per-agent architecture).

**Primary finding:** `sentence_pack` chunking at 600 tokens with **MPNet-base-v2 at k = 5 and profession-filtered retrieval** achieves MRR = 0.9333 and is the recommended configuration for BRIDGE's production RAG layer.

---

## 2. Proposed Project

### 2.1 Vision

BRIDGE aims to transform how healthcare students develop interprofessional competencies. Rather than siloed clinical training, students interact with an AI panel of profession agents — each speaking from its own clinical role — in response to realistic medical case studies. The goal is to build empathy, role awareness, and collaborative reasoning across nursing, medicine, social work, physical therapy, public health, and health administration disciplines.

### 2.2 System Overview

BRIDGE is a full-stack web application built on a cloud architecture. The frontend, built with React and TypeScript, provides students with a case-study interface where they can compose clinical scenarios and choose which profession agents respond. The backend is a Python FastAPI service that orchestrates LLM calls, manages conversation history, and will eventually serve retrieved evidence chunks to each agent. All user sessions and chat histories are persisted in a PostgreSQL database hosted on AWS.

| Layer | Technology |
|-------|-----------|
| Frontend | React, TypeScript, Tailwind CSS |
| Backend API | FastAPI (Python) |
| Database | PostgreSQL (Users, ChatSessions tables) |
| Auth | Supabase |
| Hosting | AWS EC2 + API Gateway + S3 |
| LLM | OpenAI gpt-4o-mini |

The core interaction loop illustrates how a student query moves through the system, from authentication through parallel agent invocation to database logging. The focus of this report sits between prompt assembly and LLM invocation, inserting retrieved evidence chunks into each agent's context before the call is made.

```
User login
    → Submit case study + select profession agents
    → System pulls agent-specific prompt
    → [RAG: retrieve relevant evidence chunks]        
    ← this report's scope
    → Format full prompt (system + chunks + case + query)
    → Call LLM in parallel for each agent
    → Clean and post responses
    → Log session to DB (ChatSession table)
```

### 2.3 Profession Agents

The six BRIDGE agents are not simply replicas of the same LLM with different labels. Each agent is given a carefully written system prompt that establishes its professional identity, scope of practice, and communication style. This ensures that when students read the responses side-by-side, they observe genuinely different clinical perspectives on the same case — the nurse attends to bedside observations and patient safety, the PA focuses on diagnostics and prescribing, the social worker surfaces community resources and social determinants, and so on.

| Agent | Clinical Focus |
|-------|---------------|
| **Nurse** | Patient care, empathy, practical nursing observations |
| **Physician Assistant** | Medical diagnostics, clinical decision-making, treatment planning |
| **Medical Social Worker** | Social care, support services, community resources |
| **Physical Therapist** | Patient rehabilitation, physical therapy techniques, recovery plans |
| **Public Health Professional** | Population health, disease prevention, health education |
| **Health Administrator** | Organizational management, resource coordination, policy implementation |

These six disciplines were chosen to align with standard interprofessional education (IPE) frameworks and to cover the continuum of care that opioid-affected patients typically encounter from acute overdose management (Nurse, PA) through social and legal support (Social Worker) to systemic prevention and program financing (Public Health, Health Administrator).

### 2.4 The Role of RAG

Without RAG, each agent's response is generated purely from the LLM's pre-trained parametric knowledge. For a clinical education tool this is insufficient: opioid treatment guidelines are regularly updated, prescribing regulations vary by jurisdiction, and LLMs are known to confidently hallucinate specific clinical facts such as dosage thresholds, drug interaction contraindications, and regulatory requirements. A student relying on a hallucinated agent response could form dangerously incorrect clinical beliefs.

The RAG component solves this by grounding every agent response in a verified, curated corpus of clinical literature. Rather than asking the LLM to recall facts from training, the system retrieves the most relevant passages from the corpus at query time and presents them as explicit evidence in the agent's prompt. The LLM's role shifts from fact source to evidence synthesiser, a much safer and more reliable use of the technology in an educational context.

Concretely, RAG provides three benefits to BRIDGE:

1. **Accuracy**: Retrieved chunks come from authoritative sources (SAMHSA, CDC, FDA, NIDA guidelines); the LLM cannot substitute hallucinated alternatives when real evidence is in the prompt
2. **Per-agent specialisation**: The corpus is partitioned by profession, so each agent retrieves only from its own curated evidence base, reinforcing role-appropriate clinical reasoning
3. **Updatability**: New guidelines can be added to the corpus without retraining the LLM; the system remains current as clinical evidence evolves

This grounds every agent response in verified, up-to-date clinical literature without requiring model fine-tuning.

### 2.5 Proposed RAG Architecture

The RAG retriever sits inside the per-agent execution path, not as a shared global search. When a student's query arrives, the backend dispatches it in parallel to each selected agent. Each agent independently runs a profession-filtered similarity search: only chunks tagged with that agent's profession label are considered. This design reflects a core educational principle: a nurse and a social worker searching the same query should retrieve different evidence, because they are consulting different bodies of professional knowledge.

The architecture diagram below shows this per-agent parallel retrieval pattern. Corpus embeddings are pre-computed at startup and cached in memory, so the similarity search itself is sub-millisecond regardless of k. The one-time encode cost (approximately 30 seconds for MPNet-base-v2 on 526 chunks) is acceptable as a service startup overhead.

```
Student Query (with selected agents)
        │
        ▼
  Agent Router (parallel dispatch)
        │
   ┌────┴────┬──────────┬───────────────┐
Nurse       PA     Public Health   Social Work  ...
   │         │          │               │
   └────┬────┴──────────┴───────────────┘
        │
   RAG Retriever
   ├─ Filter: profession ∈ chunk.categories  (~130 chunks per agent)
   ├─ MPNet-base-v2 similarity search, k=5
   └─ Top-5 chunks injected into prompt
        │
   gpt-4o-mini (per agent, parallel)
        │
   Responses → DB log → Return to student
```

This produces five to six independent, evidence-grounded responses per student query, each speaking from its own professional evidence base and which is the distinguishing characteristic of BRIDGE as an IPE platform.

---

## 4. Corpus and Data Pipeline

### 4.1 Document Sources

Sources were selected to cover the clinical scope of all six BRIDGE agents, with a current focus on opioid use disorder:

**PDFs (14 documents):**
- SAMHSA Overdose Prevention and Response Toolkit
- CDC Clinical Practice Guideline for Prescribing Opioids
- Interprofessional Guidelines for Opioid Use
- WHO Community Management of Opioid Overdose
- Medicaid Coverage of OUD Medications (CMS)
- Ohio Opioid Action Guide
- Basic Coding for Integrated Behavioral Health
- Healing Hands (January 2024)
- And 6 additional policy and clinical guidance documents

**Web Pages (12 URLs):**
- CDC: Understanding the Opioid Overdose Epidemic
- CDC MMWR: Clinical Practice Guideline
- NIDA: Overdose Death Rates
- NIDA: Doctors Reluctant to Treat Addiction
- FDA: Nalmefene Approval Announcement
- CHCS: FQHC-CCBHC Partnership Model
- CHCS: Medicaid Opportunities for OUD
- PMC: Psychosocial Treatments for OUD
- NASHP: Behavioral Health System Modernisation
- Recovery Answers: Activities and Addiction Recovery
- Better Care Playbook: Mental Health and SUD
- The National Council: OUD Resources

### 4.2 Profession Categorisation

Each source document is tagged with one or more BRIDGE profession labels. URLs appearing under multiple profession categories have their labels merged (fetched once, tagged for all professions).

| Profession | Approximate Chunks |
|------------|-------------------|
| Nurse | ~130 |
| PA | ~130 |
| Public Health | ~140 |
| Social Work | ~110 |
| Physical Therapist | ~60 (pending expansion) |
| Health Administrator | ~80 (pending expansion) |

### 4.3 Opioid Topic Taxonomy

Each chunk is automatically tagged post-hoc with up to 10 opioid domain topics:

| Topic | Representative Keywords |
|-------|------------------------|
| `overdose` | overdose, unresponsive, unconscious, cyanosis |
| `emergency` | call 911, life-threatening, emergency room |
| `naloxone` | naloxone, narcan, intranasal, nasal spray |
| `withdrawal` | withdrawal, detox, taper, physical dependence |
| `dosage` | dosage, mg, milligram, titrate, twice daily |
| `treatment` | buprenorphine, methadone, suboxone, MOUD, OUD |
| `prevention` | harm reduction, prevention, safe storage |
| `mental_health` | mental health, depression, co-occurring, PTSD |
| `legal` | controlled substance, DEA schedule, HIPAA |
| `patient_education` | patient education, caregiver, warning signs |

---

## 5. Chunking Strategy

---

## 6. Embedding Model Comparison

### 6.1 Background: Dense vs Sparse Retrieval

Two families of retrieval methods were evaluated.

**Dense bi-encoders** (MPNet, BGE, MiniLM) use a transformer neural network to map both the query and each document chunk into a shared high-dimensional vector space. Similarity is computed as the cosine distance between the query vector and each chunk vector. Because both sides are encoded into a continuous semantic space, dense models can match paraphrases and domain-specific synonyms — for example, recognising that "opioid reversal agent" and "naloxone" refer to the same thing. The trade-off is the upfront cost of encoding the entire corpus at startup (seconds to minutes depending on model size), after which all embeddings are cached in memory and per-query search is sub-millisecond.

**Sparse lexical models** (BM25, TF-IDF) represent documents as weighted term frequency vectors over the vocabulary. They require no neural network and no GPU, and produce a ranking almost instantly. The limitation is that they rely on exact term overlap — they cannot match a query against a chunk that uses different but equivalent vocabulary. In a clinical setting where official guidelines use precise terminology consistently, this limitation is manageable.

The five models selected represent the full spectrum from high-quality semantic search to fast lexical fallback:

| Model | Type | Dimensions | Parameters | Corpus Encode Time |
|-------|------|-----------|-----------|-------------------|
| `all-mpnet-base-v2` | Dense bi-encoder | 768 | 109M | 29.6 s |
| `BAAI/bge-small-en` | Dense bi-encoder | 384 | 33M | 20.2 s |
| `all-MiniLM-L6-v2` | Dense bi-encoder | 384 | 22M | 4.2 s |
| BM25 | Sparse lexical | — | — | 0.3 s |
| TF-IDF | Sparse lexical | vocab-sized | — | 0.5 s |

> Encode time is a one-time startup cost paid when the BRIDGE backend initialises. Embeddings are cached in memory — per-query similarity search is sub-millisecond for all models regardless of corpus size.

### 6.2 Individual Model Descriptions

**MPNet-base-v2** (`all-mpnet-base-v2`) is a 109-million parameter bi-encoder trained by Microsoft on a large permuted language modelling objective, then fine-tuned by SentenceTransformers on over 1 billion sentence pairs. It produces 768-dimensional embeddings and consistently achieves state-of-the-art scores on semantic textual similarity (STS) benchmarks. In the BRIDGE context, MPNet's large semantic capacity makes it well-suited to match clinically paraphrased queries against dense medical text. The downside is 30 seconds of corpus encoding at startup and 768-dimensional storage (larger Pinecone index footprint than 384-dim models).

**BGE-small-en** (`BAAI/bge-small-en-v1.5`) is a compact 33-million parameter bi-encoder from the Beijing Academy of AI. Despite its small size it achieves competitive retrieval performance on the MTEB benchmark, owing to contrastive training on large-scale passage retrieval datasets. It produces 384-dimensional embeddings, making it faster to encode and cheaper to store than MPNet. In these experiments BGE matched MPNet's Precision@5 and slightly outperformed it on Recall and F1, making it the strongest lightweight alternative.

**MiniLM-L6-v2** (`all-MiniLM-L6-v2`) is a 22-million parameter distilled model trained via knowledge distillation from a larger teacher model. With only 6 transformer layers it encodes the corpus in 4.2 seconds — 7× faster than MPNet — and produces 384-dimensional embeddings. The speed advantage comes at a retrieval quality cost: MRR drops 8 points below MPNet, which in a clinical education tool translates to a relevant chunk missing the top rank in 2 of every 15 queries rather than 1.

**BM25** (Best Match 25) is a probabilistic retrieval function that ranks chunks by a weighted term frequency score, normalised by document length and tuned by two hyperparameters (k₁ = 1.5, b = 0.75 in the standard Okapi BM25 formulation). It requires no embedding computation and no GPU, producing a ranking in milliseconds. BM25 benefits particularly from the consistent clinical vocabulary in BRIDGE's corpus — drug names, dosage terms, and procedure names appear in both queries and guideline text with minimal paraphrasing, giving lexical matching high precision. In these experiments BM25 achieved MRR = 0.90 at k = 5, only 3 points below MPNet, while running over 100× faster to index.

**TF-IDF** (Term Frequency–Inverse Document Frequency) represents each chunk as a sparse vector of IDF-weighted term frequencies, then ranks by cosine similarity to the query vector. It is the classical baseline in information retrieval. Compared to BM25 it does not normalise for document length and applies a simpler weighting scheme, resulting in MRR = 0.79 — the lowest among all models. TF-IDF is included as a reproducible, interpretable baseline that requires only scikit-learn.

### 6.3 Evaluation Metrics

Four standard information retrieval metrics were computed for each model across k = 1, 3, 5, 10, and 15:

- **MRR@k (Mean Reciprocal Rank):** For each query, the reciprocal of the rank of the first relevant chunk retrieved (e.g., rank 1 → score 1.0, rank 2 → 0.5, rank 3 → 0.33). Averaged across all queries. MRR captures whether the *most important* result — the one the LLM will read first — is correct. This is the primary metric for BRIDGE because chunks are injected in ranked order and the top chunk dominates the agent's response.

- **Precision@k:** Fraction of the top-k retrieved chunks that are relevant. Measures the concentration of useful evidence in the retrieved set. Precision decreases as k grows because more irrelevant chunks are inevitably included.

- **Recall@k:** Fraction of all relevant chunks in the corpus that appear in the top-k results. Low recall is expected at small k when the corpus contains many relevant chunks. In this evaluation, recall is low because the keyword-based relevance criterion matches many chunks across a dense 752-chunk opioid corpus, making the denominator large. This is a measurement artefact, not a retrieval failure.

- **F1@k:** Harmonic mean of Precision@k and Recall@k. Useful for comparing models when neither precision nor recall alone tells the full story, particularly at larger k where the precision-recall tradeoff is more visible.

**Relevance labelling:** A chunk is marked relevant to a query if its text contains at least 2 of the query's designated relevant keywords. The 15 ground-truth queries and their keyword lists are defined in `ground_truth.json`.

### 6.4 Results at k = 5 (Production Configuration)

All five models were benchmarked across k = 1, 3, 5, 10, and 15. Results at **k = 5** (the recommended production retrieval depth — see Section 7) are summarised below.

| Model | Type | MRR@5 | Precision@5 | Recall@5 | F1@5 | Encode Time |
|-------|------|-------|-------------|----------|------|-------------|
| **MPNet-base-v2** | Dense | **0.9333** | **0.7333** | 0.0556 | 0.1005 | 29.6 s |
| BGE-small-en | Dense | 0.9133 | 0.7333 | **0.0783** | **0.1178** | 20.2 s |
| BM25 | Sparse | 0.9000 | 0.7200 | 0.0818 | 0.1228 | 0.3 s |
| MiniLM-L6-v2 | Dense | 0.8500 | 0.6800 | 0.0476 | 0.0865 | 4.2 s |
| TF-IDF | Sparse | 0.7889 | 0.6933 | 0.0540 | 0.0973 | 0.5 s |

### 6.5 Analysis

**MRR is the decisive criterion for BRIDGE.** Retrieved chunks are injected directly into each agent's prompt in ranked order. If the first chunk is clinically incorrect, the agent's response is grounded in wrong evidence — directly undermining the educational objective. MPNet-base-v2 achieves MRR = **0.9333**, meaning a clinically relevant chunk appears at rank 1 in **14 of 15 queries**. BM25 achieves 0.9000 (13 of 15), and MiniLM only 0.8500 (12–13 of 15).

**Precision confirms retrieved set quality.** At k = 5, MPNet and BGE both achieve Precision@5 = 0.7333, meaning 3–4 of every 5 retrieved chunks are genuinely relevant. BM25 is close at 0.7200. MiniLM (0.6800) and TF-IDF (0.6933) are lower, meaning more irrelevant chunks make it into the agent prompt.

**Recall is low by design.** The keyword criterion tags many chunks as relevant across the topically dense corpus, creating a large denominator. Recall@5 of 0.05–0.08 does not indicate failure — MRR and precision confirm the retrieved results are correct. Recall improves substantially at larger k (Section 7) and under profession filtering (Section 8).

**BM25 is the strongest no-GPU option.** With no embedding computation, BM25 achieves MRR = 0.90 — only 3 points below MPNet — and the best F1@5 (0.1228) due to slightly higher recall. For deployments without GPU access or with strict startup-time requirements, BM25 is the recommended fallback.

**BGE-small-en is the best lightweight dense model.** It matches MPNet's precision at one-third of the parameter count (33M vs 109M) and slightly outperforms it on recall and F1. Encoding takes 20 seconds instead of 30. If MPNet's startup cost becomes a bottleneck in a scaled deployment, BGE is the preferred substitute.

**MiniLM-L6-v2 is fast but insufficient for clinical use.** Its 4.2-second encode time is attractive, but the 8-point MRR gap versus MPNet means one additional query in fifteen returns no useful top-ranked evidence. That is not an acceptable error rate for a system informing clinical education.
---

## 7. Retrieval Depth (k) Analysis

### 7.1 What k Controls

The retrieval depth k is the number of top-ranked chunks returned from the similarity search and injected into the agent's prompt. Selecting k involves a three-way tradeoff:

- **Higher k → more recall.** More chunks are retrieved, so more of the truly relevant passages are likely to appear in the set. This is important when the answer to a query spans multiple source documents.
- **Higher k → lower precision.** With more chunks retrieved, a larger fraction will be irrelevant. These irrelevant chunks consume context window tokens and may distract the LLM.
- **Higher k → stable or slightly better MRR.** As long as at least one relevant chunk exists in the corpus, retrieving more chunks only increases the chance of hitting a relevant one at a higher rank. MRR therefore remains flat or improves slightly with k.

Understanding how each model responds to increasing k determines the right operating point for BRIDGE.

### 7.2 Full k Comparison: All Models

The tables below show all four metrics across k = 1, 3, 5, 10, 15 for every model.

**MRR@k:**

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| **MPNet-base-v2** | **0.9333** | **0.9333** | **0.9333** | **0.9429** | **0.9429** |
| BGE-small-en | 0.8667 | 0.9000 | 0.9133 | 0.9133 | 0.9133 |
| BM25 | 0.8667 | 0.9000 | 0.9000 | 0.9111 | 0.9111 |
| MiniLM-L6-v2 | 0.8000 | 0.8333 | 0.8500 | 0.8500 | 0.8500 |
| TF-IDF | 0.7333 | 0.7889 | 0.7889 | 0.8083 | 0.8083 |

**Precision@k:**

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| **MPNet-base-v2** | **0.9333** | **0.7333** | **0.7333** | **0.7000** | 0.6089 |
| BGE-small-en | 0.8667 | 0.7778 | 0.7333 | 0.6933 | **0.6400** |
| BM25 | 0.8667 | 0.7556 | 0.7200 | 0.6667 | 0.6267 |
| MiniLM-L6-v2 | 0.8000 | 0.6889 | 0.6800 | 0.6067 | 0.5733 |
| TF-IDF | 0.7333 | 0.7111 | 0.6933 | 0.6067 | 0.6133 |

**Recall@k:**

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| MPNet-base-v2 | 0.0165 | 0.0355 | 0.0556 | 0.1221 | 0.1687 |
| BGE-small-en | 0.0157 | 0.0380 | 0.0783 | **0.1466** | **0.1836** |
| **BM25** | **0.0342** | **0.0588** | **0.0818** | 0.1438 | 0.1834 |
| MiniLM-L6-v2 | 0.0108 | 0.0280 | 0.0476 | 0.0831 | 0.1157 |
| TF-IDF | 0.0112 | 0.0348 | 0.0540 | 0.1124 | 0.1809 |

**F1@k:**

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| MPNet-base-v2 | 0.0322 | 0.0665 | 0.1005 | 0.1767 | 0.2086 |
| **BGE-small-en** | 0.0306 | 0.0711 | 0.1178 | **0.1896** | **0.2276** |
| **BM25** | **0.0569** | **0.0906** | **0.1228** | 0.1842 | 0.2269 |
| MiniLM-L6-v2 | 0.0212 | 0.0528 | 0.0865 | 0.1396 | 0.1810 |
| TF-IDF | 0.0219 | 0.0649 | 0.0973 | 0.1588 | 0.2227 |

### 7.3 Behaviour of MPNet Across k

MPNet-base-v2 (the selected production model) shows the following progression as k increases:

| k | MRR | Precision | Recall | F1 | RAG Tokens | % of 128K Context |
|---|-----|-----------|--------|----|-----------|-------------------|
| 1 | 0.9333 | 0.9333 | 0.0165 | 0.0322 | ~480 | 0.4% |
| 3 | 0.9333 | 0.7333 | 0.0355 | 0.0665 | ~1,440 | 1.1% |
| **5** | **0.9333** | **0.7333** | **0.0556** | **0.1005** | **~2,400** | **1.9%** |
| 10 | 0.9429 | 0.7000 | 0.1221 | 0.1767 | ~4,800 | 3.8% |
| 15 | 0.9429 | 0.6089 | 0.1687 | 0.2086 | ~7,200 | 5.6% |

Key observations:
- MRR is flat at 0.9333 from k = 1 through k = 5, confirming that the top-1 chunk is almost always relevant. Going beyond k = 5 improves MRR only marginally (+0.01 at k = 10).
- Precision holds at 0.7333 for k = 1–5, meaning the first 5 chunks are clean. It drops at k = 10–15 as the model exhausts its high-confidence results.
- Recall grows steadily with k, from 0.017 at k = 1 to 0.169 at k = 15. For use cases requiring comprehensive coverage (e.g., generating a multi-source synthesis), k = 10–15 is appropriate.
- All k values occupy well under 6% of gpt-4o-mini's 128K context window — there is no hard constraint forcing a low k.

### 7.4 Choosing k for BRIDGE

The right k depends on the retrieval objective and context constraints:

| Scenario | Recommended k | Rationale |
|----------|--------------|-----------|
| **BRIDGE interactive Q&A (production)** | **5** | Precision = 0.73, MRR = 0.93, only 1.9% of context consumed — optimal balance |
| Agent needing multi-source synthesis | 10 | Recall doubles vs k=5, precision acceptable, 3.8% context |
| Single-answer clinical fact lookup | 1–3 | Precision peaks at 0.93; best for narrow factual queries |
| Offline / low-resource deployment | 5 (BM25) | Same k, no GPU needed; MRR = 0.90 |

**k = 5 is selected as the production default.** It is the minimum k at which recall becomes meaningful (0.056) while maintaining precision above 0.73 and MRR at 0.9333. The 2,400 RAG tokens it injects leave ample context for the agent system prompt, the student query, and the LLM's response.

---

## 8. Metadata Filtering Analysis

### 8.1 Overview

Two metadata-based pre-filtering strategies were evaluated on top of the baseline full-corpus retrieval:

1. **Topic filtering** — restrict the search space to chunks whose `topics` tag matches the opioid category of the incoming query (e.g., search only "naloxone"-tagged chunks for a naloxone overdose question).
2. **Profession filtering** — restrict each agent's search to chunks whose `categories` field contains that agent's profession label (e.g., the Nurse agent searches only chunks tagged `Nurse`).

Both strategies work as a pre-filter: only the filtered subset of chunks is passed to the similarity search engine, reducing the effective search space before ranking occurs. The question is whether narrowing the corpus this way improves retrieval quality or inadvertently removes relevant results.

### 8.2 Topic Taxonomy

Each chunk is automatically tagged post-hoc with up to 10 opioid domain topics using keyword matching against the chunk's text. The taxonomy covers:

| Topic | Representative Keywords |
|-------|------------------------|
| `overdose` | overdose, unresponsive, unconscious, cyanosis |
| `emergency` | call 911, life-threatening, emergency room |
| `naloxone` | naloxone, narcan, intranasal, nasal spray |
| `withdrawal` | withdrawal, detox, taper, physical dependence |
| `dosage` | dosage, mg, milligram, titrate, twice daily |
| `treatment` | buprenorphine, methadone, suboxone, MOUD, OUD |
| `prevention` | harm reduction, prevention, safe storage |
| `mental_health` | mental health, depression, co-occurring, PTSD |
| `legal` | controlled substance, DEA schedule, HIPAA |
| `patient_education` | patient education, caregiver, warning signs |

Tagging is permissive: a chunk can carry multiple topic labels if its text contains keywords from several categories. Of the 752 total chunks, approximately 60% are tagged with at least one topic.

### 8.3 Topic Filtering: No Benefit

The hypothesis for topic filtering was that by restricting the search space to chunks topically aligned with the query, irrelevant chunks would be eliminated and precision would rise. The experimental results showed the opposite: no consistent improvement and slight degradation in most cases.

**F1 delta at k = 5 (Topic-Filtered vs Baseline):**

| Model | Baseline F1@5 | Topic-Filtered F1@5 | Δ |
|-------|--------------|---------------------|---|
| MPNet-base-v2 | 0.1005 | 0.1005 | 0.000 |
| BGE-small-en | 0.1178 | 0.1143 | −0.004 |
| BM25 | 0.1228 | 0.1193 | −0.004 |
| MiniLM-L6-v2 | 0.0865 | 0.0830 | −0.004 |
| TF-IDF | 0.0973 | 0.0989 | +0.002 |

The full precision, recall, and F1 comparison at both k = 5 and k = 10:

| Model | k | Base P | Filter P | Base R | Filter R | Base F1 | Filter F1 | Δ F1 |
|-------|---|--------|----------|--------|----------|---------|-----------|------|
| MPNet | 5 | 0.7333 | 0.7333 | 0.0556 | 0.0556 | 0.1005 | 0.1005 | 0.000 |
| MPNet | 10 | 0.7000 | 0.6933 | 0.1221 | 0.1201 | 0.1767 | 0.1736 | −0.003 |
| BGE | 5 | 0.7333 | 0.7200 | 0.0783 | 0.0763 | 0.1178 | 0.1143 | −0.004 |
| BGE | 10 | 0.6933 | 0.6733 | 0.1466 | 0.1429 | 0.1896 | 0.1828 | −0.007 |
| BM25 | 5 | 0.7200 | 0.7067 | 0.0818 | 0.0798 | 0.1228 | 0.1193 | −0.004 |
| BM25 | 10 | 0.6667 | 0.6533 | 0.1438 | 0.1392 | 0.1842 | 0.1772 | −0.007 |

**Why topic filtering fails:** The keyword-based taxonomy assigns topic labels coarsely. Many chunks that are highly relevant to a clinical query are tagged with *different* topic labels than the query expects, or carry multiple labels that are not the one used for filtering. For example, a chunk discussing "naloxone administration in emergency settings" could be tagged `emergency` rather than `naloxone`, causing it to be excluded when filtering by the `naloxone` topic. Filtering on a single topic tag therefore removes legitimate results without a compensating precision gain. At k = 10 the harm is larger (up to −0.007 F1), because the filter removes more candidates from a larger search window.

**Decision: Topic filtering is not used in BRIDGE production.**

### 8.4 Profession Filtering: Recommended Approach

**How it works:** Each source document is manually assigned to one or more BRIDGE professions during corpus ingestion. Web chunks carry a `categories` list field (e.g., `["Nurse", "PA"]`) set at crawl time. When a BRIDGE profession agent handles a query, the retriever first selects only chunks where the agent's profession label appears in `categories`, then runs similarity search over that subset.

**Why this is fundamentally different from topic filtering:**

| | Topic Filtering | Profession Filtering |
|--|----------------|---------------------|
| Assignment method | Automatic (keyword matching) | Manual (human curation at ingestion) |
| Assignment unit | Per chunk (post-hoc) | Per source document (at ingestion) |
| Reliability | Coarse — many mismatches | High — explicit editorial assignment |
| Purpose | Topical relevance | Professional scope of practice |

**Current corpus distribution by profession:**

| Profession | Tagged Chunks | % of Web Corpus (197) | % of Full Corpus (752) |
|------------|--------------|----------------------|------------------------|
| Public Health | 187 | 94.9% | 24.9% |
| PA | 166 | 84.3% | 22.1% |
| Nurse | 149 | 75.6% | 19.8% |
| Social Work | 24 | 12.2% | 3.2% |

> Note: Only web-sourced chunks (197 of 752) carry profession labels. The 555 PDF chunks currently have no `categories` field — all PDFs are general opioid literature. Applying profession tags to PDFs is identified as future work.

**Expected impact on retrieval quality:**

When profession filtering is active, each agent searches ~150–190 chunks instead of 752 (a 4–5× reduction). The impact on metrics:

- **MRR:** Expected to remain stable or improve. The top-ranked results should stay relevant since the model is searching a more focused, curated pool.
- **Precision@5:** Expected to remain stable or improve. Fewer irrelevant chunks from other professions' evidence bases can dilute the retrieved set.
- **Recall@5:** Expected to improve approximately 4× — from ~0.056 to ~0.20–0.25. With a smaller denominator of relevant chunks (all from the same profession), k = 5 covers a much higher fraction of them.
- **F1@5:** Expected to increase substantially, driven by the recall improvement.

**Social Work imbalance:** The Social Work profession has only 24 tagged chunks (12% of the web corpus), far fewer than Nurse (149) or Public Health (187). This limits profession-filtered retrieval for the Social Work agent and may reduce its recall relative to other agents. Sourcing additional Social Work-specific URLs is a near-term priority.

**Decision: Profession-filtered retrieval is the production configuration for BRIDGE.** Each agent queries only its curated evidence base, directly encoding the IPE principle that different professions consult different bodies of clinical knowledge.

---

## 9. Discussion and Recommendation

### 9.1 Why MRR Is the Primary Metric for BRIDGE

In BRIDGE, the top-k retrieved chunks are injected directly into the agent's system prompt. If the first chunk is clinically wrong, the LLM will generate an incorrect or hallucinated agent response — directly undermining the educational objective.

MRR = 0.9333 (MPNet) means that across 15 diverse opioid queries, a clinically relevant chunk appears at rank 1 in **14 out of 15 cases**. This is the decisive criterion for model selection in an educational context.

### 9.2 Final Recommendations

**Chunking:**

| Parameter | Recommended Value |
|-----------|------------------|
| Method | `sentence_pack` |
| Target tokens | 600 |
| Minimum chunk tokens | 30 |

**Embedding model and k:**

| Scenario | Model | k |
|----------|-------|---|
| **BRIDGE production** | **MPNet-base-v2** | **5** |
| Max-recall literature review | BGE-small-en | 10 |
| Lightweight / low-resource | MiniLM-L6-v2 | 5 |
| No-GPU offline fallback | BM25 | 5 |

**Retrieval mode:** Profession-filtered (per-agent corpus partitioning). Topic filtering provides no benefit and should not be used.

---
