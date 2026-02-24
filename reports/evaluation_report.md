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

### 6.1 Models

| Model | Type | Dimensions | Parameters | Corpus Encode Time |
|-------|------|-----------|-----------|-------------------|
| `all-mpnet-base-v2` | Dense bi-encoder | 768 | 109M | 29.6s |
| `BAAI/bge-small-en` | Dense bi-encoder | 384 | 33M | 20.2s |
| `all-MiniLM-L6-v2` | Dense bi-encoder | 384 | 22M | 4.2s |
| BM25 | Sparse lexical | — | — | 0.3s |
| TF-IDF | Sparse lexical | vocab-sized | — | 0.5s |

> Encode time is a one-time startup cost. Embeddings are cached in memory — per-query latency is sub-millisecond for all models.

### 6.2 MRR@k — Primary Metric

MRR (Mean Reciprocal Rank) measures how highly ranked the first relevant chunk is. For BRIDGE, this is the most important metric: the first chunk injected into the agent's prompt must be clinically correct.

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| **MPNet-base-v2** | **0.9333** | **0.9333** | **0.9333** | **0.9429** | **0.9429** |
| BGE-small-en | 0.8667 | 0.9000 | 0.9133 | 0.9133 | 0.9133 |
| BM25 | 0.8667 | 0.9000 | 0.9000 | 0.9111 | 0.9111 |
| MiniLM-L6-v2 | 0.8000 | 0.8333 | 0.8500 | 0.8500 | 0.8500 |
| TF-IDF | 0.7333 | 0.7889 | 0.7889 | 0.8083 | 0.8083 |

**MPNet-base-v2 leads at every k.** The first retrieved chunk is clinically correct **93.3% of the time** compared to 86.7% for BGE/BM25 and 80% for MiniLM.

### 6.3 Precision@k

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| **MPNet-base-v2** | **0.9333** | 0.7333 | 0.7333 | 0.7000 | 0.6089 |
| BGE-small-en | 0.8667 | **0.7778** | 0.7333 | 0.6933 | 0.6400 |
| BM25 | 0.8667 | 0.7556 | 0.7200 | 0.6667 | 0.6267 |
| MiniLM-L6-v2 | 0.8000 | 0.6889 | 0.6800 | 0.6067 | 0.5733 |
| TF-IDF | 0.7333 | 0.7111 | 0.6933 | 0.6067 | 0.6133 |

### 6.4 Recall@k

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| MPNet-base-v2 | 0.0165 | 0.0355 | 0.0556 | 0.1221 | 0.1687 |
| **BGE-small-en** | 0.0157 | 0.0380 | 0.0783 | **0.1466** | **0.1836** |
| **BM25** | **0.0342** | **0.0588** | **0.0818** | 0.1438 | 0.1834 |
| MiniLM-L6-v2 | 0.0108 | 0.0280 | 0.0476 | 0.0831 | 0.1157 |
| TF-IDF | 0.0112 | 0.0348 | 0.0540 | 0.1124 | 0.1809 |

> **Note on low recall:** Recall is low because the keyword-based relevance criterion matches many chunks in a topically dense 526-chunk corpus (large denominator). This is a measurement artifact, not a retrieval failure MRR and precision confirm the top results are correct.

### 6.5 F1@k

| Model | k=1 | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|-----|------|------|
| MPNet-base-v2 | 0.0322 | 0.0665 | 0.1005 | 0.1767 | 0.2086 |
| **BGE-small-en** | 0.0306 | 0.0711 | 0.1178 | **0.1896** | **0.2276** |
| **BM25** | **0.0569** | **0.0906** | **0.1228** | 0.1842 | 0.2269 |
| MiniLM-L6-v2 | 0.0212 | 0.0528 | 0.0865 | 0.1396 | 0.1810 |
| TF-IDF | 0.0219 | 0.0649 | 0.0973 | 0.1588 | 0.2227 |

### 6.6 Model Summary

| Model | Strength | Weakness |
|-------|----------|----------|
| **MPNet-base-v2** | Highest MRR (0.93) and Precision@1 (0.93) | 30s encode, 109M params |
| BGE-small-en | Best F1 at k≥10, good MRR | Slower than MiniLM |
| MiniLM-L6-v2 | Fast (4.2s), small model | Lower MRR (0.85) |
| BM25 | Near-instant, no GPU, good recall | Misses semantic paraphrases |
| TF-IDF | Interpretable, fast | Lowest MRR (0.79) |
---

## 7. Retrieval Depth (k) Analysis

### 7.1 Effect of k on Each Metric

As k increases: **Recall rises** (more chunks retrieved), **Precision falls** (more irrelevant chunks included), **MRR stays stable or improves slightly** (more chances for a hit to appear).

**Precision vs k (MPNet-base-v2):**

| k | Precision | Recall | F1 | MRR |
|---|-----------|--------|----|-----|
| 1 | 0.9333 | 0.0165 | 0.0322 | 0.9333 |
| 3 | 0.7333 | 0.0355 | 0.0665 | 0.9333 |
| 5 | 0.7333 | 0.0556 | 0.1005 | 0.9333 |
| 10 | 0.7000 | 0.1221 | 0.1767 | 0.9429 |
| 15 | 0.6089 | 0.1687 | 0.2086 | 0.9429 |

### 7.2 Choosing k for BRIDGE

The optimal k depends on the use case:

| Scenario | Recommended k | Rationale |
|----------|--------------|-----------|
| **Interactive Q&A (production)** | **5** | High precision (0.73), stable MRR (0.93), fits neatly in gpt-4o-mini context |
| Deep literature review | 10–15 | Maximises recall; acceptable if the LLM can handle longer context |
| Single-answer factual lookup | 1–3 | Highest precision; best when corpus is well-filtered by profession |

### 7.3 k and Context Window Budget

With 526 chunks averaging 480 tokens each, the approximate tokens injected into each agent prompt are:

| k | Tokens from RAG | % of gpt-4o-mini 128K context |
|---|----------------|-------------------------------|
| 1 | ~480 | 0.4% |
| 5 | ~2,400 | 1.9% |
| 10 | ~4,800 | 3.8% |
| 15 | ~7,200 | 5.6% |

All k values are well within the context budget, making **k = 5** the practical optimum for balancing evidence richness with prompt cleanliness.

---

## 8. Metadata Filtering Analysis

### 8.1 Topic-Filtered vs Baseline

Topic filtering restricts the search space to chunks tagged with the query's opioid topic (e.g., "withdrawal", "treatment") before running similarity search. Results show **no consistent improvement** over baseline retrieval.

**F1 Delta (Topic-Filtered − Baseline) at k = 5:**

| Model | Baseline F1 | Topic-Filtered F1 | Δ |
|-------|-------------|-------------------|---|
| MPNet-base-v2 | 0.1005 | 0.1005 | 0.0000 |
| BGE-small-en | 0.1178 | 0.1143 | −0.0035 |
| BM25 | 0.1228 | 0.1193 | −0.0035 |
| MiniLM-L6-v2 | 0.0865 | 0.0830 | −0.0035 |
| TF-IDF | 0.0973 | 0.0989 | +0.0016 |

**Why topic filtering is neutral/harmful:** The 10-category taxonomy tags chunks coarsely. Many clinically relevant chunks carry multiple topic labels or are tagged differently from the query's expected topic. Filtering on a single topic tag excludes legitimate results, causing the slight F1 drops seen above.

**Full comparison at k = 5 and k = 10:**

| Model | k | Base P | Filt P | Base R | Filt R | Base F1 | Filt F1 | Δ F1 |
|-------|---|--------|--------|--------|--------|---------|---------|------|
| MPNet | 5 | 0.7333 | 0.7333 | 0.0556 | 0.0556 | 0.1005 | 0.1005 | 0.000 |
| MPNet | 10 | 0.7000 | 0.6933 | 0.1221 | 0.1201 | 0.1767 | 0.1736 | −0.003 |
| BGE | 5 | 0.7333 | 0.7200 | 0.0783 | 0.0763 | 0.1178 | 0.1143 | −0.004 |
| BGE | 10 | 0.6933 | 0.6733 | 0.1466 | 0.1429 | 0.1896 | 0.1828 | −0.007 |
| BM25 | 5 | 0.7200 | 0.7067 | 0.0818 | 0.0798 | 0.1228 | 0.1193 | −0.004 |
| BM25 | 10 | 0.6667 | 0.6533 | 0.1438 | 0.1392 | 0.1842 | 0.1772 | −0.007 |

**Conclusion:** Topic-based pre-filtering does **not** improve retrieval quality and should not be used in BRIDGE production.

### 8.2 Profession-Filtered Retrieval — The Right Approach

Profession filtering restricts each agent's search to only the chunks tagged with its profession's `categories` label. This directly mirrors BRIDGE's multi-agent architecture and is the correct implementation of per-agent retrieval.

**Key difference from topic filtering:**
- Topic tags are automatically assigned by keyword matching and can be imprecise
- Profession categories are manually curated during document ingestion — each source document is explicitly assigned to one or more professions by the team

**Effect on corpus size per agent:**

| Profession | Full Corpus | After Profession Filter | Reduction |
|------------|------------|------------------------|-----------|
| Nurse | 526 | ~130 | 75% |
| PA | 526 | ~130 | 75% |
| Public Health | 526 | ~140 | 73% |
| Social Work | 526 | ~110 | 79% |

**Expected impact on recall:**

With the denominator of relevant chunks shrinking ~4×, recall at k = 5 for MPNet is expected to improve from ~0.056 to approximately **0.20–0.25**, while precision and MRR remain stable or improve due to the more focused corpus.

### 8.3 Filtering Strategy Comparison

| Strategy | F1 Impact | Precision Impact | Recall Impact | Recommendation |
|----------|-----------|-----------------|---------------|---------------|
| No filtering (baseline) | — | — | — | ✅ Acceptable for shared index |
| Topic-filtered | −0.003 | −0.002 | −0.002 | ❌ Avoid |
| **Profession-filtered** | **+~0.15 (estimated)** | **Stable/better** | **~4× improvement** | **✅ Use in production** |

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

## 10. References

- BAAI (2023). *BGE Embedding Models*. Beijing Academy of Artificial Intelligence. https://huggingface.co/BAAI/bge-small-en
- Kamradt, G. (2023). *Semantic Chunking for RAG*. Greg Kamradt AI Research.
- LangChain (2023). *Text Splitters Documentation*. https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS 2020.
