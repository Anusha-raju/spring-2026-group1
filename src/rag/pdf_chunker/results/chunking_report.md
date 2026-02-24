# RAG Chunking Strategy Evaluation Report

## 1. Overview

This report presents an evaluation of multiple chunking strategies for a Retrieval-Augmented Generation (RAG) pipeline. The goal of this study was to analyze how different chunk sizes and retrieval depths (Top-K) affect retrieval performance.

The evaluation was conducted across:

-   **Chunk sizes (target_tokens):** 3000, 5000, 7000, 10000
-   **Top-K values:** 3, 5, 7, 9
-   **Chunking methods:**
    -   Fixed
    -   Fixed with Overlap
    -   Recursive
    -   Sentence-based Packing
    -   Semantic-based

Performance was measured using:

-   **Hit Rate (Recall@K)**
-   **Average First Hit Rank**
-   Query Latency
-   Chunk Build Time
-   Index Build Time
-   Number of Chunks Generated



## 2. Overall Performance Summary

Across all 80 configurations (5 methods × 4 chunk sizes × 4 K values):

-   **Average Hit Rate:** 0.758
-   **Maximum Hit Rate:** 0.933
-   **Minimum Hit Rate:** 0.467
-   **Average First Hit Rank:** 1.85

On average, relevant documents were retrieved within the top two
positions, indicating strong retrieval performance.



## 3. Impact of Increasing Top-K

As K increased from 3 → 5 → 7 → 9:

-   Hit rate consistently improved.
-   Major gains occurred between K=3 and K=7.
-   Performance gains from K=7 to K=9 were marginal.

### Interpretation

Most relevant documents are ranked within the top 5 & 7 results.
Increasing K beyond 7 provides diminishing returns.



## 4. Impact of Chunk Size

Increasing chunk size resulted in:

-   Fewer total chunks
-   Reduced index build time
-   Increased semantic coverage per chunk

### Trade-offs


| Smaller Chunks | Larger Chunks |
|----------------|---------------|
| More precise | Broader context |
| More chunks | Fewer chunks |
| Higher indexing cost | Faster indexing |
| Possible context fragmentation | Better context retention |

The best-performing configurations were observed at larger chunk sizes
(7000, 10000 tokens).



## 5. Latency Analysis

-   **Average Query Latency:** \~0.09 ms\


## 6. Best Configuration

The strongest overall configuration observed:

-   **Chunk Size:** 7000
-   **Top-K:** 7
-   **Method:** Recursive

This configuration achieved the highest hit rates while maintaining
reasonable preprocessing time.


