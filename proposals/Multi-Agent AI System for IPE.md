## **Multi-Agent AI System for IPE in Healthcare**

### **1\) Project Title**

**Role-Specific Multi-Agent AI Simulator for Interprofessional Healthcare Collaboration**

### **2\) Background & Motivation**

Interprofessional collaboration is critical in real-world healthcare settings, but many students and early-career professionals struggle to understand how different roles think, prioritize tasks, communicate, and coordinate decisions. To address this, a multi-agent AI system has been developed that simulates role-specific perspectives across healthcare professions.

The current system is already useful for beginners, helping them close knowledge gaps and better understand responsibilities but experienced professionals find the responses too generic and overly repetitive, limiting real-world realism.

### **3\) Current System Overview**

The current solution is a **multi-agent AI system** that simulates interprofessional collaboration with these characteristics:

* **Goal:** Train beginners/students/early professionals to understand responsibilities across healthcare roles

* **Scenario Coverage:** Limited to a single domain (opioid scenario)

* **User Workflow:**

  1. User enters a healthcare scenario

  2. User selects professions (currently **6 roles**)

  3. The system generates **role-specific perspectives and actions** for each selected agent

* **Personalization:** Responses are customized based on user profile (credentials, experience, etc.)

### 

### **4\) Problem Statement**

While the system is effective for beginners, it currently suffers from limitations that reduce realism and professional quality:

1. Generic responses for experienced users

2. Repetitive content patterns from the LLM across roles

3. Lack of domain depth and citation-grounding for each profession

4. No structured behavior to refuse/redirect or refer scenarios to experts

5. Limited scalability: only 6 roles, and only one scenario domain

6. Development risk due to limited infrastructure: only production server exists (no test environment)

### **5\) Project Objectives**

This project proposes the next iteration of the system with the following key upgrades:

#### **Safe Referral & Delegation System**

Enable the system to detect when a user request should **not be answered directly**, and instead:

* refuse respectfully (when appropriate)

* recommend escalation/referral to the **relevant professional or expert**

* route queries to the most appropriate agent

#### **Response Quality, Decorum & Reduced Repetition**

Improve professionalism and realism by enforcing:

* stronger role-specific tone and structure

* reduced repeated phrasing maintain decorum and clinically appropriate language

* consistent formatting aligned to healthcare communication norms

#### 

#### **Add RAG-Based Knowledge System**

Integrate a Retrieval-Augmented Generation (RAG) layer to improve:

* factual grounding

* deeper role-specific expertise

* better decision explanations backed by curated knowledge sources

#### **Expand Roles and Scenario Coverage**

Move from a single opioid scenario to a broader simulator by:

* adding new professions over time

* supporting multiple healthcare scenarios

### **6\) Available Data & Resources**

**Current Data:**

* Curated article hyperlinks organized into folders by professional role  
   (a strong foundation for role-based retrieval in RAG)

**Infrastructure Constraint:**

* Only production server is available

* Need to implement a safe **test/staging environment** before major upgrades

### **7\) Proposed Technical Approach**

#### **A) Multi-Agent Orchestration Layer**

* One agent per healthcare profession

* Each agent has:

  * role identity and responsibilities

  * communication style constraints

  * role-specific knowledge access (via RAG)

#### **B) RAG Knowledge Layer (Role-Specific Retrieval)**

* Convert role-folder hyperlinks into an indexed knowledge base

* Retrieval filtered by profession or scenario type

#### **C) Referral & Routing System**

A decision layer that:

* detects unsafe / out-of-scope / high-risk queries

* triggers escalation recommendations

* routes the query to either appropriate role agent or referral response

#### **D) Output Guardrails**

Add structured output constraints such as:

* required response sections

* tone guidelines per profession

* duplication detection (to reduce repeated patterns)

### **8\) Evaluation Plan**

To ensure the system improves beyond beginner-level usefulness, evaluation will include:

**User-Level Quality Metrics**

* Beginner satisfaction score improvement

* Expert satisfaction improvement (less generic feedback)

**System Metrics**

* Reduced repetition rate across agents (qualitative and measurable similarity checks)

* Increased domain depth (more specific reasoning and grounded responses)

* Referral correctness rate (how often the system delegates appropriately)

**RAG Metrics**

* Citation coverage / grounding quality

* Context relevance for each profession

* Reduction in hallucinated clinical claims

### **9\) Deliverables**

By the end of the project, the expected deliverables include:

1. **Upgraded multi-agent AI simulator** with RAG integration

2. **Referral and escalation module**

3. Expanded professional roles (beyond initial 6\)

4. Support for multiple scenario templates (not opioid-only)

5. Test/staging server setup for safe iteration

6. Final demo and technical documentation (architecture and evaluation results)

### **10\) Weekly Timeline**

Week 1 — Jan 27, 2026

* Project kickoff, finalize scope & goals

* Confirm current system baseline and gaps

* Assign responsibilities & set success metrics

Week 2 — Feb 3, 2026

* Requirements gathering (wishlist to measurable requirements)

* Define user personas (beginner vs expert)

* Draft architecture (agents, RAG and referral)

Week 3 — Feb 10, 2026

* Data/resource review (role-based article folders)

* Decide RAG strategy

* Start building knowledge ingestion pipeline

Week 4 — Feb 17, 2026

* Build first working RAG prototype

* Run sample queries to validate role-depth improvement

* Add citation/grounding format for responses

Week 5 — Feb 24, 2026

* Implement “referral system” logic (when to not answer)

* Add escalation rules per role

* Draft refusal and referral templates (professional tone)

Week 6 — Mar 3, 2026

* Improve response decorum and reduce repetition

* Add structured output templates per profession

* Begin evaluation plan setup (test query set and metrics)

Week 7 — Mar 10, 2026

* Buffer week: optional progress / cleanup

* If working: stabilize demo and refine weak parts

Week 8 — Mar 17, 2026

* Present: problem, approach, architecture, early prototype

* Show early improvements (RAG depth and referral routing)

* Get feedback and define next sprint priorities

Week 9 — Mar 24, 2026

* Expand to more roles/professions (beyond 6\)

* Improve role differentiation and response realism

* Continue building scenario templates beyond opioids

Week 10 — Mar 31, 2026

* Add multi-scenario support (new workflows)

* Strengthen agent-to-agent coordination behavior

* Run more test cases across difficulty levels

Week 11 — Apr 7, 2026

* Evaluation sprint: test beginner vs expert usefulness

* Measure improvements (genericness ↓, depth ↑, referrals ↑)

* Log results and failure cases

Week 12 — Apr 14, 2026

* Fix weaknesses from evaluation

* Improve retrieval quality and formatting

* Finalize “safe fallback” behavior

Week 13 — Apr 21, 2026

* Final integration and deployment readiness

* Prepare slides, demo script and documentation

Week 14 — Apr 28, 2026

* Final demo (multi-agent, RAG, referral and multi-scenario)

* Submit journal/report, results and system overview

### **11\) Expected Impact**

This project will transform the system from a beginner-only learning tool into a more realistic and scalable training platform by:

* improving role authenticity and professionalism

* expanding scenario realism

* supporting deeper clinical reasoning using curated knowledge

* providing safer guidance through structured delegation/referral