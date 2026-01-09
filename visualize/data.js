const DATA = {
  "Architecture Design & Modification": {
    "canonical_name": "Architecture Design & Modification",
    "member_phrases": [
      "Architectural Design Choice",
      "Architectural Design Choice / Finding",
      "Architectural Enhancement",
      "Architectural Modification",
      "Architecture Design",
      "Architecture Modification",
      "Attention Mechanism Design",
      "Model Architecture Simplification",
      "Novel Architecture",
      "Novel Architecture / Oracle Factory"
    ],
    "contributions": [
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A message-passing graph-based neural network architecture using a bipartite graph representation of CNF formulas (clause and literal nodes) and a modified attention mechanism for selective message aggregation.",
        "problem_addressed": "Capturing the structural properties of SAT formulas (e.g., invariance to renaming, clause/literal permutation) and enabling more effective information propagation than simple averaging.",
        "evidence_location": "Section 3 Architecture",
        "comment": "Builds upon NeuroSAT but is adapted for guiding a search algorithm rather than direct SAT prediction.",
        "paper_name": "Neural heuristics for SAT solving",
        "source_file": "2005.13406_Neural_heuristics_for_SAT_solving.md",
        "contribution_index": 1,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "Designs a novel Graph Transformer architecture combining a GNN subnet with a transformer subnet using lightweight Graph Self-Attention (GSA) and Local Self-Attention (LSA) blocks.",
        "problem_addressed": "Creates a compact and robust model suitable for large-scale graphs by explicitly incorporating topological information and achieving linear memory complexity.",
        "evidence_location": "Section 4.2.1",
        "comment": "A key technical innovation for model design.",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 3,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "Two GNN models (MS-NSFG and MS-ESFG) based on different factor graph representations (node-splitting and edge-splitting) to predict solutions for MaxSAT in an end-to-end fashion.",
        "problem_addressed": "Lack of end-to-end GNN models for predicting solutions to the MaxSAT problem, focusing on solution quality rather than just satisfiability.",
        "evidence_location": "Section 3: GNN Models for MaxSAT",
        "comment": "These are the first GNN models designed to directly output variable assignments for MaxSAT.",
        "paper_name": "Can Graph Neural Networks Learn to Solve MaxSAT Pr",
        "source_file": "2111.07568_Can_Graph_Neural_Networks_Learn_to_Solve_MaxSAT_Pr.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "BPGAT, a Graph Neural Network architecture that integrates a Graph Attention Network (GAT)-style multi-head attention mechanism into the Belief Propagation Neural Network (BPNN) framework for approximate #SAT solving.",
        "problem_addressed": "Improving the generalization capabilities and performance of neural belief propagation for propositional model counting (#SAT).",
        "evidence_location": "Section 3.2: Neural Belief Propagation with Attention",
        "comment": "",
        "paper_name": "Graph Neural Networks for Propositional Model Coun",
        "source_file": "2205.04423_Graph_Neural_Networks_for_Propositional_Model_Coun.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Architectural Enhancement",
        "specific_innovation": "Replacing the fixed scalar damping parameter in BP with a learned MLP operator (Δ) for factor-to-variable messages, while keeping a fixed damping for variable-to-factor messages.",
        "problem_addressed": "Improving the convergence and stability of the neural message passing scheme, as empirically shown to yield better performance.",
        "evidence_location": "Section 3.1 (paragraph on damping) and Section C.1: Learning the Damping Parameter",
        "comment": "",
        "paper_name": "Graph Neural Networks for Propositional Model Coun",
        "source_file": "2205.04423_Graph_Neural_Networks_for_Propositional_Model_Coun.md",
        "contribution_index": 6,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A universal Graph Neural Network architecture that can be trained as an end-to-end search heuristic for any Constraint Satisfaction Problem (CSP), handling variable arity, relations, and domain size.",
        "problem_addressed": "Existing GNN-based approaches are problem-specific, requiring specialized graph reductions and architecture modifications for each CSP.",
        "evidence_location": "Abstract, Section 1, Section 4.1",
        "comment": "Designed as a recurrent heterogeneous GNN with distinct functions for each vertex type (variables, values, constraints) in the constraint value graph.",
        "paper_name": "One Model Any CSP Graph Neural Networks as Fast Gl",
        "source_file": "2208.10227_One_Model_Any_CSP_Graph_Neural_Networks_as_Fast_Gl.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "AsymSAT, a GNN-based SAT solver that integrates a recurrent neural network (RNN) layer for sequential, dependent variable assignment prediction.",
        "problem_addressed": "Existing GNN-based SAT solvers (e.g., NeuroSAT, DG-DAGRNN) make concurrent, independent predictions, which fails for symmetric problems requiring asymmetric solutions (e.g., XOR).",
        "evidence_location": "Section 4: Our Methods",
        "comment": "",
        "paper_name": "Addressing Variable Dependency in GNN-based SAT So",
        "source_file": "2304.08738_Addressing_Variable_Dependency_in_GNN-based_SAT_So.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Architectural Modification",
        "specific_innovation": "Use of two separate GRUs (GRU_init and GRU_f) in the forward message-passing passes to avoid a projection step that could cause information loss.",
        "problem_addressed": "Improves information retention during the graph embedding process compared to the prior DG-DAGRNN architecture.",
        "evidence_location": "Section 4.2: The Proposed GNN Architecture",
        "comment": "",
        "paper_name": "Addressing Variable Dependency in GNN-based SAT So",
        "source_file": "2304.08738_Addressing_Variable_Dependency_in_GNN-based_SAT_So.md",
        "contribution_index": 4,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A one-round GNN model that separates node embeddings into distinct functional and structural components and propagates them through dedicated aggregation streams.",
        "problem_addressed": "Inefficient multi-round message-passing in prior GNNs for circuits and the inability to capture structural correlations like reconvergence due to homogeneous PI embeddings.",
        "evidence_location": "Section III-D",
        "comment": "",
        "paper_name": "DeepGate2 Functionality-Aware Circuit Representati",
        "source_file": "2305.16373_DeepGate2_Functionality-Aware_Circuit_Representati.md",
        "contribution_index": 1,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture / Oracle Factory",
        "specific_innovation": "Using a Graph Neural Network (specifically an Interaction Network) as a parametrized oracle factory that maps a SAT instance to an instance-specific sampling distribution.",
        "problem_addressed": "Learns to generate high-quality oracles from the distribution of SAT instances, enabling the construction of application-specific solvers.",
        "evidence_location": "Section 2.5, Graph Neural Networks as oracle factories",
        "comment": "The GNN architecture and the method for defining its output (Eq. 16-19) are key implementation details.",
        "paper_name": "Using deep learning to construct stochastic local ",
        "source_file": "2309.11452_Using_deep_learning_to_construct_stochastic_local_.md",
        "contribution_index": 2,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A bipartite Graph Attention Network (Bip-GNN) backbone with separate clause-wise and variable-wise message-passing steps, incorporating attention mechanisms to model dependencies between variables and clauses in the transformed bipartite graphs.",
        "problem_addressed": "Traditional GNN message-passing schemes are not directly suitable for bipartite graphs with two distinct node types (variables and clauses).",
        "evidence_location": "Section 3.3 Learning with Bipartite Graph Attention Networks",
        "comment": "This architecture is specifically designed to handle the structure derived from the Max-SAT representation.",
        "paper_name": "A unified pre-training and adaptation framework fo",
        "source_file": "2312.11547_A_unified_pre-training_and_adaptation_framework_fo.md",
        "contribution_index": 1,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "The paper employs a Weighted Graph Convolutional Network (WGCN) coupled with a specific node feature initialization and a Flip function that concatenates features of literal nodes with their negated counterparts.",
        "problem_addressed": "To effectively extract structural features from the proposed WLIG representation for the task of UNSAT-core variable prediction.",
        "evidence_location": "3.3 Neural Network Model",
        "comment": "",
        "paper_name": "IB-Net Initial Branch Network for Variable Decisio",
        "source_file": "2403.03517_IB-Net_Initial_Branch_Network_for_Variable_Decisio.md",
        "contribution_index": 1,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A heterogeneous GNN model is proposed, tailored to the tripartite literal-clause graph representation of SAT instances, using edge-type-dependent convolutions.",
        "problem_addressed": "Standard GNNs do not incorporate domain-specific knowledge and are not specialized for the heterogeneous structure of SAT graphs.",
        "evidence_location": "Section 3.3 Model",
        "comment": "",
        "paper_name": "GraSS Combining Graph Neural Networks with Expert ",
        "source_file": "2405.11024_GraSS_Combining_Graph_Neural_Networks_with_Expert_.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Model Architecture Simplification",
        "specific_innovation": "Significant simplification of the NeuroSAT architecture by removing MLP message processors, replacing the voting MLP with a linear layer, removing LayerNorm, and reducing hidden state dimensions.",
        "problem_addressed": "Reducing model complexity and training time without sacrificing accuracy.",
        "evidence_location": "Section 5.1.2: Model architecture",
        "comment": "",
        "paper_name": "Understanding GNNs for Boolean Satisfiability thro",
        "source_file": "2408.15418_Understanding_GNNs_for_Boolean_Satisfiability_thro.md",
        "contribution_index": 3,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Architectural Design Choice",
        "specific_innovation": "Promotion of the Variable-Clause Graph (VCG) representation with RNN updates over the Literal-Clause Graph (LCG), due to its computational efficiency (half the variable nodes) and strong performance, especially for assignment prediction.",
        "problem_addressed": "Addresses the computational inefficiency of the LCG representation, which requires an expensive 'Flip' operation, making VCG more suitable for scaling to larger formulas.",
        "evidence_location": "Section 4.1 (Data Representation and Graph Structure), Section 4.2 (Architecture Variants), Section 5.2.1 (Comparison of Graph Representations...)",
        "comment": "Identified as a key design choice from their experimental comparison (Table 2). VCG+RNN+Assignment is selected as the most effective configuration.",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 3,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Architectural Design Choice / Finding",
        "specific_innovation": "Finding that time-step conditioning is unnecessary for the GNN function approximator within their diffusion model for SAT (i.e., $f_θ(x_t)$ works as well or better than $f_θ(x_t, t)$).",
        "problem_addressed": "Simplifies the diffusion model architecture by removing an unnecessary component, potentially improving performance.",
        "evidence_location": "Section 5.4.1 (Connection to Assignment Prediction Training)",
        "comment": "Presented as an interesting finding that contradicts standard diffusion model design. They note it was concurrently discovered by others.",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 7,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A hypergraph convolutional network (HyperGCN) integrated with a transformer module featuring a novel cross-attention mechanism between positive and negative literal node features.",
        "problem_addressed": "Capturing the logical interplay and mutual exclusivity between positive and negative literals of the same variable, which is central to the Weighted MaxSAT problem.",
        "evidence_location": "3.2 Neural Network Architecture",
        "comment": "The cross-attention mechanism allows complementary literal nodes to dynamically incorporate features from each other.",
        "paper_name": "HyperSAT Unsupervised Hypergraph Neural Networks f",
        "source_file": "2504.11885_HyperSAT_Unsupervised_Hypergraph_Neural_Networks_f.md",
        "contribution_index": 1,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Architecture Modification",
        "specific_innovation": "A modified HyperGCN update rule that removes the diagonal elements of the propagation matrix to focus computation on the influence of adjacent nodes.",
        "problem_addressed": "Better aligning node representation updates with the higher-order adjacency structure of the hypergraph.",
        "evidence_location": "3.2.1 Hypergraph Convolutional Networks",
        "comment": "This is an adjustment to the traditional HGNN updating rule (Eq. 1 vs Eq. 2).",
        "paper_name": "HyperSAT Unsupervised Hypergraph Neural Networks f",
        "source_file": "2504.11885_HyperSAT_Unsupervised_Hypergraph_Neural_Networks_f.md",
        "contribution_index": 2,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Architecture Design",
        "specific_innovation": "Designs a one-shot GNN-based policy that outputs a Log-Normal distribution for variable weights and a Bernoulli distribution for variable polarities in a single forward pass.",
        "problem_addressed": "Avoids the computational bottleneck of running a GNN for each branching decision, making ML guidance practical.",
        "evidence_location": "Section 2.2, Section 2.3",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 2,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A composite architecture integrating a Graph Neural Network (GNN) with a Large Language Model (LLM) for structure-aware algorithm discovery.",
        "problem_addressed": "Existing LLM-based CO approaches neglect the inherent topological structure of CO problems, leading to suboptimality and iterative inefficiency.",
        "evidence_location": "4 Proposed Solution",
        "comment": "",
        "paper_name": "STRCMP Integrating Graph Structural Priors with La",
        "source_file": "2506.11057_STRCMP_Integrating_Graph_Structural_Priors_with_La.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "Augments the circuit DAG with virtual AND and DIV gates to directly model joint and conditional probabilities within the graph structure.",
        "problem_addressed": "Enables accurate, graph-native modeling of conditional probabilities for circuit SAT solving, avoiding error amplification from direct arithmetic division.",
        "evidence_location": "3.2 Graph Construction",
        "comment": "",
        "paper_name": "Circuit-Aware SAT Solving Guiding CDCL via Conditi",
        "source_file": "2508.04235_Circuit-Aware_SAT_Solving_Guiding_CDCL_via_Conditi.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Novel Architecture",
        "specific_innovation": "A neural network framework (Attn-JGNN) that combines tree decomposition with a Graph Attention Network (GAT) to implement the Iterative Join-Graph Propagation (IJGP) algorithm in latent space for solving #SAT problems.",
        "problem_addressed": "Addresses the limitations of belief propagation (BP) on cyclic graph structures and the high computational overhead of global attention mechanisms in neural #SAT solvers.",
        "evidence_location": "Sections 1, 3.1",
        "comment": "The framework is the core integration of IJGP, tree decomposition, and GNNs.",
        "paper_name": "Attn-JGNN Attention Enhanced Join-Graph Neural Net",
        "source_file": "2510.15583_Attn-JGNN_Attention_Enhanced_Join-Graph_Neural_Net.md",
        "contribution_index": 0,
        "cluster_name": "Architecture Design & Modification"
      },
      {
        "contribution_type": "Attention Mechanism Design",
        "specific_innovation": "A hierarchical attention mechanism employing two distinct GAT layers: one for intra-cluster (local) message passing and another for inter-cluster (global) message passing across the join-graph.",
        "problem_addressed": "Efficiently captures both fine-grained variable-clause interactions and macro-constraints across clusters while reducing computational complexity from O((n+m)²) to O(kw²).",
        "evidence_location": "Section 3.2 (Hierarchical attention mechanism)",
        "comment": "This design directly tackles the scalability issue of global attention.",
        "paper_name": "Attn-JGNN Attention Enhanced Join-Graph Neural Net",
        "source_file": "2510.15583_Attn-JGNN_Attention_Enhanced_Join-Graph_Neural_Net.md",
        "contribution_index": 2,
        "cluster_name": "Architecture Design & Modification"
      }
    ]
  },
  "Algorithm & Approach Development": {
    "canonical_name": "Algorithm & Approach Development",
    "member_phrases": [
      "Novel Algorithm",
      "Novel Approach/Algorithm",
      "Novel algorithm",
      "Global Search Procedure",
      "Sampling Procedure",
      "Entropy-Based Branching Calculation",
      "GNN-Based Branching Heuristic",
      "Neural Heuristic Design",
      "Novel Hardness Heuristic",
      "Structure-Aware Code Generation",
      "Evolutionary Code Refinement Integration"
    ],
    "contributions": [
      {
        "contribution_type": "Neural Heuristic Design",
        "specific_innovation": "A hybrid guidance algorithm that uses the neural model's sat probability prediction to switch to a classical heuristic (JW-OS) when confidence is low.",
        "problem_addressed": "Mitigating potential weaknesses or low-confidence predictions of the pure neural heuristic to improve robustness and performance.",
        "evidence_location": "Section 4 Experimental Results (Experiment 2)",
        "comment": "",
        "paper_name": "Neural heuristics for SAT solving",
        "source_file": "2005.13406_Neural_heuristics_for_SAT_solving.md",
        "contribution_index": 3,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Novel Approach/Algorithm",
        "specific_innovation": "Introduces NeuroBack, a novel approach that uses a single offline GNN inference to predict variable phases for improving the phase selection heuristic in CDCL SAT solvers.",
        "problem_addressed": "Makes neural improvements to SAT solving practical by eliminating the need for frequent, GPU-intensive online inferences during solving.",
        "evidence_location": "Abstract, Section 1, Section 4",
        "comment": "Core contribution of the paper.",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 0,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Novel Algorithm",
        "specific_innovation": "A new distributed local algorithm for MaxSAT that provides a theoretical performance guarantee (1/2 approximation) and aligns with GNN structure.",
        "problem_addressed": "The need for a polynomial-time algorithm that aligns with GNN structure to provide a theoretical explanation for GNN performance on MaxSAT.",
        "evidence_location": "Section 4: Distributed Local Algorithm (Algorithm 1)",
        "comment": "",
        "paper_name": "Can Graph Neural Networks Learn to Solve MaxSAT Pr",
        "source_file": "2111.07568_Can_Graph_Neural_Networks_Learn_to_Solve_MaxSAT_Pr.md",
        "contribution_index": 2,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Global Search Procedure",
        "specific_innovation": "A global search procedure that samples a new full assignment in each iteration, allowing any number of variables to change values simultaneously, unlike local search heuristics.",
        "problem_addressed": "GNN-based local search heuristics scale poorly on large instances because one GNN forward pass is slower than a local search step; global search leverages GNN parallelism to refine the whole solution in parallel.",
        "evidence_location": "Abstract, Section 1, Section 4.2",
        "comment": "This is a key design choice distinguishing ANYCSP from prior neural local search methods (e.g., ECO-DQN, RLSAT).",
        "paper_name": "One Model Any CSP Graph Neural Networks as Fast Gl",
        "source_file": "2208.10227_One_Model_Any_CSP_Graph_Neural_Networks_as_Fast_Gl.md",
        "contribution_index": 2,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "GNN-Based Branching Heuristic",
        "specific_innovation": "A method to extract a branching heuristic for exact backtracking solvers from GNN-learned vertex probability distributions, using the joint information entropy of candidate variables.",
        "problem_addressed": "Replaces problem-agnostic heuristics (like MRV) with a learned heuristic that better predicts search space size, aiming to prune more branches and guide search more efficiently.",
        "evidence_location": "Section 3.4 New Branching Heuristic for Dominating-Clique Solvers",
        "comment": "",
        "paper_name": "Learning Branching Heuristics from Graph Neural Ne",
        "source_file": "2211.14405_Learning_Branching_Heuristics_from_Graph_Neural_Ne.md",
        "contribution_index": 1,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Entropy-Based Branching Calculation",
        "specific_inno vation": "Two novel approximations for calculating the joint information entropy of candidate variables within a clause: a fast version using independence assumptions and an accurate version incorporating problem-specific constraints.",
        "problem_addressed": "Enables efficient and informed branching decisions within a backtracking solver by estimating the remaining search space more accurately than simple cardinality.",
        "evidence_location": "Section 3.4 New Branching Heuristic for Dominating-Clique Solvers",
        "comment": "",
        "paper_name": "Learning Branching Heuristics from Graph Neural Ne",
        "source_file": "2211.14405_Learning_Branching_Heuristics_from_Graph_Neural_Ne.md",
        "contribution_index": 2,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Sampling Procedure",
        "specific_innovation": "Sampling multiple random initializations of literal embeddings to produce diverse candidate solutions for a single SAT problem.",
        "problem_addressed": "Improving the success rate (percentage of solved problems) for satisfiable instances by exploring multiple potential solutions.",
        "evidence_location": "Section 4: Sampling and Decimation",
        "comment": "",
        "paper_name": "Understanding GNNs for Boolean Satisfiability thro",
        "source_file": "2408.15418_Understanding_GNNs_for_Boolean_Satisfiability_thro.md",
        "contribution_index": 1,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Novel Algorithm",
        "specific_innovation": "Casts the one-shot guidance problem as a single-step Markov Decision Process (MDP) and optimizes the policy using Group Relative Policy Optimization (GRPO).",
        "problem_addressed": "Enables direct optimization for minimizing solver cost (e.g., number of decisions) without requiring expert-labeled data or predefined heuristics.",
        "evidence_location": "Section 2.4",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 3,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Structure-Aware Code Generation",
        "specific_innovation": "An LLM is conditioned on GNN-extracted structural embeddings and natural language prompts to generate solver-specific code, ensuring syntactic correctness and problem topology preservation.",
        "problem_addressed": "LLMs lack strong priors for structured domains like CO and generate code that does not respect problem-specific structural invariants.",
        "evidence_location": "4.1 Methodology",
        "comment": "",
        "paper_name": "STRCMP Integrating Graph Structural Priors with La",
        "source_file": "2506.11057_STRCMP_Integrating_Graph_Structural_Priors_with_La.md",
        "contribution_index": 1,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Evolutionary Code Refinement Integration",
        "specific_innovation": "The composite model is embedded within an evolutionary algorithm framework for iterative performance-driven refinement of generated algorithms.",
        "problem_addressed": "Prior LLM-based evolutionary frameworks require many iterations and incur high computational overhead from repeated solver invocations.",
        "evidence_location": "4.1 Methodology",
        "comment": "",
        "paper_name": "STRCMP Integrating Graph Structural Priors with La",
        "source_file": "2506.11057_STRCMP_Integrating_Graph_Structural_Priors_with_La.md",
        "contribution_index": 3,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Novel algorithm",
        "specific_innovation": "Reformulates the SAT solving heuristics of variable phase selection and clause management as conditional probability prediction tasks over the circuit graph.",
        "problem_addressed": "Bridges the disconnect between static circuit structures and the dynamic CDCL solving process, allowing circuit-level intelligence to guide the solver in real-time.",
        "evidence_location": "3.1 Problem Definition",
        "comment": "",
        "paper_name": "Circuit-Aware SAT Solving Guiding CDCL via Conditi",
        "source_file": "2508.04235_Circuit-Aware_SAT_Solving_Guiding_CDCL_via_Conditi.md",
        "contribution_index": 1,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Novel algorithm",
        "specific_innovation": "A method for computing conditional probabilities given multiple assigned nodes by logically conjoining them into a single condition via a chain of AND gates.",
        "problem_addressed": "Enables the model to infer node probabilities under realistic SAT solving scenarios where many variables have been assigned, generalizing the single-condition case.",
        "evidence_location": "3.4 Multiple Condition Probability",
        "comment": "",
        "paper_name": "Circuit-Aware SAT Solving Guiding CDCL via Conditi",
        "source_file": "2508.04235_Circuit-Aware_SAT_Solving_Guiding_CDCL_via_Conditi.md",
        "contribution_index": 5,
        "cluster_name": "Algorithm & Approach Development"
      },
      {
        "contribution_type": "Novel Hardness Heuristic",
        "specific_innovation": "Introduces curvature-based heuristics ω(G) and ω*(G) to predict the difficulty a SAT dataset poses for GNN-based solvers.",
        "problem_addressed": "Provides better predictors of GNN solver generalization error compared to traditional metrics like average clause density.",
        "evidence_location": "Section 4.2: A New Hardness Heuristic for GNN-Based Solvers",
        "comment": "",
        "paper_name": "On the Hardness of Learning GNN-based SAT Solvers ",
        "source_file": "2508.21513_On_the_Hardness_of_Learning_GNN-based_SAT_Solvers_.md",
        "contribution_index": 2,
        "cluster_name": "Algorithm & Approach Development"
      }
    ]
  },
  "Data, Dataset & Benchmark Creation": {
    "canonical_name": "Data, Dataset & Benchmark Creation",
    "member_phrases": [
      "Data Augmentation Strategy",
      "Data Generation Strategy",
      "Data Sampling/Preparation Strategy",
      "Dataset Curation",
      "New Dataset",
      "Novel Dataset",
      "New Benchmark"
    ],
    "contributions": [
      {
        "contribution_type": "New Dataset",
        "specific_innovation": "Creates DataBack, a new publicly available dataset of 120,286 SAT formulas labeled with backbone variable phases, collected from five sources and balanced via a dual-formula augmentation strategy.",
        "problem_addressed": "Provides a large-scale, diverse, and balanced dataset for training and evaluating models for backbone phase prediction.",
        "evidence_location": "Section 5",
        "comment": "A significant resource contribution.",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 5,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "Data Augmentation Strategy",
        "specific_innovation": "Proposes creating a dual formula by negating all backbone variables, which flips the phase labels, to perfectly balance the dataset.",
        "problem_addressed": "Mitigates the significant label imbalance inherent in backbone variable datasets.",
        "evidence_location": "Section 5",
        "comment": "",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 6,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "New Dataset",
        "specific_innovation": "Construction of labeled datasets for random Max2SAT and Max3SAT instances using a standard generator, with solutions obtained via the MaxHS solver.",
        "problem_addressed": "Lack of large-scale labeled datasets for training and evaluating GNNs on the MaxSAT problem.",
        "evidence_location": "Section 5.2: Data Generation",
        "comment": "",
        "paper_name": "Can Graph Neural Networks Learn to Solve MaxSAT Pr",
        "source_file": "2111.07568_Can_Graph_Neural_Networks_Learn_to_Solve_MaxSAT_Pr.md",
        "contribution_index": 5,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "Data Generation Strategy",
        "specific_innovation": "A procedure for generating a training dataset of random SAT CNF formulae with specified variable/clause ranges, filtered for satisfiability and labeled with exact model counts.",
        "problem_addressed": "Creating a data-efficient, easily generable training set that allows the model to learn scalable and generalizable approximations for #SAT.",
        "evidence_location": "Section 4.1: BPGAT Training Data",
        "comment": "",
        "paper_name": "Graph Neural Networks for Propositional Model Coun",
        "source_file": "2205.04423_Graph_Neural_Networks_for_Propositional_Model_Coun.md",
        "contribution_index": 4,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "Novel Dataset",
        "specific_innovation": "Created a dataset of symmetric circuit problems with asymmetric solutions (e.g., XOR) to specifically test the model's ability to handle variable dependency.",
        "problem_addressed": "Provides a benchmark to expose the fundamental weakness of concurrent prediction models and validate the proposed solution.",
        "evidence_location": "Section 5.1: Data preparation",
        "comment": "",
        "paper_name": "Addressing Variable Dependency in GNN-based SAT So",
        "source_file": "2304.08738_Addressing_Variable_Dependency_in_GNN-based_SAT_So.md",
        "contribution_index": 3,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "Data Sampling/Preparation Strategy",
        "specific_innovation": "A method for sampling logic gate pairs based on constraints like shared primary inputs, similar logic probability, close logic levels, and extreme truth table differences, using incomplete truth tables from limited simulation.",
        "problem_addressed": "The infeasibility of using complete truth tables for supervision due to exponential complexity and the need for a focused, high-quality dataset for efficient training.",
        "evidence_location": "Section III-B",
        "comment": "",
        "paper_name": "DeepGate2 Functionality-Aware Circuit Representati",
        "source_file": "2305.16373_DeepGate2_Functionality-Aware_Circuit_Representati.md",
        "contribution_index": 3,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "New Benchmark",
        "specific_innovation": "G4SATBench establishes the first comprehensive evaluation framework, including a unified large-scale dataset and a re-implementation suite, for GNN-based SAT solvers.",
        "problem_addressed": "The absence of a unified dataset and a fair benchmark to evaluate and compare existing GNN-based SAT solving approaches.",
        "evidence_location": "Abstract, Section 1, Section 4",
        "comment": "The benchmark is the paper's core contribution, integrating data, models, tasks, and evaluation protocols.",
        "paper_name": "G4SATBench Benchmarking and Advancing SAT Solving ",
        "source_file": "2309.16941_G4SATBench_Benchmarking_and_Advancing_SAT_Solving_.md",
        "contribution_index": 0,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "Dataset Curation",
        "specific_innovation": "A large and diverse collection of 7 SAT datasets (from random, pseudo-industrial, and combinatorial domains) is curated, each with 3 difficulty levels (easy, medium, hard) and non-trivial generation parameters.",
        "problem_addressed": "Lack of standardized, large-scale, and diverse datasets for training and evaluating GNNs on SAT problems.",
        "evidence_location": "Section 4.1, Appendix A",
        "comment": "The dataset generation uses careful parameter selection (e.g., phase-transition region for 3-SAT) to avoid trivial instances.",
        "paper_name": "G4SATBench Benchmarking and Advancing SAT Solving ",
        "source_file": "2309.16941_G4SATBench_Benchmarking_and_Advancing_SAT_Solving_.md",
        "contribution_index": 1,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      },
      {
        "contribution_type": "New Dataset",
        "specific_innovation": "The paper introduces and evaluates on a new industrial LEC circuit SAT instances dataset derived from real chip design processes, characterized by a very high percentage of UNSAT problems and large UNSAT-cores.",
        "problem_addressed": "Existing SAT datasets (e.g., from competitions) do not reflect the specific, imbalanced characteristics of problems encountered in the Logic Equivalence Checking (LEC) domain.",
        "evidence_location": "4.1 Datasets",
        "comment": "",
        "paper_name": "IB-Net Initial Branch Network for Variable Decisio",
        "source_file": "2403.03517_IB-Net_Initial_Branch_Network_for_Variable_Decisio.md",
        "contribution_index": 4,
        "cluster_name": "Data, Dataset & Benchmark Creation"
      }
    ]
  },
  "Evaluation & Metrics": {
    "canonical_name": "Evaluation & Metrics",
    "member_phrases": [
      "Evaluation Metric",
      "Evaluation Metric / Analysis",
      "Evaluation Metric / Analysis Method",
      "Evaluation Metric / Performance Analysis",
      "Evaluation Protocol",
      "Empirical Validation Protocol",
      "Novel Evaluation",
      "Benchmarking Analysis",
      "Unified Evaluation Framework",
      "Comparison Framework"
    ],
    "contributions": [
      {
        "contribution_type": "Evaluation Protocol",
        "specific_innovation": "Systematic evaluation of neural heuristics against classical heuristics (DLIS, JW-OS) within DPLL and CDCL on the SR(n) problem family, measuring percentage of instances solved within a step limit and direct step count comparison.",
        "problem_addressed": "Demonstrating the practical competitiveness of learned heuristics, especially on larger problem instances.",
        "evidence_location": "Section 4 Experimental Results (Experiments 1 & 2)",
        "comment": "",
        "paper_name": "Neural heuristics for SAT solving",
        "source_file": "2005.13406_Neural_heuristics_for_SAT_solving.md",
        "contribution_index": 4,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Evaluation Metric",
        "specific_innovation": "Introduction of 'gap to optimal objective' and 'accuracy of assignments' as metrics to evaluate the quality of predicted MaxSAT solutions.",
        "problem_addressed": "Previous SAT-focused GNN work lacked metrics to assess the quality of predicted solutions; these metrics better capture performance on the optimization task.",
        "evidence_location": "Section 5.4: Accuracy of Models",
        "comment": "",
        "paper_name": "Can Graph Neural Networks Learn to Solve MaxSAT Pr",
        "source_file": "2111.07568_Can_Graph_Neural_Networks_Learn_to_Solve_MaxSAT_Pr.md",
        "contribution_index": 3,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Evaluation Metric",
        "specific_innovation": "Using Root Mean Squared Error (RMSE) and Mean Relative Error (MRE) on the logarithm of the model count to benchmark against state-of-the-art approximate solvers like ApproxMC.",
        "problem_addressed": "Quantitatively assessing the accuracy and scalability of the approximate #SAT solver, focusing on relative error which is crucial for counting problems with exponentially large outputs.",
        "evidence_location": "Section 4.2: Results",
        "comment": "",
        "paper_name": "Graph Neural Networks for Propositional Model Coun",
        "source_file": "2205.04423_Graph_Neural_Networks_for_Propositional_Model_Coun.md",
        "contribution_index": 5,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Evaluation Metric",
        "specific_innovation": "Used 'solution rate' (percentage of problems for which a correct satisfying assignment is predicted) as the primary performance metric.",
        "problem_addressed": "Provides a more rigorous evaluation of a model's SAT-solving capability than mere satisfiability classification accuracy.",
        "evidence_location": "Section 5.2: Experimental setup and result",
        "comment": "",
        "paper_name": "Addressing Variable Dependency in GNN-based SAT So",
        "source_file": "2304.08738_Addressing_Variable_Dependency_in_GNN-based_SAT_So.md",
        "contribution_index": 6,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Evaluation Metric / Performance Analysis",
        "specific_innovation": "A comprehensive empirical evaluation demonstrating that GNN-based oracles significantly boost SLS solver performance on random 3-SAT instances.",
        "problem_addressed": "Quantifies the practical improvement gained by using learned oracles, showing increased solvable difficulty (α) and reduced steps.",
        "evidence_location": "Section 4, Results",
        "comment": "The results (Table 1, Figures 4-7) validate the core hypothesis of the paper.",
        "paper_name": "Using deep learning to construct stochastic local ",
        "source_file": "2309.11452_Using_deep_learning_to_construct_stochastic_local_.md",
        "contribution_index": 4,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Unified Evaluation Framework",
        "specific_innovation": "The framework provides a unified implementation and evaluation protocol for various GNN models, graph encodings, prediction tasks, training objectives, and inference algorithms.",
        "problem_addressed": "Difficulty in fairly comparing disparate GNN-based SAT solvers due to varied implementations, tasks, and objectives.",
        "evidence_location": "Section 4",
        "comment": "Enables systematic benchmarking of GNNs on LCG*/VCG* graphs for tasks like satisfiability, assignment, and unsat-core prediction.",
        "paper_name": "G4SATBench Benchmarking and Advancing SAT Solving ",
        "source_file": "2309.16941_G4SATBench_Benchmarking_and_Advancing_SAT_Solving_.md",
        "contribution_index": 2,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Benchmarking Analysis",
        "specific_innovation": "A comprehensive empirical evaluation reveals that unsupervised training outperforms supervised learning for assignment prediction, and GNNs struggle to generalize to out-of-distribution and larger-scale instances.",
        "problem_addressed": "Lack of a clear understanding of the relative performance, generalization capabilities, and training dynamics of different GNN-based SAT solvers.",
        "evidence_location": "Section 5",
        "comment": "The analysis provides key insights, such as the instability of one unsupervised loss and the superior performance of NeuroSAT and GGNN.",
        "paper_name": "G4SATBench Benchmarking and Advancing SAT Solving ",
        "source_file": "2309.16941_G4SATBench_Benchmarking_and_Advancing_SAT_Solving_.md",
        "contribution_index": 3,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Evaluation Metric / Analysis Method",
        "specific_innovation": "Using the Silhouette score of literal embeddings as a proxy for satisfiability classification, replacing the final voting layer.",
        "problem_addressed": "Providing evidence that the formation of two clusters in the embedding space is the primary mechanism the GNN uses for SAT prediction.",
        "evidence_location": "Section 4",
        "comment": "",
        "paper_name": "Understanding GNNs for Boolean Satisfiability thro",
        "source_file": "2408.15418_Understanding_GNNs_for_Boolean_Satisfiability_thro.md",
        "contribution_index": 6,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Evaluation Metric / Analysis",
        "specific_innovation": "Comprehensive evaluation metrics including Average Gap (unsatisfied clauses), Gap separated by SAT/UNSAT status, SAT Accuracy (percentage of satisfiable instances solved), and Decision Accuracy, with a focus on test-time scaling.",
        "problem_addressed": "Provides a nuanced performance assessment beyond simple classification accuracy, capturing how close the model gets to satisfying formulas and its ability to scale compute at inference.",
        "evidence_location": "Section 5.1 (Training and Evaluation Methodology), Section 5.2 (Quantitative Evaluation), Section 5.3 (Test-time Scaling)",
        "comment": "The 'Gap' metric is central to their evaluation, and they extensively analyze performance as a function of iterations and resampling attempts.",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 2,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Evaluation Protocol",
        "specific_innovation": "Evaluates RLAF across two distinct SAT solvers (Glucose, March) and three diverse SAT problem distributions, demonstrating generalization to larger and harder instances than seen during training.",
        "problem_addressed": "Demonstrates the effectiveness and generalization capability of the proposed method beyond a single solver or problem type.",
        "evidence_location": "Section 3, Section 3.1",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 5,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Comparison Framework",
        "specific_innovation": "Benchmarks RLAF against supervised baselines that predict handcrafted properties (backbone and UNSAT core membership) and transforms predictions into solver guidance using the same weight/polarity injection mechanism.",
        "problem_addressed": "Provides a fair and direct comparison to demonstrate that RL-based training outperforms supervised learning of predefined heuristics.",
        "evidence_location": "Section 3.2, Appendix B.4",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 6,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Novel Evaluation",
        "specific_innovation": "Extensive empirical evaluation across two CO domains (MILP and SAT) and nine benchmark datasets, comparing against neural and LLM-based baselines on optimality and efficiency metrics.",
        "problem_addressed": "Need to demonstrate the framework's effectiveness and superiority over existing methods in practical settings.",
        "evidence_location": "5 Experimental Evaluation",
        "comment": "",
        "paper_name": "STRCMP Integrating Graph Structural Priors with La",
        "source_file": "2506.11057_STRCMP_Integrating_Graph_Structural_Priors_with_La.md",
        "contribution_index": 5,
        "cluster_name": "Evaluation & Metrics"
      },
      {
        "contribution_type": "Empirical Validation Protocol",
        "specific_innovation": "Uses a test-time graph rewiring procedure based on stochastic Ricci flow to show performance improvements on GNN-based SAT solvers.",
        "problem_addressed": "Demonstrates that reducing graph curvature directly alleviates oversquashing and improves solver accuracy without retraining.",
        "evidence_location": "Section 4.1: The Relationship Between Curvature and Satisfiability",
        "comment": "Described in detail in Appendix A.3",
        "paper_name": "On the Hardness of Learning GNN-based SAT Solvers ",
        "source_file": "2508.21513_On_the_Hardness_of_Learning_GNN-based_SAT_Solvers_.md",
        "contribution_index": 3,
        "cluster_name": "Evaluation & Metrics"
      }
    ]
  },
  "Loss Function & Training Design": {
    "canonical_name": "Loss Function & Training Design",
    "member_phrases": [
      "Loss Function for Minimum Dominating Clique",
      "Novel Loss Function",
      "Novel Loss Function Design",
      "Novel Reward Scheme",
      "Novel Supervision Method",
      "Novel Training Paradigm",
      "Training/Optimization Procedure"
    ],
    "contributions": [
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A multi-task training objective combining sat prediction loss (for whole formula satisfiability) and policy prediction loss (per-literal prediction of solution existence) across all message-passing iterations.",
        "problem_addressed": "Training the network to simultaneously learn global formula satisfiability and per-literal guidance signals useful for branching decisions.",
        "evidence_location": "Section 3 Architecture",
        "comment": "",
        "paper_name": "Neural heuristics for SAT solving",
        "source_file": "2005.13406_Neural_heuristics_for_SAT_solving.md",
        "contribution_index": 2,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "Employs a two-stage training process (pre-training on a diverse dataset, followed by fine-tuning on a domain-specific dataset) for the GNN model.",
        "problem_addressed": "Equips the model with general knowledge of backbone phases and then specializes it for challenging SAT competition problems.",
        "evidence_location": "Section 4.2.3",
        "comment": "",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 4,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "Applies transfer learning, training the GNN solely on predicting backbone variable phases and then using it to predict phases for all variables.",
        "problem_addressed": "Leverages knowledge from predicting consistent backbone phases to guide predictions for non-backbone variables appearing in the majority of satisfying assignments.",
        "evidence_location": "Section 4.2",
        "comment": "A core insight of the approach.",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 7,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "Implementation of GNNs with specific aggregating (MLP + sum) and updating (LSTM) functions for the MaxSAT factor graphs, trained with BCE loss on variable assignments.",
        "problem_addressed": "Adapting GNN architectures and training procedures from SAT to the MaxSAT optimization task.",
        "evidence_location": "Section 5.1: Building the Models",
        "comment": "",
        "paper_name": "Can Graph Neural Networks Learn to Solve MaxSAT Pr",
        "source_file": "2111.07568_Can_Graph_Neural_Networks_Learn_to_Solve_MaxSAT_Pr.md",
        "contribution_index": 4,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A two-phase training protocol involving pre-training on small random Boolean formulae followed by efficient fine-tuning on specific formula distributions.",
        "problem_addressed": "Enabling the model to generalize effectively to different, unseen formula distributions (e.g., SAT-encoded combinatorial problems) with minimal distribution-specific labeled data.",
        "evidence_location": "Section 4.1: Fine-tuning Protocol",
        "comment": "",
        "paper_name": "Graph Neural Networks for Propositional Model Coun",
        "source_file": "2205.04423_Graph_Neural_Networks_for_Propositional_Model_Coun.md",
        "contribution_index": 1,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "specific_innovation": "Training the model to minimize the Mean Squared Error (MSE) between the natural logarithm of the true model count and the model's predicted log-count.",
        "problem_addressed": "Directly learning to approximate the logarithm of the number of satisfying assignments (#SAT), which is a more stable regression target than the raw count.",
        "evidence_location": "Section 4.1: BPGAT Training Protocol",
        "comment": "",
        "paper_name": "Graph Neural Networks for Propositional Model Coun",
        "source_file": "2205.04423_Graph_Neural_Networks_for_Propositional_Model_Coun.md",
        "contribution_index": 2,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Reward Scheme",
        "specific_innovation": "A reward function that only gives positive reward for improvements over the best quality seen so far, encouraging exploration by not punishing steps away from local maxima.",
        "problem_addressed": "Simple reward schemes based directly on assignment quality cause the policy to get stuck in local maxima and stagnate.",
        "evidence_location": "Section 4.2, Equation 3",
        "comment": "The reward is zero if the new assignment does not improve over the previous best, and equal to the improvement margin if it does.",
        "paper_name": "One Model Any CSP Graph Neural Networks as Fast Gl",
        "source_file": "2208.10227_One_Model_Any_CSP_Graph_Neural_Networks_as_Fast_Gl.md",
        "contribution_index": 3,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "Unsupervised training using policy gradient descent (REINFORCE) on an exponentially large action space (all possible assignments) with a reward scheme designed for iterative improvement.",
        "problem_addressed": "Training a GNN policy to handle global search over an action space of all assignments, which can be extremely large (up to 10^50 actions).",
        "evidence_location": "Abstract, Section 4.2, Section A.2",
        "comment": "The paper shows REINFORCE is surprisingly effective despite the large action space, without needing a baseline or critic network.",
        "paper_name": "One Model Any CSP Graph Neural Networks as Fast Gl",
        "source_file": "2208.10227_One_Model_Any_CSP_Graph_Neural_Networks_as_Fast_Gl.md",
        "contribution_index": 4,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function Design",
        "specific_innovation": "A new loss function based on the correlation inequality from the probabilistic method to maximize the probability measure of valid solutions (e.g., cliques or dominating cliques) without restrictive assumptions.",
        "problem_addressed": "Addresses the limitations of prior probabilistic-method GNN loss functions, which rely on the first-moment method and require careful parameter tuning to guarantee solution existence.",
        "evidence_location": "Section 3.1 New Probabilistic-Method GNN Model",
        "comment": "",
        "paper_name": "Learning Branching Heuristics from Graph Neural Ne",
        "source_file": "2211.14405_Learning_Branching_Heuristics_from_Graph_Neural_Ne.md",
        "contribution_index": 0,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Loss Function for Minimum Dominating Clique",
        "specific_innovation": "An enhanced loss function incorporating a refined expected size term (Equation 7) that more accurately targets small-sized dominating cliques during training.",
        "problem_addressed": "Improves the GNN's ability to learn distributions that favor minimal solutions for the minimum dominating clique problem, beyond just feasibility.",
        "evidence_location": "Section 3.3 Loss Function for (Minimum) Dominating-Clique Problem",
        "comment": "",
        "paper_name": "Learning Branching Heuristics from Graph Neural Ne",
        "source_file": "2211.14405_Learning_Branching_Heuristics_from_Graph_Neural_Ne.md",
        "contribution_index": 4,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "specific_innovation": "A multi-bit Cross-Entropy loss function for supervised learning of SAT solution assignments, contrasting with single-bit SAT/UNSAT classification.",
        "problem_addressed": "Enables direct, supervised learning of satisfying assignments, improving training data efficiency compared to prior work.",
        "evidence_location": "Section 4.3: Training",
        "comment": "",
        "paper_name": "Addressing Variable Dependency in GNN-based SAT So",
        "source_file": "2304.08738_Addressing_Variable_Dependency_in_GNN-based_SAT_So.md",
        "contribution_index": 1,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "Supervised training on circuit-SAT problem pairs (graph, satisfying assignment) generated by an external oracle solver.",
        "problem_addressed": "Provides a more direct and data-efficient learning objective for predicting full satisfying assignments, compared to unsupervised or single-bit supervised methods.",
        "evidence_location": "Section 4.1: Problem formulation, Section 4.3: Training",
        "comment": "",
        "paper_name": "Addressing Variable Dependency in GNN-based SAT So",
        "source_file": "2304.08738_Addressing_Variable_Dependency_in_GNN-based_SAT_So.md",
        "contribution_index": 2,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "specific_innovation": "A functionality-aware loss function that minimizes the difference between pairwise node embedding distance and the Hamming distance of their incomplete truth tables.",
        "problem_addressed": "Weak supervision in prior circuit representation learning methods that relied only on statistical logic probability, which cannot differentiate gates with the same probability but different truth tables.",
        "evidence_location": "Section III-C",
        "comment": "",
        "paper_name": "DeepGate2 Functionality-Aware Circuit Representati",
        "source_file": "2305.16373_DeepGate2_Functionality-Aware_Circuit_Representati.md",
        "contribution_index": 0,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A multi-stage, multi-task training strategy where the model first learns simpler tasks (logic probability, structural correlation) before learning the harder functionality-aware task.",
        "problem_addressed": "The high cost and difficulty of directly learning functional similarity from a large, sparse set of node pairs.",
        "evidence_location": "Section III-E",
        "comment": "",
        "paper_name": "DeepGate2 Functionality-Aware Circuit Representati",
        "source_file": "2305.16373_DeepGate2_Functionality-Aware_Circuit_Representati.md",
        "contribution_index": 2,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "specific_innovation": "A new, theoretically motivated loss function called the Lovász Local (LLL) loss, derived from conditions for efficient SLS solver performance.",
        "problem_addressed": "Provides a training objective that directly rewards an oracle for exploiting the local structure of SAT instances, which is linked to theoretical performance guarantees.",
        "evidence_location": "Section 2.4, Learning oracle factories with the LLL loss",
        "comment": "This loss is central to the paper's goal of constructing solvers with performance bounds.",
        "paper_name": "Using deep learning to construct stochastic local ",
        "source_file": "2309.11452_Using_deep_learning_to_construct_stochastic_local_.md",
        "contribution_index": 0,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A training procedure for the GNN-based oracle factory that minimizes a combined loss (LLL + Gibbs cross-entropy) on a dataset of SAT instances.",
        "problem_addressed": "Optimizes the oracle factory to produce distributions that satisfy theoretical performance conditions and approximate a Gibbs distribution over solutions.",
        "evidence_location": "Section 2.4, Learning oracle factories with the LLL loss",
        "comment": "The combined loss function and its practical estimation (Appendix A.1) define the training process.",
        "paper_name": "Using deep learning to construct stochastic local ",
        "source_file": "2309.11452_Using_deep_learning_to_construct_stochastic_local_.md",
        "contribution_index": 3,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A two-stage unified training framework consisting of (1) a supervised pre-training stage on generated Max-SAT instances to learn generalizable features, and (2) a fine-tuning stage using a domain adaptation setup with adversarial training to align features between a source Max-SAT domain and a target CO domain.",
        "problem_addressed": "The lack of a method to learn and transfer generalizable knowledge across different CO problems, and the challenge of limited target-domain data for specific CO tasks.",
        "evidence_location": "Section 3.4 Pre-Training and Fine-Tuning with Domain Adaptation",
        "comment": "This is the core procedural innovation of the paper, designed to extract and utilize transferable features.",
        "paper_name": "A unified pre-training and adaptation framework fo",
        "source_file": "2312.11547_A_unified_pre-training_and_adaptation_framework_fo.md",
        "contribution_index": 2,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "The model is trained for UNSAT-core variable prediction (a classification task) using the Focal loss instead of standard Cross-Entropy loss.",
        "problem_addressed": "To handle the severe class imbalance present in LEC UNSAT problems, where a large proportion of variables belong to the UNSAT-core.",
        "evidence_location": "3.4 Training",
        "comment": "",
        "paper_name": "IB-Net Initial Branch Network for Variable Decisio",
        "source_file": "2403.03517_IB-Net_Initial_Branch_Network_for_Variable_Decisio.md",
        "contribution_index": 3,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "problem_addressed": "Standard cross-entropy loss does not align with the objective of minimizing solver runtime, as it penalizes all incorrect predictions equally.",
        "specific_innovation": "A regret-like loss function is introduced that penalizes predictions based on the additional runtime induced compared to the optimal solver.",
        "evidence_location": "Section 3.4 Training",
        "comment": "",
        "paper_name": "GraSS Combining Graph Neural Networks with Expert ",
        "source_file": "2405.11024_GraSS_Combining_Graph_Neural_Networks_with_Expert_.md",
        "contribution_index": 3,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A curriculum training procedure that incrementally increases problem complexity (number of variables) and the number of message-passing iterations during GNN training.",
        "problem_addressed": "Slow training convergence of GNNs for SAT solving.",
        "evidence_location": "Section 3: Curriculum for Training GNNs",
        "comment": "",
        "paper_name": "Understanding GNNs for Boolean Satisfiability thro",
        "source_file": "2408.15418_Understanding_GNNs_for_Boolean_Satisfiability_thro.md",
        "contribution_index": 0,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "specific_innovation": "Proposed and experimented with training a GNN using a differentiable SDP-based MAX-SAT objective function instead of a binary classification loss.",
        "problem_addressed": "Exploring alternative supervision signals that provide richer gradient information compared to single-bit satisfiability labels.",
        "evidence_location": "Section 5.2.2: Training directly with the MAX-SAT SDP objective function",
        "comment": "Presented as a preliminary exploration; achieved lower accuracy but showed less dependence on curriculum training.",
        "paper_name": "Understanding GNNs for Boolean Satisfiability thro",
        "source_file": "2408.15418_Understanding_GNNs_for_Boolean_Satisfiability_thro.md",
        "contribution_index": 5,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Supervision Method",
        "specific_innovation": "A closest assignment supervision method that dynamically uses a MaxSAT solver to compute the solution (for SAT) or optimal partial assignment (for UNSAT) with minimal Hamming distance to the model's current prediction.",
        "problem_addressed": "It addresses the limitation of static assignment supervision where the model is penalized for predicting other valid solutions, particularly improving performance on problems with larger solution spaces.",
        "evidence_location": "Section 4.3 (Supervision Tasks and Objectives), Section 5.2 (Quantitative Evaluation), Table 3",
        "comment": "Introduced as a novel supervision method. The results show significant improvements, especially on benchmarks with more variables/solutions.",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 0,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A curriculum learning strategy that starts training with small formulas (5 variables) and gradually increases formula size upon reaching accuracy thresholds.",
        "problem_addressed": "It accelerates training and improves generalization by allowing the model to learn basic logical reasoning patterns before tackling more complex instances.",
        "evidence_location": "Appendix A.1.1 (Curriculum Learning)",
        "comment": "Described as a training trick. Figure 6 shows it reduces training time from over 5 hours to ~30 minutes to reach 85% validation accuracy.",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 1,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "Use of Exponential Moving Average (EMA) of model parameters during validation/testing to stabilize training and improve generalization.",
        "problem_addressed": "Mitigates high variance and fluctuations in validation accuracy observed during training without EMA.",
        "evidence_location": "Appendix A.1.2 (Exponential Moving Average (EMA))",
        "comment": "Described as a training trick. They note it provides a smooth validation curve compared to high-variance non-EMA training.",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 6,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "specific_innovation": "An unsupervised multi-objective loss function combining a primary task loss (relaxed Weighted MaxSAT objective) and a shared representation constraint loss.",
        "problem_addressed": "Enabling gradient-based optimization without labeled data and encouraging distinct feature representations for complementary literals to improve separability.",
        "evidence_location": "3.2.3 Loss Function",
        "comment": "The shared constraint loss minimizes the Frobenius norm of the sum of positive and negative literal representations, pushing them apart.",
        "paper_name": "HyperSAT Unsupervised Hypergraph Neural Networks f",
        "source_file": "2504.11885_HyperSAT_Unsupervised_Hypergraph_Neural_Networks_f.md",
        "contribution_index": 3,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Training Paradigm",
        "specific_innovation": "Introduces Reinforcement Learning from Algorithm Feedback (RLAF) as a paradigm for training GNNs to guide SAT solver branching heuristics directly using the solver's computational cost as reward.",
        "problem_addressed": "SAT solver performance is heavily reliant on hand-crafted heuristics, which are difficult to adapt to specific problem distributions.",
        "evidence_location": "Abstract, Section 1, Section 2.4",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 0,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "Trains the GNN policy using GRPO with a group-relative advantage computed from multiple solver runs per instance, coupled with a KL-divergence penalty for stability.",
        "problem_addressed": "Addresses the challenge of optimizing a policy with a non-differentiable, expensive environment (the SAT solver) and high-variance rewards.",
        "evidence_location": "Section 2.4, Figure 3",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 4,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A two-phase training protocol involving data curation (collecting problem-code-performance triplets) and post-training (SFT + DPO) on the composite GNN-LLM model.",
        "problem_addressed": "The composite model exhibits suboptimal performance due to architectural incompatibility disrupting the LLM's native token prediction.",
        "evidence_location": "4.1 Methodology",
        "comment": "",
        "paper_name": "STRCMP Integrating Graph Structural Priors with La",
        "source_file": "2506.11057_STRCMP_Integrating_Graph_Structural_Priors_with_La.md",
        "contribution_index": 2,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A two-stage training strategy consisting of pattern-based pre-training (100 patterns/minibatch) followed by workload-aware fine-tuning (20,000 patterns).",
        "problem_addressed": "Enables the model to capture both fine-grained circuit behavior and broad statistical properties of input distributions, leading to accurate and generalizable probability predictions.",
        "evidence_location": "3.3 Two-Stage Training Strategy",
        "comment": "",
        "paper_name": "Circuit-Aware SAT Solving Guiding CDCL via Conditi",
        "source_file": "2508.04235_Circuit-Aware_SAT_Solving_Guiding_CDCL_via_Conditi.md",
        "contribution_index": 2,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Training/Optimization Procedure",
        "specific_innovation": "A dynamic attention mechanism that progressively increases the number of attention heads during training and assigns learnable weights to heads, allowing unimportant heads to be pruned.",
        "problem_addressed": "Balances model expressiveness and computational cost across different training stages and formula complexities, improving training speed and reducing resource consumption.",
        "evidence_location": "Section 3.2 (Dynamic attention mechanism)",
        "comment": "This is an optimization for the training process and model efficiency.",
        "paper_name": "Attn-JGNN Attention Enhanced Join-Graph Neural Net",
        "source_file": "2510.15583_Attn-JGNN_Attention_Enhanced_Join-Graph_Neural_Net.md",
        "contribution_index": 3,
        "cluster_name": "Loss Function & Training Design"
      },
      {
        "contribution_type": "Novel Loss Function",
        "specific_innovation": "A constraint-aware regularization term added to the main RMSE loss, which penalizes variable assignments that violate clauses and prioritizes easily satisfied clauses.",
        "problem_addressed": "Explicitly guides the model to satisfy the logical constraints of the CNF formula, leading to more accurate approximations of the model count.",
        "evidence_location": "Section 3.2 (Constraint-Aware Mechanism)",
        "comment": "This loss modification is a form of domain-knowledge integration.",
        "paper_name": "Attn-JGNN Attention Enhanced Join-Graph Neural Net",
        "source_file": "2510.15583_Attn-JGNN_Attention_Enhanced_Join-Graph_Neural_Net.md",
        "contribution_index": 4,
        "cluster_name": "Loss Function & Training Design"
      }
    ]
  },
  "Input & Feature Design": {
    "canonical_name": "Input & Feature Design",
    "member_phrases": [
      "Embedding Initialization Strategy",
      "Node Feature Design",
      "Positional Encoding"
    ],
    "contributions": [
      {
        "contribution_type": "Embedding Initialization Strategy",
        "specific_innovation": "Primary Input Encoding (PIE): assigning orthogonal vectors as unique structural embeddings for each Primary Input while using a uniform vector for their functional embeddings.",
        "problem_addressed": "The lack of unique identifiers for PIs in prior models, which prevented efficient modeling of reconvergent structures and required many message-passing rounds.",
        "evidence_location": "Section III-D",
        "comment": "",
        "paper_name": "DeepGate2 Functionality-Aware Circuit Representati",
        "source_file": "2305.16373_DeepGate2_Functionality-Aware_Circuit_Representati.md",
        "contribution_index": 4,
        "cluster_name": "Input & Feature Design"
      },
      {
        "contribution_type": "Node Feature Design",
        "specific_innovation": "Novel hand-designed node features for clause and literal nodes, inspired by SATzilla features, are incorporated into the graph representation.",
        "problem_addressed": "Existing SAT solver selection methods rely on global hand-picked features that are costly to compute and ignore structural information.",
        "evidence_location": "Section 3.2 Representation and features",
        "comment": "Details provided in Appendix A (Table 7).",
        "paper_name": "GraSS Combining Graph Neural Networks with Expert ",
        "source_file": "2405.11024_GraSS_Combining_Graph_Neural_Networks_with_Expert_.md",
        "contribution_index": 1,
        "cluster_name": "Input & Feature Design"
      },
      {
        "contribution_type": "Positional Encoding",
        "specific_innovation": "Clause nodes are enriched with sinusoidal positional encodings to capture the order of clauses within the CNF formula.",
        "problem_addressed": "Solver runtime can be sensitive to clause ordering, but standard graph representations and models are permutation-invariant and ignore this variation.",
        "evidence_location": "Section 3.2.1 Clause positional embeddings",
        "comment": "",
        "paper_name": "GraSS Combining Graph Neural Networks with Expert ",
        "source_file": "2405.11024_GraSS_Combining_Graph_Neural_Networks_with_Expert_.md",
        "contribution_index": 2,
        "cluster_name": "Input & Feature Design"
      }
    ]
  },
  "Graph Representation & Encoding": {
    "canonical_name": "Graph Representation & Encoding",
    "member_phrases": [
      "Graph Representation",
      "Novel Graph Encoding",
      "Novel Graph Representation",
      "Problem Representation",
      "Unified Problem Representation"
    ],
    "contributions": [
      {
        "contribution_type": "Graph Representation",
        "specific_innovation": "Proposes a compact undirected bipartite graph for CNF formulas, enhanced with meta nodes and edges to reduce graph diameter and improve learnability.",
        "problem_addressed": "Creates a more learnable representation for large-scale SAT formulas that facilitates effective message passing in GNNs.",
        "evidence_location": "Section 4.1",
        "comment": "",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 2,
        "cluster_name": "Graph Representation & Encoding"
      },
      {
        "contribution_type": "Novel Graph Representation",
        "specific_innovation": "A tripartite constraint value graph representation that compactly encodes the CSP instance and current assignment through binary labels on vertices and edges, making it generic for any CSP.",
        "problem_addressed": "Need for a graph representation that can universally represent any CSP instance for a single GNN, regardless of constraint arity, relations, or domain size.",
        "evidence_location": "Section 4, Figure 1",
        "comment": "Nodes represent variables, values, and constraints; edges connect variables to values and constraints to values, with edge labels encoding the effect of changing a variable's value on constraint satisfaction.",
        "paper_name": "One Model Any CSP Graph Neural Networks as Fast Gl",
        "source_file": "2208.10227_One_Model_Any_CSP_Graph_Neural_Networks_as_Fast_Gl.md",
        "contribution_index": 1,
        "cluster_name": "Graph Representation & Encoding"
      },
      {
        "contribution_type": "Unified Problem Representation",
        "specific_innovation": "A method to transform diverse combinatorial optimization (CO) problems on graphs into a standardized form using the Maximum Satisfiability (Max-SAT) problem, converting graph constraints and objectives into clauses and then into a bipartite graph representation.",
        "problem_addressed": "Existing GNN frameworks for CO problems are designed for specific tasks, lacking transferability, and original graphs fail to capture the mathematical logicality and properties inherent in CO problems.",
        "evidence_location": "Section 3.2 Problem Transfer via Max-SAT Problem",
        "comment": "This serves as the foundational step for the proposed unified framework, enabling the application of generic pre-training and adaptation.",
        "paper_name": "A unified pre-training and adaptation framework fo",
        "source_file": "2312.11547_A_unified_pre-training_and_adaptation_framework_fo.md",
        "contribution_index": 0,
        "cluster_name": "Graph Representation & Encoding"
      },
      {
        "contribution_type": "Novel Graph Encoding",
        "specific_innovation": "The paper introduces the Weighted Literal-Incidence Graph (WLIG), where literals are nodes and edges represent co-occurrence in clauses, with edge weights counting common incidences.",
        "problem_addressed": "Standard graph encodings (like Literal-Clause Graph) include redundant clause node information that is less useful for predicting UNSAT-core variables in imbalanced LEC datasets.",
        "evidence_location": "3.2 Graph Formulation",
        "comment": "",
        "paper_name": "IB-Net Initial Branch Network for Variable Decisio",
        "source_file": "2403.03517_IB-Net_Initial_Branch_Network_for_Variable_Decisio.md",
        "contribution_index": 0,
        "cluster_name": "Graph Representation & Encoding"
      },
      {
        "contribution_type": "Graph Representation",
        "specific_innovation": "SAT instances are represented as attributed tripartite literal-clause graphs, enabling the model to use the complete instance data in a size-independent way.",
        "problem_addressed": "Existing methods rely on hand-picked features that summarize the instance, losing structural information.",
        "evidence_location": "Section 3.2 Representation and features",
        "comment": "This is the foundational representation for the entire approach.",
        "paper_name": "GraSS Combining Graph Neural Networks with Expert ",
        "source_file": "2405.11024_GraSS_Combining_Graph_Neural_Networks_with_Expert_.md",
        "contribution_index": 4,
        "cluster_name": "Graph Representation & Encoding"
      },
      {
        "contribution_type": "Problem Representation",
        "specific_innovation": "A hypergraph representation for Weighted MaxSAT instances where positive and negative literals are distinct nodes and clauses are hyperedges with clause weights as edge weights.",
        "problem_addressed": "Capturing higher-order dependencies in Weighted MaxSAT and addressing the limitation of prior factor graph representations which only model pairwise connections.",
        "evidence_location": "3.1 Hypergraph Modeling",
        "comment": "Explicitly treats complementary literals as separate nodes, differing from prior methods that merge variables into single nodes.",
        "paper_name": "HyperSAT Unsupervised Hypergraph Neural Networks f",
        "source_file": "2504.11885_HyperSAT_Unsupervised_Hypergraph_Neural_Networks_f.md",
        "contribution_index": 0,
        "cluster_name": "Graph Representation & Encoding"
      }
    ]
  },
  "Solver Integration & Application": {
    "canonical_name": "Solver Integration & Application",
    "member_phrases": [
      "Integration into a Solver",
      "Integration into a Solver / Algorithm",
      "Integration into a Solver Selection Framework",
      "Integration into a solver",
      "integration into a solver",
      "Neural Heuristic Integration",
      "Application Framework"
    ],
    "contributions": [
      {
        "contribution_type": "Neural Heuristic Integration",
        "specific_innovation": "Integration of a message-passing graph neural network with a modified attention mechanism as the branching heuristic (choose-literal) within DPLL and CDCL SAT solving algorithms.",
        "problem_addressed": "Enhancing the branching decision quality in backtracking-based SAT solvers to reduce the number of steps (and thus expected running time).",
        "evidence_location": "Section 1 Introduction, Section 3 Architecture",
        "comment": "",
        "paper_name": "Neural heuristics for SAT solving",
        "source_file": "2005.13406_Neural_heuristics_for_SAT_solving.md",
        "contribution_index": 0,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "Integrates the NeuroBack phase predictions into the state-of-the-art Kissat SAT solver via phase initialization.",
        "problem_addressed": "Provides a concrete and effective way to leverage neural predictions within an existing, high-performance SAT solving framework.",
        "evidence_location": "Section 4.3",
        "comment": "Creates the NeuroBack-Kissat solver.",
        "paper_name": "NeuroBack Improving CDCL SAT Solving using Graph N",
        "source_file": "2110.14053_NeuroBack_Improving_CDCL_SAT_Solving_using_Graph_N.md",
        "contribution_index": 1,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "Using a GNN-based model as an end-to-end approximate solver for the #SAT problem, outputting an estimation of the log-number of models directly from the input formula's factor graph.",
        "problem_addressed": "Providing a fast, learned alternative to traditional approximate #SAT solvers, without requiring distribution-specific hand-crafted heuristics.",
        "evidence_location": "Section 3: Method",
        "comment": "",
        "paper_name": "Graph Neural Networks for Propositional Model Coun",
        "source_file": "2205.04423_Graph_Neural_Networks_for_Propositional_Model_Coun.md",
        "contribution_index": 3,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "Integration of the learned branching heuristic into the exact dominating-clique solver of Culberson et al., replacing its MRV heuristic with the entropy-based selection.",
        "problem_addressed": "Demonstrates the practical application of the GNN-learned heuristic to enhance a state-of-the-art exact solver for an NP-complete problem.",
        "evidence_location": "Section 3.4 New Branching Heuristic for Dominating-Clique Solvers",
        "comment": "",
        "paper_name": "Learning Branching Heuristics from Graph Neural Ne",
        "source_file": "2211.14405_Learning_Branching_Heuristics_from_Graph_Neural_Ne.md",
        "contribution_index": 3,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Application Framework",
        "specific_innovation": "A framework for applying probabilistic-method GNNs to learn branching heuristics for exact solvers of constrained vertex subset problems.",
        "problem_addressed": "Provides a new way to use GNNs to enhance classical backtracking algorithms, moving beyond their typical use in greedy approximation or sampling.",
        "evidence_location": "Abstract, Section 5 Concluding Remarks",
        "comment": "This is the overarching high-level contribution of the paper.",
        "paper_name": "Learning Branching Heuristics from Graph Neural Ne",
        "source_file": "2211.14405_Learning_Branching_Heuristics_from_Graph_Neural_Ne.md",
        "contribution_index": 5,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a solver",
        "specific_innovation": "Integration of the learned functional similarity into a SAT sweeper for logic synthesis, guiding the selection of candidate equivalence classes to reduce invalid SAT solver calls.",
        "problem_addressed": "Inefficiency in traditional SAT-sweeping engines which select equivalence candidates based only on structure, without considering functional similarity.",
        "evidence_location": "Section V-A",
        "comment": "",
        "paper_name": "DeepGate2 Functionality-Aware Circuit Representati",
        "source_file": "2305.16373_DeepGate2_Functionality-Aware_Circuit_Representati.md",
        "contribution_index": 5,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a solver",
        "specific_innovation": "Integration of the learned gate functional similarity into a modern SAT solver (CaDiCaL) to guide variable decision, assigning reverse values to correlated variables to prompt conflicts.",
        "problem_addressed": "The reliance of prior SAT acceleration techniques on time-consuming logic simulation to obtain functional correlation.",
        "evidence_location": "Section V-B",
        "comment": "",
        "paper_name": "DeepGate2 Functionality-Aware Circuit Representati",
        "source_file": "2305.16373_DeepGate2_Functionality-Aware_Circuit_Representati.md",
        "contribution_index": 6,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "Two new oracle-based Stochastic Local Search (SLS) algorithms: an oracle-based Moser-Tardos (MT) algorithm and an oracle-based WalkSAT algorithm.",
        "problem_addressed": "Generalizes standard SLS algorithms (MT, WalkSAT) to leverage learned, instance-specific sampling oracles for variable updates, improving performance.",
        "evidence_location": "Section 2.2, Oracle-based SLS",
        "comment": "These algorithms (Algorithm 1 and 2) form the core framework into which the learned oracle is integrated.",
        "paper_name": "Using deep learning to construct stochastic local ",
        "source_file": "2309.11452_Using_deep_learning_to_construct_stochastic_local_.md",
        "contribution_index": 1,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "integration into a solver",
        "specific_innovation": "A post-processing inference step that uses a 2-improvement local search algorithm on the model's predictions to obtain discrete, constraint-satisfying feasible solutions for CO problems with hard clauses.",
        "problem_addressed": "The need to convert the model's continuous predictions into valid discrete solutions that satisfy the hard constraints of the original CO problem (e.g., MIS, MDS).",
        "evidence_location": "Section 3.5 Inference with Local Search Method",
        "comment": "This bridges the gap between the neural network's output and a valid combinatorial solution.",
        "paper_name": "A unified pre-training and adaptation framework fo",
        "source_file": "2312.11547_A_unified_pre-training_and_adaptation_framework_fo.md",
        "contribution_index": 3,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "A one-time, offline interaction method is proposed where the network's predicted UNSAT-core probabilities are used to initialize the CDCL solver's decision queue and decision scores before solving begins.",
        "problem_addressed": "Previous neural approaches either solved problems end-to-end (unreliable) or interacted periodically with the solver, which is inefficient for the high-frequency UNSAT problems in LEC.",
        "evidence_location": "3.5 Interaction with Solver",
        "comment": "",
        "paper_name": "IB-Net Initial Branch Network for Variable Decisio",
        "source_file": "2403.03517_IB-Net_Initial_Branch_Network_for_Variable_Decisio.md",
        "contribution_index": 2,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver Selection Framework",
        "specific_innovation": "A complete workflow (GraSS) is presented, from graph construction and feature engineering through GNN processing to solver prediction.",
        "problem_addressed": "There was no GNN-based method for automatic SAT solver selection that fully leverages instance structure and domain knowledge.",
        "evidence_location": "Section 3 APPROACH",
        "comment": "This is the overarching system contribution.",
        "paper_name": "GraSS Combining Graph Neural Networks with Expert ",
        "source_file": "2405.11024_GraSS_Combining_Graph_Neural_Networks_with_Expert_.md",
        "contribution_index": 5,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a solver",
        "specific_innovation": "A decimation procedure inspired by Belief Propagation that fixes variables with extreme embeddings, simplifies the formula, and re-runs the GNN.",
        "problem_addressed": "Further increasing the percentage of solved problems by iteratively assigning variables and simplifying the problem.",
        "evidence_location": "Section 4: Sampling and Decimation",
        "comment": "",
        "paper_name": "Understanding GNNs for Boolean Satisfiability thro",
        "source_file": "2408.15418_Understanding_GNNs_for_Boolean_Satisfiability_thro.md",
        "contribution_index": 2,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver / Algorithm",
        "specific_innovation": "Extension of the base GNN into a diffusion model for incremental sampling, and its further combination with classical unit propagation in a tree-search-like algorithm to fix variable assignments.",
        "problem_addressed": "Provides another mechanism for test-time compute scaling and improves solution quality by integrating neural predictions with deterministic logical deduction.",
        "evidence_location": "Section 5.4 (Diffusion Model Extension), Section 5.4.2 (Interleaving Diffusion Steps with Unit Propagation), Appendix A.3",
        "comment": "The diffusion model allows flexible inference steps. The hybrid algorithm with unit propagation shows an additional ~10% accuracy improvement (Table 7).",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 4,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "HyperSAT, an end-to-end unsupervised neural framework that integrates hypergraph modeling, the novel architecture, and the multi-objective loss to solve Weighted MaxSAT.",
        "problem_addressed": "Providing a learning-based method for the underdeveloped area of Weighted MaxSAT solvers, specifically handling non-uniform clause weights.",
        "evidence_location": "3 HyperSAT",
        "comment": "This is the overall framework contribution, as stated in the abstract and contributions list.",
        "paper_name": "HyperSAT Unsupervised Hypergraph Neural Networks f",
        "source_file": "2504.11885_HyperSAT_Unsupervised_Hypergraph_Neural_Networks_f.md",
        "contribution_index": 4,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "Proposes a generic mechanism to inject inferred variable weights and polarities into the branching heuristics of existing SAT solvers by scaling the variable scoring function and setting the branching sign.",
        "problem_addressed": "Prior methods for integrating ML guidance into SAT solvers were often solver-specific or required expensive periodic updates (e.g., resetting VSIDS scores).",
        "evidence_location": "Section 1, Section 2.1, Algorithm 3",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 1,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a solver",
        "specific_innovation": "Dynamically integrates GNN-predicted conditional probabilities into the CDCL solver's inprocessing stage to guide variable phase selection based on a threshold rule.",
        "problem_addressed": "Improves solver efficiency by biasing variable assignments toward values more likely to satisfy the circuit, significantly reducing solving time for SAT instances.",
        "evidence_location": "3.1 Problem Definition (Phase Selection), 5.2 Node Probability for Phase Selection",
        "comment": "",
        "paper_name": "Circuit-Aware SAT Solving Guiding CDCL via Conditi",
        "source_file": "2508.04235_Circuit-Aware_SAT_Solving_Guiding_CDCL_via_Conditi.md",
        "contribution_index": 3,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a solver",
        "specific_innovation": "Introduces a novel clause quality metric, 'clause probability', and uses it to filter learned clauses during database elimination in the CDCL solver.",
        "problem_addressed": "Provides a circuit-aware metric superior to CNF-level heuristics (like LBD) for identifying and retaining high-quality, structurally informative clauses, improving performance on UNSAT instances.",
        "evidence_location": "3.1 Problem Definition (Clause Management), 5.3 Clause Probability for Clause Management",
        "comment": "",
        "paper_name": "Circuit-Aware SAT Solving Guiding CDCL via Conditi",
        "source_file": "2508.04235_Circuit-Aware_SAT_Solving_Guiding_CDCL_via_Conditi.md",
        "contribution_index": 4,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "The integration of the Iterative Join-Graph Propagation (IJGP) algorithm into a neural network framework to serve as the underlying inference engine for approximate model counting.",
        "problem_addressed": "Replaces the standard belief propagation (BP) algorithm to avoid redundant message passing on loopy structures, thereby improving solution accuracy and enabling more flexible graph structure control via tree-width.",
        "evidence_location": "Sections 1, 3.1, 3.3",
        "comment": "This is a key algorithmic choice that differentiates the work from predecessors like NSNet (which uses BP).",
        "paper_name": "Attn-JGNN Attention Enhanced Join-Graph Neural Net",
        "source_file": "2510.15583_Attn-JGNN_Attention_Enhanced_Join-Graph_Neural_Net.md",
        "contribution_index": 1,
        "cluster_name": "Solver Integration & Application"
      },
      {
        "contribution_type": "Integration into a Solver",
        "specific_innovation": "The adaptation of the Bethe free energy approximation to the join-graph structure (Bethe-Join) and its estimation via an MLP to approximate the partition function (log Z).",
        "problem_addressed": "Provides the final step to translate the neural network's message-passing outputs into an approximate count of satisfying assignments (#SAT solution).",
        "evidence_location": "Section 3.3",
        "comment": "This completes the pipeline from formula encoding to count estimation.",
        "paper_name": "Attn-JGNN Attention Enhanced Join-Graph Neural Net",
        "source_file": "2510.15583_Attn-JGNN_Attention_Enhanced_Join-Graph_Neural_Net.md",
        "contribution_index": 5,
        "cluster_name": "Solver Integration & Application"
      }
    ]
  },
  "Theoretical Analysis & Framework": {
    "canonical_name": "Theoretical Analysis & Framework",
    "member_phrases": [
      "Connection to Phase Transitions",
      "Interpretation/Theoretical Connection",
      "Theoretical Framework Integration",
      "Theoretical Identification/Problem Statement",
      "Theoretical Proof/Framework",
      "Interpretability Framework / Analysis",
      "Characterization of Average Curvature"
    ],
    "contributions": [
      {
        "contribution_type": "Theoretical Proof/Framework",
        "specific_innovation": "A theoretical proof that a single-layer GNN can achieve a 1/2 approximation ratio for MaxSAT by aligning with a novel distributed local algorithm.",
        "problem_addressed": "Lack of theoretical understanding of why GNNs can solve NP-hard problems like MaxSAT.",
        "evidence_location": "Section 4: Theoretical Analysis",
        "comment": "Applies algorithmic alignment theory to MaxSAT for the first time.",
        "paper_name": "Can Graph Neural Networks Learn to Solve MaxSAT Pr",
        "source_file": "2111.07568_Can_Graph_Neural_Networks_Learn_to_Solve_MaxSAT_Pr.md",
        "contribution_index": 1,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Theoretical Identification/Problem Statement",
        "specific_innovation": "Identified that concurrent prediction in GNN-based SAT solvers cannot solve symmetric problems requiring asymmetric assignments due to neglecting variable dependency.",
        "problem_addressed": "Highlights a fundamental limitation in existing end-to-end GNN-based SAT solving methods.",
        "evidence_location": "Section 3: Variable Dependency in SAT Solving",
        "comment": "",
        "paper_name": "Addressing Variable Dependency in GNN-based SAT So",
        "source_file": "2304.08738_Addressing_Variable_Dependency_in_GNN-based_SAT_So.md",
        "contribution_index": 5,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Theoretical Framework Integration",
        "specific_innovation": "Bridging theoretical computer science results (Lovász Local Lemma, Moser-Tardos algorithm guarantees) with deep learning for constructing SAT solvers.",
        "problem_addressed": "Motivates and provides a foundation for the design of the loss function and solver framework, aiming for solvers with performance bounds.",
        "evidence_location": "Section 2.3, Performance guarantees for the MT algorithm",
        "comment": "This integration is the paper's foundational conceptual contribution, linking theory to practice.",
        "paper_name": "Using deep learning to construct stochastic local ",
        "source_file": "2309.11452_Using_deep_learning_to_construct_stochastic_local_.md",
        "contribution_index": 5,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Interpretation/Theoretical Connection",
        "specific_innovation": "Established an empirical and conceptual connection between GNN message-passing for SAT and two classical approximation algorithms: Semidefinite Programming (SDP) relaxations and Belief Propagation.",
        "problem_addressed": "Lack of interpretability for how GNNs solve SAT problems, enabling the design of improved training and inference methods.",
        "evidence_location": "Abstract, Sections 2.3, 2.4, 3, 4, 5.2.1",
        "comment": "This is the core conceptual contribution motivating the technical improvements.",
        "paper_name": "Understanding GNNs for Boolean Satisfiability thro",
        "source_file": "2408.15418_Understanding_GNNs_for_Boolean_Satisfiability_thro.md",
        "contribution_index": 4,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Interpretability Framework / Analysis",
        "specific_innovation": "Analysis through the lens of continuous optimization, interpreting the GNN's message-passing as implicitly performing gradient-based optimization on a continuous relaxation of the MaxSAT objective, similar to SDP relaxations.",
        "problem_addressed": "Addresses the 'black-box' nature of neural SAT solvers by providing an intuitive, algorithmically-grounded explanation for their reasoning process and generalization ability.",
        "evidence_location": "Section 6 (Interpreting the Trained Model), Section 7 (Discussion)",
        "comment": "This is a core contribution aimed at understanding *how* the GNNs work. Evidence includes embedding space clustering (Figure 5) and gap reduction trajectories (Figure 3).",
        "paper_name": "Neural Approaches to SAT Solving Design Choices an",
        "source_file": "2504.01173_Neural_Approaches_to_SAT_Solving_Design_Choices_an.md",
        "contribution_index": 5,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Theoretical Proof/Framework",
        "specific_innovation": "Provides an information-theoretic proof that integrating additional (structural) priors into a generative model does not lower its performance upper bound and can enhance it.",
        "problem_addressed": "Lack of theoretical justification for why incorporating structural priors should benefit LLM-driven algorithm discovery for CO problems.",
        "evidence_location": "4.2 Theoretical Analysis",
        "comment": "",
        "paper_name": "STRCMP Integrating Graph Structural Priors with La",
        "source_file": "2506.11057_STRCMP_Integrating_Graph_Structural_Priors_with_La.md",
        "contribution_index": 4,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Theoretical Proof/Framework",
        "specific_innovation": "Proves that bipartite graphs from random k-SAT formulas are inherently negatively curved, with curvature decreasing with problem difficulty.",
        "problem_addressed": "Provides a geometric explanation for the performance degradation of GNN-based SAT solvers on harder instances.",
        "evidence_location": "Section 3: Curvature of Random k-SAT Problems and Its Relationship with GNNs",
        "comment": "",
        "paper_name": "On the Hardness of Learning GNN-based SAT Solvers ",
        "source_file": "2508.21513_On_the_Hardness_of_Learning_GNN-based_SAT_Solvers_.md",
        "contribution_index": 0,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Characterization of Average Curvature",
        "specific_innovation": "Derives an exact expression for the average Balanced Forman Curvature (BFC) in the limit of unsolvable random k-SAT problems.",
        "problem_addressed": "Quantifies how average curvature depends on clause density (α) and clause size (k), linking structural properties to solver difficulty.",
        "evidence_location": "Section 3: Characterizing Average Curvature (Theorem 3.1)",
        "comment": "",
        "paper_name": "On the Hardness of Learning GNN-based SAT Solvers ",
        "source_file": "2508.21513_On_the_Hardness_of_Learning_GNN-based_SAT_Solvers_.md",
        "contribution_index": 4,
        "cluster_name": "Theoretical Analysis & Framework"
      },
      {
        "contribution_type": "Connection to Phase Transitions",
        "specific_innovation": "Empirically shows a phase-transition-like phenomenon in solver success probability as a function of the mean and variance of curvature.",
        "problem_addressed": "Relates geometric properties (curvature) to the well-known SAT/UNSAT algorithmic phase transition, providing a new geometric order parameter.",
        "evidence_location": "Section 4.1: The Relationship Between Curvature and Satisfiability (Figure 2)",
        "comment": "",
        "paper_name": "On the Hardness of Learning GNN-based SAT Solvers ",
        "source_file": "2508.21513_On_the_Hardness_of_Learning_GNN-based_SAT_Solvers_.md",
        "contribution_index": 5,
        "cluster_name": "Theoretical Analysis & Framework"
      }
    ]
  },
  "Empirical & Behavioral Analysis": {
    "canonical_name": "Empirical & Behavioral Analysis",
    "member_phrases": [
      "Ablation Study",
      "Analysis of Limitation",
      "Behavioral Analysis and Insight",
      "Empirical Analysis",
      "Experimental Analysis"
    ],
    "contributions": [
      {
        "contribution_type": "Experimental Analysis",
        "specific_innovation": "Systematic evaluation of model generalization across different instance sizes (scaling), clause-to-variable ratios, and clause lengths (k).",
        "problem_addressed": "Assessing the practical potential and limits of GNNs for MaxSAT, including their ability to generalize to harder instances.",
        "evidence_location": "Section 5.5: Generalizing to Other Distributions",
        "comment": "This is an analysis contribution rather than a new method.",
        "paper_name": "Can Graph Neural Networks Learn to Solve MaxSAT Pr",
        "source_file": "2111.07568_Can_Graph_Neural_Networks_Learn_to_Solve_MaxSAT_Pr.md",
        "contribution_index": 6,
        "cluster_name": "Empirical & Behavioral Analysis"
      },
      {
        "contribution_type": "Ablation Study",
        "specific_innovation": "Empirical comparison of global vs. local search (ANYCSP_loc) and different reward schemes (ANYCSP_qual) to validate the design choices.",
        "problem_addressed": "To isolate and demonstrate the importance of the global search action space and the proposed reward scheme.",
        "evidence_location": "Section C (Ablation)",
        "comment": "ANYCSP_loc restricts to single-variable changes; ANYCSP_qual uses direct quality as reward; results show both components are crucial.",
        "paper_name": "One Model Any CSP Graph Neural Networks as Fast Gl",
        "source_file": "2208.10227_One_Model_Any_CSP_Graph_Neural_Networks_as_Fast_Gl.md",
        "contribution_index": 5,
        "cluster_name": "Empirical & Behavioral Analysis"
      },
      {
        "contribution_type": "Behavioral Analysis and Insight",
        "specific_innovation": "Through comparative analysis, it is found that GNN-based solvers learn a solving strategy akin to greedy local search (simultaneously flipping many variables) but fail to learn the backtracking/CDCL heuristic in the latent space.",
        "problem_addressed": "The need to understand what heuristics GNNs implicitly learn and how they compare to traditional SAT solver strategies.",
        "evidence_location": "Section 6",
        "comment": "Key findings include GNNs' inability to benefit from clause-augmented instances without explicit training on them, and their convergence behavior resembling GSAT.",
        "paper_name": "G4SATBench Benchmarking and Advancing SAT Solving ",
        "source_file": "2309.16941_G4SATBench_Benchmarking_and_Advancing_SAT_Solving_.md",
        "contribution_index": 4,
        "cluster_name": "Empirical & Behavioral Analysis"
      },
      {
        "contribution_type": "Empirical Analysis",
        "specific_innovation": "Analyzes the correlation of learned variable weights between policies trained with different solvers, showing they capture solver-agnostic structural properties.",
        "problem_addressed": "Provides insight into what the learned policies capture and how they relate to known variable properties (e.g., UNSAT core membership).",
        "evidence_location": "Section 3.3, Figure 5",
        "comment": "",
        "paper_name": "Learning from Algorithm Feedback One-Shot SAT Solv",
        "source_file": "2505.16053_Learning_from_Algorithm_Feedback_One-Shot_SAT_Solv.md",
        "contribution_index": 7,
        "cluster_name": "Empirical & Behavioral Analysis"
      },
      {
        "contribution_type": "Analysis of Limitation",
        "specific_innovation": "Establishes a direct link between negative graph Ricci curvature and the oversquashing phenomenon in GNNs for SAT solving.",
        "problem_addressed": "Explains why GNNs struggle with long-range dependencies in SAT instances, identifying a representational learning bottleneck separate from algorithmic hardness.",
        "evidence_location": "Section 3.1: Message-Passing Bottlenecks and Downstream Performance",
        "comment": "",
        "paper_name": "On the Hardness of Learning GNN-based SAT Solvers ",
        "source_file": "2508.21513_On_the_Hardness_of_Learning_GNN-based_SAT_Solvers_.md",
        "contribution_index": 1,
        "cluster_name": "Empirical & Behavioral Analysis"
      }
    ]
  }
};
