###########################
# app.py - Streamlit App
###########################

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import random
import string
import time

# --------------------------------------------
# Global Constants & Data
# --------------------------------------------

# A dictionary of baseline conscious entities for comparison
# Key: Name of entity
# Value: A dictionary of "scores" or param-values that we can feed into our model as references
BASE_ENTITIES = {
    "Human (average adult)": {
        "neurons": 86e9,
        "connectivity": 0.7,
        "synaptic_fidelity": 0.95,
        "processing_speed": 1.0,
        "plasticity": 0.9,
        "self_recognition": True,
        "memory_persistence": True,
        "intentionality": True,
        "pain_pleasure_sim": True,
        "iit_importance": 0.8,
        "gwt_importance": 0.8
    },
    "Chimpanzee": {
        "neurons": 6.2e9,
        "connectivity": 0.65,
        "synaptic_fidelity": 0.9,
        "processing_speed": 1.0,
        "plasticity": 0.85,
        "self_recognition": True,
        "memory_persistence": True,
        "intentionality": True,
        "pain_pleasure_sim": True,
        "iit_importance": 0.75,
        "gwt_importance": 0.7
    },
    "Flatworm (Planarian)": {
        "neurons": 10000,
        "connectivity": 0.4,
        "synaptic_fidelity": 0.5,
        "processing_speed": 0.5,
        "plasticity": 0.2,
        "self_recognition": False,
        "memory_persistence": False,
        "intentionality": False,
        "pain_pleasure_sim": False,
        "iit_importance": 0.3,
        "gwt_importance": 0.2
    },
    "GPT-4-like Language Model": {
        "neurons": 0,  # Not directly comparable, so we keep 0 or an arbitrary value
        "connectivity": 0.9,
        "synaptic_fidelity": 0.0,
        "processing_speed": 1000,
        "plasticity": 0.0,
        "self_recognition": False,
        "memory_persistence": False,
        "intentionality": False,
        "pain_pleasure_sim": False,
        "iit_importance": 0.1,
        "gwt_importance": 0.1
    },
    "Hypothetical Hyper-Intelligent AI": {
        "neurons": 1e12,
        "connectivity": 0.95,
        "synaptic_fidelity": 1.0,
        "processing_speed": 10.0,
        "plasticity": 1.0,
        "self_recognition": True,
        "memory_persistence": True,
        "intentionality": True,
        "pain_pleasure_sim": True,
        "iit_importance": 0.95,
        "gwt_importance": 0.95
    },
}

# Some preset philosophical frameworks and their weighting factors
PHILOSOPHICAL_FRAMEWORKS = {
    "Integrated Information Theory (IIT)": {
        "weight_neuron_count": 0.2,
        "weight_connectivity": 0.4,
        "weight_syn_fidelity": 0.3,
        "weight_speed": 0.1
    },
    "Global Workspace Theory (GWT)": {
        "weight_connectivity": 0.3,
        "weight_plasticity": 0.4,
        "weight_broadcast": 0.3
    },
    "Functionalism": {
        "weight_functionality": 0.7,
        "weight_biological": 0.0,  # functionalism doesn't care about substrate
        "weight_experience": 0.3
    },
    "Biological Essentialism": {
        "weight_functionality": 0.2,
        "weight_biological": 0.8,  # must be 'biological' or extremely close to it
    }
}

# --------------------------------------------
# Helper Functions
# --------------------------------------------

def compute_iit_score(neuron_count, connectivity, synaptic_fidelity, processing_speed):
    """
    A mock function that calculates an IIT-like 'Phi' measure based on
    neuron count, connectivity, synaptic fidelity, and processing speed.
    """
    # For demonstration, let's do something non-trivial but partly random
    base_score = np.log10(neuron_count + 1) * connectivity * synaptic_fidelity
    speed_factor = (processing_speed ** 0.1)  # diminishing returns at high speeds
    phi = base_score * speed_factor

    # Normalize to range [0..1] with an arbitrary scale
    max_phi_est = 100  # arbitrary normalization
    phi_score = min(phi / max_phi_est, 1.0)
    return phi_score

def compute_gwt_score(connectivity, plasticity, broadcast_power):
    """
    A simplified GWT scoring approach:
    GWT says consciousness arises when info is broadcast across modules.
    """
    # Weighted product approach
    score = (connectivity ** 0.5) * (plasticity ** 0.5) * (broadcast_power ** 0.5)
    # Some normalization factor
    return min(score * 1.2, 1.0)

def compute_functionalism_score(self_recognition, memory_persistence,
                                intentionality, pain_pleasure_sim):
    """
    A mock function that evaluates how functionally 'human-like' or 
    'conscious-like' the entity is, ignoring substrate.
    """
    # Let's treat each boolean as a half point
    total_traits = 4
    active_traits = sum([self_recognition, memory_persistence, intentionality, pain_pleasure_sim])
    fraction = active_traits / total_traits
    return fraction  # range 0..1

def compute_biological_essentialism_score(neuron_count, synaptic_fidelity):
    """
    Biological essentialism might focus on the 'authenticity' of the
    biological substrate or near-perfect replication. 
    """
    # If fidelity < 0.5, the system is seen as not biologically real enough
    # Let's do a simple approach: 
    #   more neurons + high fidelity => more "biologically valid"
    #   otherwise => lower
    if synaptic_fidelity < 0.3:
        return 0.0
    # simple approach:
    scale = np.log10(neuron_count + 1) / 10.0  # normalizing factor
    score = scale * synaptic_fidelity
    return min(score, 1.0)

def bayesian_probability(iit_score, gwt_score, functionalism_score, bio_ess_score):
    """
    Very simplified Bayesian-like aggregator. Each dimension represents
    evidence supporting consciousness. We'll treat them as partially 
    independent (not truly, but for demonstration).
    """
    # We'll just combine these with an arbitrary approach:
    # p(conscious) = 1 - Product(1 - subscore)
    # subscore can be weighted differently if we want to reflect certain frameworks
    sub_scores = [iit_score, gwt_score, functionalism_score, bio_ess_score]
    p_not_conscious = 1.0
    for s in sub_scores:
        p_not_conscious *= (1 - s)
    return 1 - p_not_conscious

def build_random_graph(n_nodes, connectivity):
    """
    Build a random directed graph with 'n_nodes' and approximate 
    'connectivity' fraction of edges present.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    total_possible_edges = n_nodes*(n_nodes-1)
    edges_to_add = int(connectivity * total_possible_edges)

    edges = []
    while len(edges) < edges_to_add:
        src = np.random.randint(0, n_nodes)
        dst = np.random.randint(0, n_nodes)
        if src != dst and (src, dst) not in edges:
            edges.append((src, dst))

    G.add_edges_from(edges)
    return G

def highlight_graph(G, ax, title):
    """
    Render a networkx graph on a matplotlib axis for demonstration.
    """
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=10,
                           edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    ax.set_title(title)
    ax.axis('off')

# --------------------------------------------
# Streamlit App Layout
# --------------------------------------------

def main():
    st.set_page_config(
        page_title="Digital Consciousness Probability Estimator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Digital Consciousness Probability Estimator")
    st.write("""
    This sophisticated app calculates the probability that a Whole Brain Emulation (WBE)
    or other entity is conscious. Adjust **neuroscientific**, **phenomenological**, 
    and **philosophical** parameters, then see how likely the entity is to have a 
    genuine subjective experience.
    """)

    # We'll keep some results in session state to allow for "History"
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar for user input
    with st.sidebar:
        st.header("Entity Parameters")
        st.write("## Neuroscientific/Computational")
        neuron_count = st.number_input(
            "Neuron Count (approx.)",
            min_value=0.0, max_value=1e12, value=86e9, step=1e6,
            help="Number of neurons or neuron-like units in the simulated brain."
        )
        connectivity = st.slider(
            "Connectivity (0 = no interconnect, 1 = fully connected)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.01
        )
        synaptic_fidelity = st.slider(
            "Synaptic Fidelity (0 = extremely lossy, 1 = perfect)",
            min_value=0.0, max_value=1.0, value=0.9, step=0.01
        )
        processing_speed = st.slider(
            "Processing Speed Factor (relative to human baseline)",
            min_value=0.1, max_value=100.0, value=1.0, step=0.1
        )
        plasticity = st.slider(
            "Plasticity / Learning Capacity (0-1)",
            min_value=0.0, max_value=1.0, value=0.8, step=0.01
        )

        st.write("## Phenomenological")
        self_recognition = st.checkbox("Self Recognition (Mirror Test)", True)
        memory_persistence = st.checkbox("Long-Term Memory Persistence", True)
        intentionality = st.checkbox("Displays Goals/Intentions", True)
        pain_pleasure_sim = st.checkbox("Can Experience Pain/Pleasure", True)

        st.write("## Philosophical Weights")
        # Let user choose frameworks or define custom weighting
        use_iit = st.checkbox("Use Integrated Information Theory", True)
        use_gwt = st.checkbox("Use Global Workspace Theory", True)
        use_functionalism = st.checkbox("Use Functionalism", True)
        use_bio_ess = st.checkbox("Use Biological Essentialism", True)

    # We also allow user to pick a baseline entity for comparison
    st.write("### Compare to a Baseline Entity")
    baseline_choice = st.selectbox(
        "Choose an entity to compare",
        list(BASE_ENTITIES.keys()),
        index=0
    )

    st.markdown("---")

    # Compute scores
    iit_score = compute_iit_score(neuron_count, connectivity, synaptic_fidelity, processing_speed) if use_iit else 0.0
    # For GWT, let's define a "broadcast power" as (processing_speed) * 0.5, just for fun
    broadcast_power = max(min(processing_speed/10, 1.0), 0.1)
    gwt_score = compute_gwt_score(connectivity, plasticity, broadcast_power) if use_gwt else 0.0
    functionalism_score = compute_functionalism_score(
        self_recognition, memory_persistence, intentionality, pain_pleasure_sim
    ) if use_functionalism else 0.0
    bio_ess_score = compute_biological_essentialism_score(neuron_count, synaptic_fidelity) if use_bio_ess else 0.0

    consciousness_probability = bayesian_probability(
        iit_score,
        gwt_score,
        functionalism_score,
        bio_ess_score
    )

    col1, col2, col3 = st.columns([0.7, 1.0, 1.0])

    with col1:
        st.subheader("Calculated Probability of Consciousness")
        st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{consciousness_probability*100:.1f}%</h1>", unsafe_allow_html=True)

        if st.button("Save This Result to History"):
            record = {
                "params": {
                    "neuron_count": neuron_count,
                    "connectivity": connectivity,
                    "synaptic_fidelity": synaptic_fidelity,
                    "processing_speed": processing_speed,
                    "plasticity": plasticity,
                    "self_recognition": self_recognition,
                    "memory_persistence": memory_persistence,
                    "intentionality": intentionality,
                    "pain_pleasure_sim": pain_pleasure_sim,
                    "use_iit": use_iit,
                    "use_gwt": use_gwt,
                    "use_functionalism": use_functionalism,
                    "use_bio_ess": use_bio_ess
                },
                "prob": consciousness_probability,
                "time": time.ctime()
            }
            st.session_state.history.append(record)
            st.success("Result saved!")

    # Show each sub-score in col2
    with col2:
        st.subheader("Sub-Scores")
        st.write(f"**IIT Score:** {iit_score*100:.1f}% (if used)") 
        st.write(f"**GWT Score:** {gwt_score*100:.1f}% (if used)") 
        st.write(f"**Functionalism Score:** {functionalism_score*100:.1f}% (if used)") 
        st.write(f"**Biological Essentialism Score:** {bio_ess_score*100:.1f}% (if used)") 

    # Compare to baseline in col3
    with col3:
        st.subheader(f"Comparison: {baseline_choice}")
        base_data = BASE_ENTITIES[baseline_choice]
        
        # Compute baseline scores
        base_iit = compute_iit_score(
            base_data["neurons"],
            base_data["connectivity"],
            base_data["synaptic_fidelity"],
            base_data["processing_speed"]
        ) if use_iit else 0.0
        
        # GWT
        base_broadcast = max(min(base_data["processing_speed"]/10, 1.0), 0.1)
        base_gwt = compute_gwt_score(
            base_data["connectivity"], 
            base_data["plasticity"],
            base_broadcast
        ) if use_gwt else 0.0
        
        # Functionalism
        base_func = compute_functionalism_score(
            base_data["self_recognition"],
            base_data["memory_persistence"],
            base_data["intentionality"],
            base_data["pain_pleasure_sim"]
        ) if use_functionalism else 0.0
        
        # Bio Ess
        base_bio = compute_biological_essentialism_score(
            base_data["neurons"],
            base_data["synaptic_fidelity"]
        ) if use_bio_ess else 0.0

        base_prob = bayesian_probability(base_iit, base_gwt, base_func, base_bio)
        st.markdown(f"<h2 style='text-align: center; color: #4BB543;'>{base_prob*100:.1f}%</h2>", unsafe_allow_html=True)
        st.caption("Probability that this baseline entity is conscious (based on current chosen frameworks).")

    st.markdown("---")
    st.subheader("Graphical Demonstrations")
    tab1, tab2 = st.tabs(["IIT Network Simulation", "History & Charts"])

    with tab1:
        st.write("### Example Random Network to Demonstrate ‘Integration’")
        # Create a random graph
        n_nodes = min(int(np.log10(neuron_count+1)), 50)  # We'll keep up to 50 nodes for display
        G = build_random_graph(n_nodes, connectivity)
        # Render in matplotlib
        fig, ax = plt.subplots(figsize=(5, 4))
        highlight_graph(G, ax, "Random Graph Approx. Representation")
        st.pyplot(fig)

        st.write("""
        > **Note**: In a real setting, we'd map the actual brain connectome. 
        > Here we show a random network with approximate connectivity. 
        > As connectivity & fidelity increase, it’s likelier to have a higher integrated information score.
        """)

    with tab2:
        st.write("## Submission History & Probability Over Time")
        # Turn session history into a DataFrame
        history_df = pd.DataFrame([
            {
                "time": record["time"],
                "neuron_count": record["params"]["neuron_count"],
                "prob": record["prob"]
            }
            for record in st.session_state.history
        ])

        if len(history_df) > 0:
            history_df["prob_percent"] = history_df["prob"] * 100
            st.dataframe(history_df)

            fig_line = px.line(history_df, x="time", y="prob_percent", title="Probability Over Time")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No history saved yet. Click 'Save This Result to History' to log a scenario.")

    # Footer
    st.markdown("---")
    st.write("### About This App")
    st.write("""
    **Digital Consciousness Probability Estimator** 
    is a highly *experimental* tool that integrates simplified versions of 
    Integrated Information Theory (IIT), Global Workspace Theory (GWT), 
    functionalist approaches, and biological essentialism.
    
    **Disclaimer**: The displayed probabilities are not meant to be 
    definitive scientific conclusions. Rather, they serve as a 
    demonstration of how different assumptions and frameworks can 
    dramatically alter one's estimate of digital consciousness.
    """)

    # Potential for PDF export, advanced features, etc.
    st.write("For more rigorous usage, incorporate real connectome data, validated theories, and formal Bayesian modeling.")


# Run the app
if __name__ == "__main__":
    main()
